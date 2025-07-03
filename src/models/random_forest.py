"""
Random Forest model implementation for multi-class credit risk assessment.
Includes feature importance analysis, hyperparameter tuning, and out-of-bag scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import yaml

logger = logging.getLogger(__name__)


class RandomForestCreditRiskModel:
    """
    Random Forest model for multi-class credit risk assessment.
    Supports feature importance analysis, hyperparameter tuning, and comprehensive evaluation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Random Forest model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['random_forest']
        self.training_config = self.config['training']
        self.risk_config = self.config['risk_categories']
        
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.best_params = None
        self.feature_importance_ = None
        
        # Initialize with default parameters
        self.params = self.model_config['params'].copy()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights for cost-sensitive learning
        """
        logger.info("Training Random Forest model...")
        
        self.feature_names = list(X_train.columns)
        
        # Initialize model with parameters
        self.model = RandomForestClassifier(**self.params, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        self.is_trained = True
        self._calculate_feature_importance()
        
        logger.info("Random Forest model training completed!")
        
        # Log model information
        logger.info(f"Number of trees: {self.model.n_estimators}")
        logger.info(f"Number of classes: {len(self.model.classes_)}")
        logger.info(f"Classes: {self.model.classes_}")
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_:
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features for prediction
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           n_trials: int = 100, timeout: int = 3600) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Best parameters found
        """
        logger.info("Starting hyperparameter tuning for Random Forest...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
            }
            
            # Only enable oob_score if bootstrap=True
            if params['bootstrap']:
                params['oob_score'] = trial.suggest_categorical('oob_score', [True, False])
            else:
                params['oob_score'] = False
            
            # Cross-validation
            cv_folds = self.training_config['cv_folds']
            
            try:
                temp_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(
                    temp_model, X_train, y_train, 
                    cv=cv_folds, 
                    scoring=self.training_config['cv_scoring']
                )
                return scores.mean()
            except Exception as e:
                logger.warning(f"Trial failed with params {params}: {str(e)}")
                return 0.0
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        
        # Update model parameters
        self.params.update(self.best_params)
        
        logger.info(f"Hyperparameter tuning completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation scores
        """
        logger.info("Performing cross-validation...")
        
        # Create temporary model for CV
        temp_model = RandomForestClassifier(**self.params, random_state=42, n_jobs=-1)
        
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate multiple metrics
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_scores = {}
        
        for metric in metrics:
            scores = cross_val_score(temp_model, X, y, cv=cv, scoring=metric)
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        logger.info("Cross-validation completed!")
        return cv_scores
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        if self.model is None:
            return
        
        # Get feature importance from Random Forest
        importance_scores = self.model.feature_importances_
        
        # Create DataFrame
        self.feature_importance_ = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Feature importance DataFrame
        """
        if self.feature_importance_ is None:
            self._calculate_feature_importance()
        
        return self.feature_importance_.head(top_n)
    
    def get_tree_feature_importance_std(self) -> pd.DataFrame:
        """
        Get feature importance with standard deviation across trees.
        
        Returns:
            Feature importance with std across trees
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get importance from each tree
        importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        
        # Calculate mean and std
        importance_mean = np.mean(importances, axis=0)
        importance_std = np.std(importances, axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': importance_mean,
            'importance_std': importance_std
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score if available.
        
        Returns:
            OOB score or None if not available
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        else:
            return None
    
    def get_tree_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from individual trees.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions from each tree
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions from each tree
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T
        
        return tree_predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'best_params': self.best_params,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.feature_importance_ = model_data['feature_importance']
        self.best_params = model_data.get('best_params')
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        evaluation = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        # Add OOB score if available
        oob_score = self.get_oob_score()
        if oob_score is not None:
            evaluation['oob_score'] = oob_score
        
        return evaluation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': 'Random Forest',
            'parameters': self.params,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'num_classes': len(self.model.classes_),
            'classes': self.model.classes_.tolist(),
            'best_hyperparameters': self.best_params,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }
        
        # Add OOB score if available
        oob_score = self.get_oob_score()
        if oob_score is not None:
            info['oob_score'] = oob_score
        
        if self.feature_importance_ is not None:
            info['top_features'] = self.feature_importance_.head(10).to_dict('records')
        
        return info


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                             n_informative=15, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Initialize and train model
    model = RandomForestCreditRiskModel()
    
    # Tune hyperparameters (small example)
    best_params = model.tune_hyperparameters(X_train, y_train, n_trials=10)
    print(f"Best parameters: {best_params}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    evaluation = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {evaluation['accuracy']:.4f}")
    print(f"Test F1-macro: {evaluation['f1_macro']:.4f}")
    if 'oob_score' in evaluation:
        print(f"OOB Score: {evaluation['oob_score']:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    print("\nTop 10 features:")
    print(importance)
    
    # Feature importance with std
    importance_std = model.get_tree_feature_importance_std().head(5)
    print("\nTop 5 features with std across trees:")
    print(importance_std)