"""
Logistic Regression model implementation for multi-class credit risk assessment.
Includes regularization, feature selection, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import yaml

logger = logging.getLogger(__name__)


class LogisticRegressionCreditRiskModel:
    """
    Logistic Regression model for multi-class credit risk assessment.
    Supports regularization, hyperparameter tuning, and comprehensive evaluation.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Logistic Regression model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['logistic_regression']
        self.training_config = self.config['training']
        self.risk_config = self.config['risk_categories']
        
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.best_params = None
        
        # Initialize with default parameters
        self.params = self.model_config['params'].copy()
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Train the Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights for cost-sensitive learning
        """
        logger.info("Training Logistic Regression model...")
        
        self.feature_names = list(X_train.columns)
        
        # Initialize model with parameters
        self.model = LogisticRegression(**self.params, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        self.is_trained = True
        logger.info("Logistic Regression model training completed!")
        
        # Log model information
        logger.info(f"Model coefficients shape: {self.model.coef_.shape}")
        logger.info(f"Number of classes: {len(self.model.classes_)}")
        logger.info(f"Classes: {self.model.classes_}")
    
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
        logger.info("Starting hyperparameter tuning for Logistic Regression...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'solver': 'saga',  # saga supports all penalties
                'multi_class': trial.suggest_categorical('multi_class', ['ovr', 'multinomial']),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'max_iter': trial.suggest_int('max_iter', 500, 2000)
            }
            
            # Handle l1_ratio for elasticnet
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
            # Adjust solver based on penalty
            if params['penalty'] == 'l1':
                params['solver'] = 'liblinear'
                params['multi_class'] = 'ovr'  # liblinear only supports ovr
            elif params['penalty'] == 'l2':
                params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
                if params['solver'] == 'liblinear':
                    params['multi_class'] = 'ovr'
            
            # Cross-validation
            cv_folds = self.training_config['cv_folds']
            
            try:
                temp_model = LogisticRegression(**params, random_state=42)
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
        temp_model = LogisticRegression(**self.params, random_state=42)
        
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
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance based on coefficient magnitudes.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Feature importance DataFrame
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # For multi-class, take mean of absolute coefficients across classes
        if len(self.model.coef_.shape) > 1:
            # Multi-class case
            importance_scores = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # Binary case
            importance_scores = np.abs(self.model.coef_[0])
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients for each class.
        
        Returns:
            DataFrame with coefficients for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting coefficients")
        
        if len(self.model.coef_.shape) > 1:
            # Multi-class case
            coef_df = pd.DataFrame(
                self.model.coef_.T,
                index=self.feature_names,
                columns=[f'Class_{cls}' for cls in self.model.classes_]
            )
        else:
            # Binary case
            coef_df = pd.DataFrame(
                self.model.coef_.T,
                index=self.feature_names,
                columns=['Coefficient']
            )
        
        return coef_df
    
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
        
        return evaluation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': 'Logistic Regression',
            'parameters': self.params,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'num_classes': len(self.model.classes_),
            'classes': self.model.classes_.tolist(),
            'best_hyperparameters': self.best_params,
            'intercept': self.model.intercept_.tolist(),
            'n_iter': self.model.n_iter_.tolist() if hasattr(self.model, 'n_iter_') else None
        }
        
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
    model = LogisticRegressionCreditRiskModel()
    
    # Tune hyperparameters (small example)
    best_params = model.tune_hyperparameters(X_train, y_train, n_trials=10)
    print(f"Best parameters: {best_params}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate
    evaluation = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {evaluation['accuracy']:.4f}")
    print(f"Test F1-macro: {evaluation['f1_macro']:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    print("\nTop 10 features:")
    print(importance)
    
    # Model coefficients
    coefficients = model.get_coefficients()
    print("\nModel coefficients:")
    print(coefficients.head())