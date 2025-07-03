"""
XGBoost model implementation for multi-class credit risk assessment.
Includes advanced features like early stopping, feature importance, and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import optuna
import yaml
import shap

logger = logging.getLogger(__name__)


class XGBoostCreditRiskModel:
    """
    XGBoost model for multi-class credit risk assessment.
    Supports hyperparameter tuning, feature importance analysis, and model interpretability.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize XGBoost model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['xgboost']
        self.training_config = self.config['training']
        self.risk_config = self.config['risk_categories']
        
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.best_params = None
        self.feature_importance_ = None
        self.shap_explainer = None
        
        # Initialize with default parameters
        self.params = self.model_config['params'].copy()
        self.params['num_class'] = self.risk_config['num_classes']
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Train the XGBoost model using scikit-learn interface for compatibility.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights for cost-sensitive learning
        """
        logger.info("Training XGBoost model...")
        
        self.feature_names = list(X_train.columns)
        
        try:
            # Use scikit-learn XGBoost interface for better compatibility
            from xgboost import XGBClassifier
            
            # Convert parameters for scikit-learn interface
            sklearn_params = {
                'objective': 'multi:softprob',
                'n_estimators': self.params.get('n_estimators', 100),
                'max_depth': self.params.get('max_depth', 6),
                'learning_rate': self.params.get('learning_rate', 0.1),
                'subsample': self.params.get('subsample', 0.8),
                'colsample_bytree': self.params.get('colsample_bytree', 0.8),
                'reg_alpha': self.params.get('reg_alpha', 0),
                'reg_lambda': self.params.get('reg_lambda', 1),
                'random_state': 42,
                'verbosity': 0
            }
            
            self.model = XGBClassifier(**sklearn_params)
            
            # Prepare validation data for early stopping
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            
            # Handle early stopping for newer XGBoost versions
            if X_val is not None and y_val is not None and self.training_config['early_stopping']['enabled']:
                try:
                    # Try new XGBoost 2.0+ format
                    fit_params['eval_set'] = [(X_val, y_val)]
                    fit_params['early_stopping_rounds'] = self.training_config['early_stopping']['rounds']
                    fit_params['verbose'] = False
                    
                    self.model.fit(X_train, y_train, **fit_params)
                    
                except TypeError as e:
                    if 'early_stopping_rounds' in str(e):
                        # Fall back to basic training without early stopping
                        logger.warning("Early stopping not supported in this XGBoost version, training without it")
                        basic_params = {k: v for k, v in fit_params.items() 
                                      if k not in ['eval_set', 'early_stopping_rounds', 'verbose']}
                        self.model.fit(X_train, y_train, **basic_params)
                    else:
                        raise
            else:
                # Basic training without early stopping
                self.model.fit(X_train, y_train, **fit_params)
            
            self.is_trained = True
            self._calculate_feature_importance()
            
            logger.info("XGBoost model training completed using scikit-learn interface!")
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            raise
    
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
        
        # Since we're using XGBClassifier, this should work directly
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
        
        # Since we're using XGBClassifier, this should work directly
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
        logger.info("Starting hyperparameter tuning...")
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'objective': 'multi:softprob',
                'num_class': self.risk_config['num_classes'],
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',
                'verbosity': 0,
                
                # Hyperparameters to tune
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7)
            }
            
            # Cross-validation
            cv_folds = self.training_config['cv_folds']
            cv_scoring = self.training_config['cv_scoring']
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            
            cv_results = xgb.cv(
                params=params,
                dtrain=dtrain,
                num_boost_round=params['n_estimators'],
                nfold=cv_folds,
                stratified=True,
                shuffle=True,
                seed=42,
                return_train_metric=False,
                verbose_eval=False
            )
            
            # Return the best validation score
            best_score = cv_results['test-mlogloss-mean'].min()
            return best_score
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        self.best_params['objective'] = 'multi:softprob'
        self.best_params['num_class'] = self.risk_config['num_classes']
        self.best_params['eval_metric'] = 'mlogloss'
        self.best_params['tree_method'] = 'hist'
        
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
        temp_model = xgb.XGBClassifier(**self.params)
        
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
        
        # For scikit-learn XGBClassifier
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            
            # Create DataFrame
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        else:
            logger.warning("Could not calculate feature importance for this XGBoost model")
    
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
    
    def setup_shap_explainer(self, X_background: pd.DataFrame, max_samples: int = 100) -> None:
        """
        Setup SHAP explainer for model interpretability.
        
        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum number of background samples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before setting up SHAP explainer")
        
        logger.info("Setting up SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            background_sample = X_background.sample(n=max_samples, random_state=42)
        else:
            background_sample = X_background
        
        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        logger.info("SHAP explainer setup completed!")
    
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get SHAP values for given samples.
        
        Args:
            X: Features to explain
            
        Returns:
            SHAP values
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer must be setup first using setup_shap_explainer()")
        
        return self.shap_explainer.shap_values(X)
    
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
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return evaluation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': 'XGBoost',
            'parameters': self.params,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'num_classes': self.params.get('num_class', 5),
            'best_hyperparameters': self.best_params
        }
        
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
    model = XGBoostCreditRiskModel()
    
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