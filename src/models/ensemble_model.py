"""
Ensemble model implementation for multi-class credit risk assessment.
Combines multiple base models using voting or stacking approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import yaml

logger = logging.getLogger(__name__)


class EnsembleCreditRiskModel:
    """
    Ensemble model for multi-class credit risk assessment.
    Supports voting and stacking ensemble methods.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize Ensemble model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['ensemble']
        self.training_config = self.config['training']
        self.risk_config = self.config['risk_categories']
        
        self.model = None
        self.base_models = {}
        self.feature_names = None
        self.is_trained = False
        self.ensemble_method = self.model_config['method']
        self.voting_type = self.model_config.get('voting_type', 'soft')
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              base_models: Optional[Dict[str, Any]] = None,
              sample_weight: Optional[np.ndarray] = None) -> None:
        """
        Train the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            base_models: Dictionary of pre-trained base models
            sample_weight: Sample weights for cost-sensitive learning
        """
        logger.info(f"Training Ensemble model using {self.ensemble_method} method...")
        
        self.feature_names = list(X_train.columns)
        
        if base_models is None:
            raise ValueError("Base models must be provided for ensemble training")
        
        self.base_models = base_models
        
        # Prepare estimators for ensemble
        estimators = []
        for name, model in base_models.items():
            if hasattr(model, 'model') and model.model is not None:
                estimators.append((name, model.model))
            else:
                logger.warning(f"Model {name} is not trained or doesn't have a model attribute")
        
        if len(estimators) < 2:
            raise ValueError("At least 2 trained base models are required for ensemble")
        
        # Create ensemble model
        if self.ensemble_method == 'voting':
            self.model = VotingClassifier(
                estimators=estimators,
                voting=self.voting_type
            )
        elif self.ensemble_method == 'stacking':
            # Use logistic regression as meta-learner
            meta_learner = LogisticRegression(
                multi_class='ovr',
                random_state=42,
                max_iter=1000
            )
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3  # Use 3-fold CV for meta-features
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Train the ensemble
        self.model.fit(X_train, y_train, sample_weight=sample_weight)
        
        self.is_trained = True
        logger.info("Ensemble model training completed!")
        
        # Log ensemble information
        logger.info(f"Ensemble method: {self.ensemble_method}")
        logger.info(f"Number of base models: {len(estimators)}")
        logger.info(f"Base models: {[name for name, _ in estimators]}")
        if self.ensemble_method == 'voting':
            logger.info(f"Voting type: {self.voting_type}")
    
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
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Ensemble model doesn't support probability prediction")
    
    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models.
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary of base model predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        base_predictions = {}
        
        for name, base_model in self.base_models.items():
            if hasattr(base_model, 'predict'):
                try:
                    predictions = base_model.predict(X)
                    base_predictions[name] = predictions
                except Exception as e:
                    logger.warning(f"Could not get predictions from {name}: {str(e)}")
        
        return base_predictions
    
    def get_base_model_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get prediction probabilities from individual base models.
        
        Args:
            X: Features for prediction
            
        Returns:
            Dictionary of base model probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        base_probabilities = {}
        
        for name, base_model in self.base_models.items():
            if hasattr(base_model, 'predict_proba'):
                try:
                    probabilities = base_model.predict_proba(X)
                    base_probabilities[name] = probabilities
                except Exception as e:
                    logger.warning(f"Could not get probabilities from {name}: {str(e)}")
        
        return base_probabilities
    
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
        
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Calculate multiple metrics
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        cv_scores = {}
        
        for metric in metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        logger.info("Cross-validation completed!")
        return cv_scores
    
    def get_ensemble_weights(self) -> Optional[Dict[str, float]]:
        """
        Get ensemble weights if available.
        
        Returns:
            Dictionary of model weights or None
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'weights_') and self.model.weights_ is not None:
            # For weighted voting
            estimator_names = [name for name, _ in self.model.estimators]
            return dict(zip(estimator_names, self.model.weights_))
        elif hasattr(self.model, 'final_estimator_') and self.ensemble_method == 'stacking':
            # For stacking, get meta-learner coefficients
            if hasattr(self.model.final_estimator_, 'coef_'):
                estimator_names = [name for name, _ in self.model.estimators]
                # Average coefficients across classes for multi-class
                if len(self.model.final_estimator_.coef_.shape) > 1:
                    avg_coefs = np.mean(np.abs(self.model.final_estimator_.coef_), axis=0)
                else:
                    avg_coefs = np.abs(self.model.final_estimator_.coef_[0])
                
                return dict(zip(estimator_names, avg_coefs))
        
        return None
    
    def get_feature_importance(self, method: str = 'average') -> Optional[pd.DataFrame]:
        """
        Get feature importance by aggregating from base models.
        
        Args:
            method: Method to aggregate importance ('average', 'weighted')
            
        Returns:
            Feature importance DataFrame or None
        """
        if not self.is_trained:
            return None
        
        # Collect feature importances from base models
        importances = {}
        
        for name, base_model in self.base_models.items():
            if hasattr(base_model, 'get_feature_importance'):
                try:
                    model_importance = base_model.get_feature_importance()
                    importances[name] = model_importance.set_index('feature')['importance']
                except Exception as e:
                    logger.warning(f"Could not get feature importance from {name}: {str(e)}")
        
        if not importances:
            return None
        
        # Create combined DataFrame
        importance_df = pd.DataFrame(importances).fillna(0)
        
        if method == 'average':
            # Simple average
            combined_importance = importance_df.mean(axis=1)
        elif method == 'weighted':
            # Weighted average using ensemble weights
            weights = self.get_ensemble_weights()
            if weights:
                # Normalize weights
                total_weight = sum(weights.values())
                normalized_weights = {k: v/total_weight for k, v in weights.items()}
                
                weighted_sum = pd.Series(0, index=importance_df.index)
                for model_name, weight in normalized_weights.items():
                    if model_name in importance_df.columns:
                        weighted_sum += importance_df[model_name] * weight
                
                combined_importance = weighted_sum
            else:
                # Fallback to average
                combined_importance = importance_df.mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'feature': combined_importance.index,
            'importance': combined_importance.values
        }).sort_values('importance', ascending=False)
        
        return result_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained ensemble model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'base_models': self.base_models,
            'feature_names': self.feature_names,
            'ensemble_method': self.ensemble_method,
            'voting_type': self.voting_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained ensemble model.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.base_models = model_data['base_models']
        self.feature_names = model_data['feature_names']
        self.ensemble_method = model_data['ensemble_method']
        self.voting_type = model_data.get('voting_type', 'soft')
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the ensemble model on test data.
        
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
        y_pred_proba = self.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
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
            'predictions': y_pred.tolist()
        }
        
        if y_pred_proba is not None:
            evaluation['prediction_probabilities'] = y_pred_proba.tolist()
        
        # Add base model predictions for analysis
        base_predictions = self.get_base_model_predictions(X_test)
        evaluation['base_model_predictions'] = {
            name: preds.tolist() for name, preds in base_predictions.items()
        }
        
        # Add ensemble weights
        weights = self.get_ensemble_weights()
        if weights:
            evaluation['ensemble_weights'] = weights
        
        return evaluation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble model information."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        info = {
            'status': 'trained',
            'model_type': 'Ensemble',
            'ensemble_method': self.ensemble_method,
            'voting_type': self.voting_type if self.ensemble_method == 'voting' else None,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'num_base_models': len(self.base_models),
            'base_model_names': list(self.base_models.keys())
        }
        
        # Add ensemble weights
        weights = self.get_ensemble_weights()
        if weights:
            info['ensemble_weights'] = weights
        
        # Add feature importance if available
        feature_importance = self.get_feature_importance()
        if feature_importance is not None:
            info['top_features'] = feature_importance.head(10).to_dict('records')
        
        return info


if __name__ == "__main__":
    # Example usage would require pre-trained base models
    print("Ensemble model implementation completed!")
    print("Note: This model requires pre-trained base models to function.")
    print("See the main training script for proper usage example.")