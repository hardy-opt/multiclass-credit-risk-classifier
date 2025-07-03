"""
Model evaluation utilities for credit risk assessment.
Provides comprehensive evaluation metrics and analysis tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, cohen_kappa_score
)
import yaml

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator for credit risk assessment models.
    Provides detailed evaluation metrics and analysis.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluation_config = self.config['evaluation']
        self.risk_config = self.config['risk_categories']
        
        # Get class names for better reporting
        self.class_names = self.risk_config['class_names']
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info("Performing comprehensive classification evaluation...")
        
        results = {}
        
        # Basic classification metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results['per_class_metrics'] = {}
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        for i, cls in enumerate(unique_classes):
            class_name = self.class_names.get(cls, f"Class_{cls}")
            if i < len(precision_per_class):
                results['per_class_metrics'][class_name] = {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i])
                }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_normalized'] = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        
        # Classification report
        target_names = [self.class_names.get(cls, f"Class_{cls}") for cls in unique_classes]
        results['classification_report'] = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        
        # Cohen's Kappa
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) > 2:
                    # Multi-class ROC AUC
                    results['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                          multi_class='ovr', average='macro')
                    results['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, 
                                                          multi_class='ovo', average='macro')
                else:
                    # Binary ROC AUC
                    results['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        logger.info("Classification evaluation completed!")
        return results
    
    def evaluate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                cost_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate business-specific metrics for credit risk.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_matrix: Cost matrix for misclassification costs
            
        Returns:
            Dictionary containing business metrics
        """
        logger.info("Calculating business metrics...")
        
        results = {}
        
        # Get cost matrix from config if not provided
        if cost_matrix is None:
            cost_matrix = self._get_cost_matrix_from_config()
        
        # Calculate total cost
        if cost_matrix is not None:
            cm = confusion_matrix(y_true, y_pred)
            total_cost = np.sum(cm * cost_matrix)
            avg_cost_per_sample = total_cost / len(y_true)
            
            results['total_misclassification_cost'] = float(total_cost)
            results['average_cost_per_sample'] = float(avg_cost_per_sample)
        
        # Risk category distribution analysis
        true_distribution = pd.Series(y_true).value_counts().sort_index()
        pred_distribution = pd.Series(y_pred).value_counts().sort_index()
        
        results['true_risk_distribution'] = {
            self.class_names.get(cls, f"Class_{cls}"): int(count)
            for cls, count in true_distribution.items()
        }
        results['predicted_risk_distribution'] = {
            self.class_names.get(cls, f"Class_{cls}"): int(count)
            for cls, count in pred_distribution.items()
        }
        
        # Calculate risk category accuracy
        risk_accuracy = {}
        for cls in np.unique(y_true):
            class_mask = y_true == cls
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
                risk_accuracy[self.class_names.get(cls, f"Class_{cls}")] = float(class_accuracy)
        
        results['risk_category_accuracy'] = risk_accuracy
        
        # Calculate conservative vs aggressive predictions
        conservative_predictions = np.sum(y_pred > y_true)  # Predicting higher risk than actual
        aggressive_predictions = np.sum(y_pred < y_true)    # Predicting lower risk than actual
        
        results['conservative_predictions'] = int(conservative_predictions)
        results['aggressive_predictions'] = int(aggressive_predictions)
        results['conservative_rate'] = float(conservative_predictions / len(y_true))
        results['aggressive_rate'] = float(aggressive_predictions / len(y_true))
        
        logger.info("Business metrics calculation completed!")
        return results
    
    def _get_cost_matrix_from_config(self) -> Optional[np.ndarray]:
        """Get cost matrix from configuration."""
        try:
            cost_config = self.risk_config.get('cost_matrix', {})
            if not cost_config:
                return None
            
            # Convert cost matrix to numpy array
            classes = ['very_low', 'low', 'medium', 'high', 'very_high']
            cost_matrix = np.zeros((len(classes), len(classes)))
            
            for i, actual_class in enumerate(classes):
                if actual_class in cost_config:
                    cost_matrix[i] = cost_config[actual_class]
            
            return cost_matrix
        except Exception as e:
            logger.warning(f"Could not create cost matrix from config: {str(e)}")
            return None
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and rank them.
        
        Args:
            model_results: Dictionary of model names to their evaluation results
            
        Returns:
            Model comparison and ranking
        """
        logger.info("Comparing model performances...")
        
        comparison = {
            'model_rankings': {},
            'metric_comparisons': {},
            'best_models_by_metric': {},
            'summary': {}
        }
        
        # Metrics to compare
        comparison_metrics = [
            'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 
            'recall_macro', 'cohen_kappa'
        ]
        
        # Extract metrics for comparison
        metric_data = {}
        for metric in comparison_metrics:
            metric_data[metric] = {}
            for model_name, results in model_results.items():
                metric_data[metric][model_name] = results.get(metric, 0)
        
        comparison['metric_comparisons'] = metric_data
        
        # Find best model for each metric
        for metric in comparison_metrics:
            best_model = max(metric_data[metric], key=metric_data[metric].get)
            comparison['best_models_by_metric'][metric] = {
                'model': best_model,
                'score': metric_data[metric][best_model]
            }
        
        # Calculate overall ranking using weighted score
        weights = {
            'accuracy': 0.15,
            'f1_macro': 0.25,
            'f1_weighted': 0.20,
            'precision_macro': 0.15,
            'recall_macro': 0.15,
            'cohen_kappa': 0.10
        }
        
        model_scores = {}
        for model_name in model_results.keys():
            weighted_score = sum(
                weights.get(metric, 0) * metric_data[metric].get(model_name, 0)
                for metric in comparison_metrics
            )
            model_scores[model_name] = weighted_score
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['model_rankings'] = {
            f'rank_{i+1}': {'model': model, 'weighted_score': score}
            for i, (model, score) in enumerate(ranked_models)
        }
        
        # Summary statistics
        comparison['summary'] = {
            'total_models': len(model_results),
            'best_overall_model': ranked_models[0][0] if ranked_models else None,
            'best_overall_score': ranked_models[0][1] if ranked_models else None,
            'metric_leaders': {
                metric: comparison['best_models_by_metric'][metric]['model']
                for metric in comparison_metrics
            }
        }
        
        logger.info("Model comparison completed!")
        return comparison
    
    def generate_evaluation_report(self, model_name: str, y_true: np.ndarray, 
                                 y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for a single model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Comprehensive evaluation report
        """
        logger.info(f"Generating evaluation report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(y_true),
                'num_classes': len(np.unique(y_true)),
                'class_distribution': pd.Series(y_true).value_counts().to_dict()
            }
        }
        
        # Classification metrics
        report['classification_metrics'] = self.evaluate_classification(y_true, y_pred, y_pred_proba)
        
        # Business metrics
        report['business_metrics'] = self.evaluate_business_metrics(y_true, y_pred)
        
        # Prediction analysis
        report['prediction_analysis'] = self._analyze_predictions(y_true, y_pred)
        
        logger.info(f"Evaluation report for {model_name} completed!")
        return report
    
    def _analyze_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction patterns and errors."""
        analysis = {}
        
        # Prediction accuracy by true class
        class_accuracy = {}
        for cls in np.unique(y_true):
            mask = y_true == cls
            if np.sum(mask) > 0:
                accuracy = accuracy_score(y_true[mask], y_pred[mask])
                class_accuracy[self.class_names.get(cls, f"Class_{cls}")] = float(accuracy)
        
        analysis['accuracy_by_true_class'] = class_accuracy
        
        # Common misclassifications
        cm = confusion_matrix(y_true, y_pred)
        misclassifications = []
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    misclassifications.append({
                        'true_class': self.class_names.get(i, f"Class_{i}"),
                        'predicted_class': self.class_names.get(j, f"Class_{j}"),
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / np.sum(cm[i]) * 100)
                    })
        
        # Sort by count
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        analysis['top_misclassifications'] = misclassifications[:10]
        
        # Error severity analysis
        error_distances = np.abs(y_true - y_pred)
        analysis['error_severity'] = {
            'mean_error_distance': float(np.mean(error_distances)),
            'max_error_distance': int(np.max(error_distances)),
            'error_distance_distribution': pd.Series(error_distances).value_counts().to_dict()
        }
        
        return analysis


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                             n_informative=15, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Generate comprehensive report
    report = evaluator.generate_evaluation_report("RandomForest", y_test, y_pred, y_pred_proba)
    
    print("Evaluation Report Generated!")
    print(f"Accuracy: {report['classification_metrics']['accuracy']:.4f}")
    print(f"F1-Macro: {report['classification_metrics']['f1_macro']:.4f}")
    print(f"Cohen's Kappa: {report['classification_metrics']['cohen_kappa']:.4f}")