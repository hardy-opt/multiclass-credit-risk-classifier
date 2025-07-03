"""
Visualization utilities for credit risk assessment models.
Provides comprehensive plotting and visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import logging
import os
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import yaml

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Comprehensive visualization tool for credit risk assessment models.
    Creates publication-ready plots and interactive visualizations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.risk_config = self.config['risk_categories']
        
        # Setup plotting style
        self.figure_size = self.viz_config['figure_size']
        self.dpi = self.viz_config['dpi']
        self.style = self.viz_config['style']
        self.color_palette = self.viz_config['color_palette']
        
        # Set style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
        
        # Class names for better labeling
        self.class_names = self.risk_config['class_names']
        
        # Risk category colors
        self.risk_colors = {
            0: '#228B22',  # Very Low Risk - Green
            1: '#90EE90',  # Low Risk - Light Green
            2: '#FFD700',  # Medium Risk - Gold
            3: '#FF8C00',  # High Risk - Orange
            4: '#DC143C'   # Very High Risk - Red
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str, save_path: Optional[str] = None,
                            normalize: bool = True) -> None:
        """
        Plot confusion matrix with customization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
            normalize: Whether to normalize the matrix
        """
        logger.info(f"Creating confusion matrix plot for {model_name}...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_to_plot = cm_normalized
            fmt = '.2%'
            title = f'Normalized Confusion Matrix - {model_name}'
        else:
            cm_to_plot = cm
            fmt = 'd'
            title = f'Confusion Matrix - {model_name}'
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Get class labels
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_labels = [self.class_names.get(cls, f"Class {cls}") for cls in unique_classes]
        
        # Create heatmap
        sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating ROC curves for {model_name}...")
        
        # Get unique classes
        unique_classes = sorted(np.unique(y_true))
        n_classes = len(unique_classes)
        
        # Create binary labels for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=unique_classes)
        
        if n_classes == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot ROC curve for each class
        colors = [self.risk_colors.get(cls, f'C{i}') for i, cls in enumerate(unique_classes)]
        
        for i, (cls, color) in enumerate(zip(unique_classes, colors)):
            if i < y_pred_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_name = self.class_names.get(cls, f"Class {cls}")
                ax.plot(fpr, tpr, color=color, lw=2, 
                       label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(f'ROC Curves - {model_name}', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot Precision-Recall curves for multi-class classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating Precision-Recall curves for {model_name}...")
        
        # Get unique classes
        unique_classes = sorted(np.unique(y_true))
        n_classes = len(unique_classes)
        
        # Create binary labels for each class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=unique_classes)
        
        if n_classes == 2:
            y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # Plot PR curve for each class
        colors = [self.risk_colors.get(cls, f'C{i}') for i, cls in enumerate(unique_classes)]
        
        for i, (cls, color) in enumerate(zip(unique_classes, colors)):
            if i < y_pred_proba.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
                pr_auc = auc(recall, precision)
                
                class_name = self.class_names.get(cls, f"Class {cls}")
                ax.plot(recall, precision, color=color, lw=2,
                       label=f'{class_name} (AUC = {pr_auc:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax.set_title(f'Precision-Recall Curves - {model_name}', fontsize=16, fontweight='bold')
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str,
                              save_path: Optional[str] = None, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance
            model_name: Name of the model
            save_path: Path to save the plot
            top_n: Number of top features to display
        """
        logger.info(f"Creating feature importance plot for {model_name}...")
        
        # Get top features
        top_features = importance_df.head(top_n).copy()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)), dpi=self.dpi)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color='steelblue', alpha=0.7, edgecolor='navy')
        
        # Customize plot
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_results: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison results.
        
        Args:
            comparison_results: Results from model comparison
            save_path: Path to save the plot
        """
        logger.info("Creating model comparison plot...")
        
        # Extract metrics for plotting
        metric_comparisons = comparison_results['metric_comparisons']
        
        # Create subplot for each metric
        metrics = list(metric_comparisons.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                models = list(metric_comparisons[metric].keys())
                scores = list(metric_comparisons[metric].values())
                
                # Create bar plot
                bars = axes[i].bar(models, scores, alpha=0.7, 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(models))))
                
                axes[i].set_title(f'{metric.replace("_", " ").title()}', 
                                fontweight='bold', fontsize=12)
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_risk_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot risk category distribution comparison.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        logger.info("Creating risk distribution plot...")
        
        # Calculate distributions
        true_dist = pd.Series(y_true).value_counts().sort_index()
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        # Align indices
        all_classes = sorted(set(true_dist.index) | set(pred_dist.index))
        true_dist = true_dist.reindex(all_classes, fill_value=0)
        pred_dist = pred_dist.reindex(all_classes, fill_value=0)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Bar plot comparison
        x = np.arange(len(all_classes))
        width = 0.35
        
        class_labels = [self.class_names.get(cls, f"Class {cls}") for cls in all_classes]
        colors_true = [self.risk_colors.get(cls, 'gray') for cls in all_classes]
        
        bars1 = ax1.bar(x - width/2, true_dist.values, width, label='True', 
                       color=colors_true, alpha=0.7, edgecolor='black')
        bars2 = ax1.bar(x + width/2, pred_dist.values, width, label='Predicted', 
                       color=colors_true, alpha=0.4, edgecolor='black')
        
        ax1.set_xlabel('Risk Categories', fontweight='bold')
        ax1.set_ylabel('Count', fontweight='bold')
        ax1.set_title('Risk Category Distribution Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height * 0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Pie chart for true distribution
        ax2.pie(true_dist.values, labels=class_labels, colors=colors_true,
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('True Risk Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Risk distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(self, shap_values: np.ndarray, X: pd.DataFrame, 
                         model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values
            X: Feature values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        try:
            import shap
            logger.info(f"Creating SHAP summary plot for {model_name}...")
            
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # For multi-class, plot summary for each class
            if len(shap_values.shape) == 3:
                # Multi-class case
                for i in range(shap_values.shape[2]):
                    plt.subplot(1, shap_values.shape[2], i+1)
                    shap.summary_plot(shap_values[:, :, i], X, 
                                    title=f'{model_name} - {self.class_names.get(i, f"Class {i}")}',
                                    show=False)
            else:
                # Binary or single class case
                shap.summary_plot(shap_values, X, title=f'{model_name} - SHAP Summary', show=False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("SHAP not available. Skipping SHAP plots.")
        except Exception as e:
            logger.error(f"Error creating SHAP plot: {str(e)}")
    
    def plot_learning_curves(self, train_scores: List[float], val_scores: List[float],
                           model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot learning curves.
        
        Args:
            train_scores: Training scores over iterations
            val_scores: Validation scores over iterations
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info(f"Creating learning curves for {model_name}...")
        
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        iterations = range(1, len(train_scores) + 1)
        
        ax.plot(iterations, train_scores, 'o-', color='blue', label='Training Score', linewidth=2)
        ax.plot(iterations, val_scores, 's-', color='red', label='Validation Score', linewidth=2)
        
        ax.set_xlabel('Iterations', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'Learning Curves - {model_name}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add best score annotations
        best_train_idx = np.argmax(train_scores)
        best_val_idx = np.argmax(val_scores)
        
        ax.annotate(f'Best Train: {train_scores[best_train_idx]:.3f}',
                   xy=(best_train_idx + 1, train_scores[best_train_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
        
        ax.annotate(f'Best Val: {val_scores[best_val_idx]:.3f}',
                   xy=(best_val_idx + 1, val_scores[best_val_idx]),
                   xytext=(10, -15), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             model_name: str, save_path: Optional[str] = None) -> None:
        """
        Plot calibration curve for probability predictions.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        try:
            from sklearn.calibration import calibration_curve
            logger.info(f"Creating calibration curve for {model_name}...")
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # For multi-class, create calibration curve for each class
            unique_classes = sorted(np.unique(y_true))
            colors = [self.risk_colors.get(cls, f'C{i}') for i, cls in enumerate(unique_classes)]
            
            for i, (cls, color) in enumerate(zip(unique_classes, colors)):
                if i < y_pred_proba.shape[1]:
                    y_true_binary = (y_true == cls).astype(int)
                    y_prob = y_pred_proba[:, i]
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_true_binary, y_prob, n_bins=10
                    )
                    
                    class_name = self.class_names.get(cls, f"Class {cls}")
                    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                           color=color, label=class_name, linewidth=2)
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
            
            ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontweight='bold')
            ax.set_title(f'Calibration Curve - {model_name}', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Calibration curve saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Calibration curve plotting requires scikit-learn >= 0.22")
        except Exception as e:
            logger.error(f"Error creating calibration curve: {str(e)}")
    
    def create_model_report_dashboard(self, model_results: Dict[str, Any], 
                                    model_name: str, save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            model_results: Complete model evaluation results
            model_name: Name of the model
            save_dir: Directory to save plots
        """
        logger.info(f"Creating comprehensive dashboard for {model_name}...")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Extract data from results
        y_true = np.array(model_results.get('y_true', []))
        y_pred = np.array(model_results.get('y_pred', []))
        y_pred_proba = np.array(model_results.get('y_pred_proba', []))
        
        if len(y_true) == 0:
            logger.warning("No evaluation data found in model results")
            return
        
        # Create plots
        plots_created = []
        
        try:
            # Confusion Matrix
            save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png") if save_dir else None
            self.plot_confusion_matrix(y_true, y_pred, model_name, save_path)
            if save_path:
                plots_created.append(save_path)
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {str(e)}")
        
        try:
            # ROC Curves
            if len(y_pred_proba) > 0:
                save_path = os.path.join(save_dir, f"{model_name}_roc_curves.png") if save_dir else None
                self.plot_roc_curves(y_true, y_pred_proba, model_name, save_path)
                if save_path:
                    plots_created.append(save_path)
        except Exception as e:
            logger.error(f"Error creating ROC curves: {str(e)}")
        
        try:
            # Risk Distribution
            save_path = os.path.join(save_dir, f"{model_name}_risk_distribution.png") if save_dir else None
            self.plot_risk_distribution(y_true, y_pred, save_path)
            if save_path:
                plots_created.append(save_path)
        except Exception as e:
            logger.error(f"Error creating risk distribution plot: {str(e)}")
        
        logger.info(f"Dashboard creation completed! Created {len(plots_created)} plots.")
        if plots_created:
            logger.info(f"Plots saved: {plots_created}")


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
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Create plots
    visualizer.plot_confusion_matrix(y_test, y_pred, "RandomForest")
    visualizer.plot_roc_curves(y_test, y_pred_proba, "RandomForest")
    visualizer.plot_risk_distribution(y_test, y_pred)
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(20)]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    visualizer.plot_feature_importance(importance_df, "RandomForest")
    
    print("Visualization examples completed!")