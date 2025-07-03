def run_complete_pipeline(self, tune_hyperparameters: bool = True) -> None:
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        
        try:
            # Load and prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
            
            # Train models
            self.train_models(X_train, X_val, y_train, y_val, tune_hyperparameters)
            
            # Evaluate models - pass all required arguments
            self.evaluate_models(X_test, y_test, X_train, y_train, X_val, y_val)
            
            # Save models
            self.save_models()
            
            # Generate visualizations
            self.generate_visualizations(X_test, y_test)
            
            # Save results
            self.save_results()
            
            logger.info("Training pipeline completed successfully!")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise#!/usr/bin/env python3
"""
Main training script for multi-class credit risk assessment models.
Orchestrates the entire training pipeline including data loading, preprocessing,
model training, evaluation, and result saving.
"""

import os
import sys
import logging
import argparse
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_path)

try:
    from data.data_loader import GermanCreditDataLoader
    from data.preprocessor import DataPreprocessor
    from data.risk_categorizer import RiskCategorizer
    from models.logistic_regression import LogisticRegressionCreditRiskModel
    from models.xgboost_model import XGBoostCreditRiskModel
    from models.random_forest import RandomForestCreditRiskModel
    from models.ensemble_model import EnsembleCreditRiskModel
    from utils.evaluation import ModelEvaluator
    from utils.visualization import ModelVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Please make sure you're running this script from the project root directory.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """Complete training pipeline for credit risk models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize training pipeline."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = GermanCreditDataLoader(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.risk_categorizer = RiskCategorizer(config_path)
        self.evaluator = ModelEvaluator(config_path)
        self.visualizer = ModelVisualizer(config_path)
        
        # Initialize models based on config
        self.models = {}
        self._initialize_models(config_path)
        
        # Results storage
        self.results = {}
        self.trained_models = {}
        
        # Create output directories
        self._create_output_directories()
    
    def _initialize_models(self, config_path: str) -> None:
        """Initialize models based on configuration."""
        models_config = self.config['models']
        
        if models_config['logistic_regression']['enabled']:
            self.models['logistic_regression'] = LogisticRegressionCreditRiskModel(config_path)
        
        if models_config['xgboost']['enabled']:
            self.models['xgboost'] = XGBoostCreditRiskModel(config_path)
        
        if models_config['random_forest']['enabled']:
            self.models['random_forest'] = RandomForestCreditRiskModel(config_path)
        
        if models_config['ensemble']['enabled']:
            self.models['ensemble'] = EnsembleCreditRiskModel(config_path)
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        output_config = self.config['output']
        
        for path_key in ['models_path', 'plots_path', 'reports_path', 'metrics_path']:
            path = output_config[path_key]
            os.makedirs(path, exist_ok=True)
        
        # Create subdirectories for plots
        plots_config = self.config['visualization']['plots']
        for plot_type in plots_config:
            plot_dir = os.path.join(output_config['plots_path'], plot_type)
            os.makedirs(plot_dir, exist_ok=True)
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
    
    def load_and_prepare_data(self) -> tuple:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")
        
        # Load raw data
        features, targets = self.data_loader.load_data()
        
        # Create multi-class risk categories
        risk_categories = self.risk_categorizer.create_risk_categories(features, targets)
        
        # Preprocess features
        X_processed = self.preprocessor.fit_transform(features, risk_categories)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        stratify = risk_categories if self.config['data']['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, risk_categories,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Further split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=random_state,
            stratify=y_train
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Save processed data
        self._save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test) -> None:
        """Save processed data for future use."""
        processed_path = self.config['data']['processed_data_path']
        
        X_train.to_csv(os.path.join(processed_path, 'X_train.csv'), index=False)
        X_val.to_csv(os.path.join(processed_path, 'X_val.csv'), index=False)
        X_test.to_csv(os.path.join(processed_path, 'X_test.csv'), index=False)
        
        pd.Series(y_train).to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
        pd.Series(y_val).to_csv(os.path.join(processed_path, 'y_val.csv'), index=False)
        pd.Series(y_test).to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)
        
        # Save feature names
        feature_info = {
            'feature_names': list(X_train.columns),
            'num_features': len(X_train.columns),
            'preprocessing_info': self.preprocessor.get_preprocessing_info()
        }
        
        with open(os.path.join(processed_path, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)
    
    def train_models(self, X_train, X_val, y_train, y_val, tune_hyperparameters: bool = True) -> None:
        """Train all enabled models."""
        logger.info("Starting model training...")
        
        training_config = self.config['training']
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Hyperparameter tuning if enabled
                if tune_hyperparameters and training_config['hyperparameter_tuning']['enabled']:
                    if hasattr(model, 'tune_hyperparameters'):
                        logger.info(f"Tuning hyperparameters for {model_name}...")
                        n_trials = training_config['hyperparameter_tuning']['n_trials']
                        timeout = training_config['hyperparameter_tuning']['timeout']
                        best_params = model.tune_hyperparameters(X_train, y_train, n_trials, timeout)
                        logger.info(f"Best parameters for {model_name}: {best_params}")
                
                # Train the model
                if model_name == 'ensemble':
                    # Ensemble needs pre-trained base models
                    base_models = {name: model for name, model in self.trained_models.items() 
                                 if name != 'ensemble'}
                    model.train(X_train, y_train, X_val, y_val, base_models=base_models)
                else:
                    model.train(X_train, y_train, X_val, y_val)
                
                self.trained_models[model_name] = model
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
    
    def evaluate_models(self, X_test, y_test, X_train=None, y_train=None, X_val=None, y_val=None) -> None:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        for model_name, model in self.trained_models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                # Get model evaluation
                evaluation = model.evaluate(X_test, y_test)
                self.results[model_name] = evaluation
                
                # Cross-validation if supported and training data is provided
                if hasattr(model, 'cross_validate') and X_train is not None and y_train is not None:
                    try:
                        if X_val is not None and y_val is not None:
                            X_cv = pd.concat([X_train, X_val])
                            y_cv = pd.concat([y_train, y_val])
                        else:
                            X_cv = X_train
                            y_cv = y_train
                        
                        cv_scores = model.cross_validate(
                            X_cv, y_cv,
                            cv_folds=self.config['training']['cv_folds']
                        )
                        evaluation['cross_validation'] = cv_scores
                    except Exception as cv_error:
                        logger.warning(f"Cross-validation failed for {model_name}: {str(cv_error)}")
                
                logger.info(f"Evaluation completed for {model_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
    
    def save_models(self) -> None:
        """Save all trained models."""
        logger.info("Saving trained models...")
        
        models_path = self.config['output']['models_path']
        model_format = self.config['output']['model_format']
        
        for model_name, model in self.trained_models.items():
            try:
                if model_format == 'pickle':
                    filepath = os.path.join(models_path, f"{model_name}.pkl")
                elif model_format == 'joblib':
                    filepath = os.path.join(models_path, f"{model_name}.joblib")
                else:
                    filepath = os.path.join(models_path, f"{model_name}.pkl")
                
                model.save_model(filepath)
                logger.info(f"Saved {model_name} to {filepath}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
    
    def generate_visualizations(self, X_test, y_test) -> None:
        """Generate visualizations for model results."""
        logger.info("Generating visualizations...")
        
        plots_config = self.config['visualization']['plots']
        
        for model_name, model in self.trained_models.items():
            try:
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Generate plots based on configuration
                if 'confusion_matrix' in plots_config:
                    self.visualizer.plot_confusion_matrix(
                        y_test, y_pred, model_name,
                        save_path=f"results/plots/confusion_matrix/{model_name}_confusion_matrix.png"
                    )
                
                if 'roc_curve' in plots_config and y_pred_proba is not None:
                    self.visualizer.plot_roc_curves(
                        y_test, y_pred_proba, model_name,
                        save_path=f"results/plots/roc_curve/{model_name}_roc_curves.png"
                    )
                
                if 'feature_importance' in plots_config and hasattr(model, 'get_feature_importance'):
                    importance_df = model.get_feature_importance()
                    self.visualizer.plot_feature_importance(
                        importance_df, model_name,
                        save_path=f"results/plots/feature_importance/{model_name}_feature_importance.png"
                    )
                
                if 'shap_summary' in plots_config and hasattr(model, 'get_shap_values'):
                    try:
                        model.setup_shap_explainer(X_test.sample(n=100, random_state=42))
                        shap_values = model.get_shap_values(X_test.head(50))
                        self.visualizer.plot_shap_summary(
                            shap_values, X_test.head(50), model_name,
                            save_path=f"results/plots/shap_summary/{model_name}_shap_summary.png"
                        )
                    except Exception as e:
                        logger.warning(f"Could not generate SHAP plots for {model_name}: {str(e)}")
                
                logger.info(f"Generated visualizations for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating visualizations for {model_name}: {str(e)}")
    
    def save_results(self) -> None:
        """Save evaluation results and metrics."""
        logger.info("Saving results...")
        
        # Save individual model results
        metrics_path = self.config['output']['metrics_path']
        
        for model_name, results in self.results.items():
            result_file = os.path.join(metrics_path, f"{model_name}_results.json")
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Create summary comparison
        summary = self._create_model_comparison_summary()
        summary_file = os.path.join(metrics_path, "model_comparison_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save training configuration and metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'models_trained': list(self.trained_models.keys()),
            'data_info': self.data_loader.get_data_info(),
            'risk_analysis': self.risk_categorizer.analyze_risk_distribution(
                pd.Series(self.results[list(self.results.keys())[0]]['predictions'] if self.results else [])
            )
        }
        
        metadata_file = os.path.join(metrics_path, "training_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Results saved successfully!")
    
    def _create_model_comparison_summary(self) -> dict:
        """Create a summary comparing all models."""
        if not self.results:
            return {}
        
        metrics_to_compare = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']
        comparison = {}
        
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name, results in self.results.items():
                comparison[metric][model_name] = results.get(metric, 0)
        
        # Find best model for each metric
        best_models = {}
        for metric in metrics_to_compare:
            if comparison[metric]:
                best_model = max(comparison[metric], key=comparison[metric].get)
                best_models[metric] = {
                    'model': best_model,
                    'score': comparison[metric][best_model]
                }
        
        return {
            'metrics_comparison': comparison,
            'best_models': best_models,
            'overall_best': self._determine_overall_best_model()
        }
    
    def _determine_overall_best_model(self) -> dict:
        """Determine the overall best performing model."""
        if not self.results:
            return {}
        
        # Weight different metrics
        weights = {
            'accuracy': 0.2,
            'precision_macro': 0.2,
            'recall_macro': 0.2,
            'f1_macro': 0.3,
            'f1_weighted': 0.1
        }
        
        model_scores = {}
        for model_name, results in self.results.items():
            weighted_score = sum(
                weights.get(metric, 0) * results.get(metric, 0)
                for metric in weights.keys()
            )
            model_scores[model_name] = weighted_score
        
        if model_scores:
            best_model = max(model_scores, key=model_scores.get)
            return {
                'model': best_model,
                'weighted_score': model_scores[best_model],
                'individual_scores': self.results[best_model]
            }
        
        return {}
    
    def run_complete_pipeline(self, tune_hyperparameters: bool = True) -> None:
        """Run the complete training pipeline."""
        logger.info("Starting complete training pipeline...")
        
        try:
            # Load and prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_prepare_data()
            
            # Train models
            self.train_models(X_train, X_val, y_train, y_val, tune_hyperparameters)
            
            # Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # Save models
            self.save_models()
            
            # Generate visualizations
            self.generate_visualizations(X_test, y_test)
            
            # Save results
            self.save_results()
            
            logger.info("Training pipeline completed successfully!")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise
    
    def _print_summary(self) -> None:
        """Print training summary."""
        print("\n" + "="*80)
        print("TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        print(f"Models Trained: {list(self.trained_models.keys())}")
        print(f"Total Models: {len(self.trained_models)}")
        
        if self.results:
            print("\nModel Performance Summary:")
            print("-" * 40)
            for model_name, results in self.results.items():
                print(f"{model_name.upper()}:")
                print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
                print(f"  F1-Macro: {results.get('f1_macro', 0):.4f}")
                print(f"  F1-Weighted: {results.get('f1_weighted', 0):.4f}")
                print()
            
            # Best model
            best_model_info = self._determine_overall_best_model()
            if best_model_info:
                print(f"BEST OVERALL MODEL: {best_model_info['model'].upper()}")
                print(f"Weighted Score: {best_model_info['weighted_score']:.4f}")
        
        print("="*80)


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Train Credit Risk Assessment Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--no-tune', action='store_true',
                      help='Skip hyperparameter tuning')
    parser.add_argument('--models', nargs='+', 
                      choices=['logistic_regression', 'xgboost', 'random_forest', 'ensemble'],
                      help='Specific models to train (default: all enabled in config)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline(args.config)
    
    # Filter models if specified
    if args.models:
        pipeline.models = {name: model for name, model in pipeline.models.items() 
                          if name in args.models}
    
    # Run pipeline
    tune_hyperparameters = not args.no_tune
    pipeline.run_complete_pipeline(tune_hyperparameters)


if __name__ == "__main__":
    main()