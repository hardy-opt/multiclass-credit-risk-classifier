"""
Data preprocessing module for credit risk assessment.
Handles feature scaling, encoding, selection, and data preparation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import category_encoders as ce
import yaml
import joblib

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessor for credit risk assessment.
    Handles scaling, encoding, feature selection, and class balancing.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        
        # Initialize components
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.resampler = None
        self.preprocessor_pipeline = None
        
        # Feature information
        self.numerical_features = []
        self.categorical_features = []
        self.selected_features = []
        self.feature_names_after_encoding = []
        
        # Fitted status
        self.is_fitted = False
        
        # Setup preprocessing components
        self._setup_scaler()
        self._setup_feature_selector()
        self._setup_resampler()
    
    def _setup_scaler(self) -> None:
        """Setup the feature scaler based on configuration."""
        scaling_method = self.preprocessing_config['scaling_method']
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}. Using StandardScaler.")
            self.scaler = StandardScaler()
    
    def _setup_feature_selector(self) -> None:
        """Setup feature selection based on configuration."""
        if not self.preprocessing_config['feature_selection']['enabled']:
            self.feature_selector = None
            return
        
        method = self.preprocessing_config['feature_selection']['method']
        k_best = self.preprocessing_config['feature_selection']['k_best']
        
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
        elif method == 'chi2':
            self.feature_selector = SelectKBest(score_func=chi2, k=k_best)
        elif method == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k_best)
        else:
            logger.warning(f"Unknown feature selection method: {method}. Disabling feature selection.")
            self.feature_selector = None
    
    def _setup_resampler(self) -> None:
        """Setup resampling method for class imbalance."""
        if not self.preprocessing_config['resampling']['enabled']:
            self.resampler = None
            return
        
        method = self.preprocessing_config['resampling']['method']
        
        if method == 'smote':
            self.resampler = SMOTE(random_state=42)
        elif method == 'adasyn':
            self.resampler = ADASYN(random_state=42)
        elif method == 'random_oversample':
            self.resampler = RandomOverSampler(random_state=42)
        elif method == 'random_undersample':
            self.resampler = RandomUnderSampler(random_state=42)
        elif method == 'smoteenn':
            self.resampler = SMOTEENN(random_state=42)
        elif method == 'smotetomek':
            self.resampler = SMOTETomek(random_state=42)
        else:
            logger.warning(f"Unknown resampling method: {method}. Disabling resampling.")
            self.resampler = None
    
    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        """Identify numerical and categorical features."""
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Identified {len(self.numerical_features)} numerical features")
        logger.info(f"Identified {len(self.categorical_features)} categorical features")
    
    def _create_categorical_encoder(self) -> Any:
        """Create categorical encoder based on configuration."""
        encoding_method = self.preprocessing_config['categorical_encoding']
        
        if encoding_method == 'onehot':
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        elif encoding_method == 'target':
            return ce.TargetEncoder(handle_unknown='value', handle_missing='value')
        elif encoding_method == 'ordinal':
            return ce.OrdinalEncoder(handle_unknown='value', handle_missing='value')
        else:
            logger.warning(f"Unknown encoding method: {encoding_method}. Using OneHotEncoder.")
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting data preprocessor...")
        
        # Identify feature types
        self._identify_feature_types(X)
        
        # Create preprocessing pipeline
        preprocessors = []
        
        # Numerical features preprocessing
        if self.numerical_features:
            numerical_pipeline = Pipeline([
                ('scaler', self.scaler)
            ])
            preprocessors.append(('num', numerical_pipeline, self.numerical_features))
        
        # Categorical features preprocessing
        if self.categorical_features:
            categorical_encoder = self._create_categorical_encoder()
            
            # For target encoder, we need to fit with target variable
            if isinstance(categorical_encoder, ce.TargetEncoder):
                categorical_pipeline = Pipeline([
                    ('encoder', categorical_encoder)
                ])
            else:
                categorical_pipeline = Pipeline([
                    ('encoder', categorical_encoder)
                ])
            
            preprocessors.append(('cat', categorical_pipeline, self.categorical_features))
        
        # Create column transformer
        if preprocessors:
            self.preprocessor_pipeline = ColumnTransformer(
                transformers=preprocessors,
                remainder='drop'
            )
            
            # Fit the pipeline
            if self.categorical_features and isinstance(self._create_categorical_encoder(), ce.TargetEncoder):
                # For target encoder, we need to pass y during fit
                X_transformed = self.preprocessor_pipeline.fit_transform(X, y)
            else:
                X_transformed = self.preprocessor_pipeline.fit_transform(X)
        else:
            X_transformed = X.values
        
        # Convert to DataFrame for easier handling
        X_transformed_df = pd.DataFrame(X_transformed)
        
        # Feature selection
        if self.feature_selector is not None:
            logger.info("Performing feature selection...")
            X_transformed_df = pd.DataFrame(
                self.feature_selector.fit_transform(X_transformed_df, y)
            )
            self.selected_features = self.feature_selector.get_support(indices=True)
            logger.info(f"Selected {len(self.selected_features)} features out of {X_transformed.shape[1]}")
        
        # Store feature names after encoding
        self.feature_names_after_encoding = [f'feature_{i}' for i in range(X_transformed_df.shape[1])]
        
        self.is_fitted = True
        logger.info("Data preprocessor fitting completed!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply preprocessing pipeline
        if self.preprocessor_pipeline is not None:
            X_transformed = self.preprocessor_pipeline.transform(X)
        else:
            X_transformed = X.values
        
        # Convert to DataFrame
        X_transformed_df = pd.DataFrame(X_transformed)
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_transformed_df = pd.DataFrame(
                self.feature_selector.transform(X_transformed_df)
            )
        
        # Set column names
        X_transformed_df.columns = self.feature_names_after_encoding
        
        return X_transformed_df
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit preprocessor, transform data, and apply resampling.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Tuple of (resampled_features, resampled_labels)
        """
        # First apply preprocessing
        X_transformed = self.fit_transform(X, y)
        
        # Apply resampling if enabled
        if self.resampler is not None:
            logger.info(f"Applying resampling with {type(self.resampler).__name__}")
            logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
            
            X_resampled, y_resampled = self.resampler.fit_resample(X_transformed, y)
            
            # Convert back to pandas
            X_resampled = pd.DataFrame(X_resampled, columns=X_transformed.columns)
            y_resampled = pd.Series(y_resampled)
            
            logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
            
            return X_resampled, y_resampled
        else:
            return X_transformed, y
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        return self.feature_names_after_encoding
    
    def get_feature_importance_mapping(self) -> Dict[str, float]:
        """Get feature importance from feature selector if available."""
        if self.feature_selector is None or not self.is_fitted:
            return {}
        
        if hasattr(self.feature_selector, 'scores_'):
            scores = self.feature_selector.scores_
            selected_indices = self.feature_selector.get_support(indices=True)
            
            importance_mapping = {}
            for i, idx in enumerate(selected_indices):
                feature_name = self.feature_names_after_encoding[i]
                importance_mapping[feature_name] = scores[idx]
            
            return importance_mapping
        
        return {}
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing information."""
        info = {
            'is_fitted': self.is_fitted,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'scaling_method': self.preprocessing_config['scaling_method'],
            'categorical_encoding': self.preprocessing_config['categorical_encoding'],
            'feature_selection_enabled': self.preprocessing_config['feature_selection']['enabled'],
            'resampling_enabled': self.preprocessing_config['resampling']['enabled']
        }
        
        if self.is_fitted:
            info.update({
                'num_features_after_preprocessing': len(self.feature_names_after_encoding),
                'feature_names_after_preprocessing': self.feature_names_after_encoding
            })
            
            if self.feature_selector is not None:
                info.update({
                    'feature_selection_method': self.preprocessing_config['feature_selection']['method'],
                    'num_selected_features': len(self.selected_features),
                    'selected_feature_indices': self.selected_features.tolist()
                })
            
            if self.resampler is not None:
                info['resampling_method'] = self.preprocessing_config['resampling']['method']
        
        return info
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'preprocessor_pipeline': self.preprocessor_pipeline,
            'feature_selector': self.feature_selector,
            'resampler': self.resampler,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'selected_features': self.selected_features,
            'feature_names_after_encoding': self.feature_names_after_encoding,
            'preprocessing_config': self.preprocessing_config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a fitted preprocessor.
        
        Args:
            filepath: Path to load the preprocessor from
        """
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor_pipeline = preprocessor_data['preprocessor_pipeline']
        self.feature_selector = preprocessor_data['feature_selector']
        self.resampler = preprocessor_data['resampler']
        self.numerical_features = preprocessor_data['numerical_features']
        self.categorical_features = preprocessor_data['categorical_features']
        self.selected_features = preprocessor_data['selected_features']
        self.feature_names_after_encoding = preprocessor_data['feature_names_after_encoding']
        self.preprocessing_config = preprocessor_data['preprocessing_config']
        self.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_loader import GermanCreditDataLoader
    
    # Load data
    loader = GermanCreditDataLoader()
    features, targets = loader.load_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit and transform
    X_transformed, y_resampled = preprocessor.fit_resample(features, targets)
    
    print("Preprocessing completed!")
    print(f"Original shape: {features.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Original class distribution: {targets.value_counts().to_dict()}")
    print(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
    
    # Get preprocessing info
    info = preprocessor.get_preprocessing_info()
    print(f"Preprocessing info: {info}")