"""
Data loader module for German Credit Risk dataset.
Handles downloading, loading, and initial data validation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from ucimlrepo import fetch_ucirepo
import yaml

logger = logging.getLogger(__name__)


class GermanCreditDataLoader:
    """Data loader for German Credit Risk dataset from UCI repository."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.raw_data_path = self.data_config['raw_data_path']
        self.uci_dataset_id = self.data_config['uci_dataset_id']
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        # Feature names mapping
        self.feature_names = {
            'Attribute1': 'checking_account_status',
            'Attribute2': 'duration_months',
            'Attribute3': 'credit_history',
            'Attribute4': 'purpose',
            'Attribute5': 'credit_amount',
            'Attribute6': 'savings_account',
            'Attribute7': 'employment_since',
            'Attribute8': 'installment_rate',
            'Attribute9': 'personal_status_sex',
            'Attribute10': 'other_debtors',
            'Attribute11': 'residence_since',
            'Attribute12': 'property',
            'Attribute13': 'age_years',
            'Attribute14': 'other_installment_plans',
            'Attribute15': 'housing',
            'Attribute16': 'existing_credits',
            'Attribute17': 'job',
            'Attribute18': 'num_dependents',
            'Attribute19': 'telephone',
            'Attribute20': 'foreign_worker'
        }
        
        # Categorical feature mappings
        self.categorical_mappings = self._get_categorical_mappings()
    
    def download_data(self) -> None:
        """Download data from UCI repository."""
        try:
            logger.info("Downloading German Credit Data from UCI repository...")
            dataset = fetch_ucirepo(id=self.uci_dataset_id)
            
            # Extract features and targets
            X = dataset.data.features
            y = dataset.data.targets
            
            # Rename columns to meaningful names
            X.columns = [self.feature_names.get(col, col) for col in X.columns]
            
            # Save raw data
            X.to_csv(os.path.join(self.raw_data_path, 'features.csv'), index=False)
            y.to_csv(os.path.join(self.raw_data_path, 'targets.csv'), index=False)
            
            # Save combined data
            combined_data = pd.concat([X, y], axis=1)
            combined_data.to_csv(os.path.join(self.raw_data_path, 'german_credit_raw.csv'), index=False)
            
            # Save metadata
            metadata = {
                'dataset_info': dataset.metadata,
                'variable_info': dataset.variables,
                'feature_names': self.feature_names,
                'categorical_mappings': self.categorical_mappings
            }
            
            with open(os.path.join(self.raw_data_path, 'metadata.yaml'), 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            logger.info("Data downloaded successfully!")
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def load_data(self, use_cached: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the German Credit dataset.
        
        Args:
            use_cached: Whether to use cached data if available
            
        Returns:
            Tuple of (features, targets)
        """
        features_path = os.path.join(self.raw_data_path, 'features.csv')
        targets_path = os.path.join(self.raw_data_path, 'targets.csv')
        
        if not use_cached or not (os.path.exists(features_path) and os.path.exists(targets_path)):
            self.download_data()
        
        try:
            features = pd.read_csv(features_path)
            targets = pd.read_csv(targets_path).squeeze()  # Convert to Series
            
            logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict:
        """Get comprehensive information about the dataset."""
        features, targets = self.load_data()
        
        info = {
            'shape': features.shape,
            'target_distribution': targets.value_counts().to_dict(),
            'missing_values': features.isnull().sum().to_dict(),
            'data_types': features.dtypes.to_dict(),
            'numerical_features': list(features.select_dtypes(include=[np.number]).columns),
            'categorical_features': list(features.select_dtypes(include=['object']).columns),
            'target_names': {1: 'Good Credit', 2: 'Bad Credit'}
        }
        
        return info
    
    def validate_data(self, features: pd.DataFrame, targets: pd.Series) -> bool:
        """
        Validate the loaded data for consistency and completeness.
        
        Args:
            features: Feature dataframe
            targets: Target series
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if data is not empty
            if features.empty or targets.empty:
                logger.error("Data is empty")
                return False
            
            # Check if shapes match
            if len(features) != len(targets):
                logger.error("Features and targets have different lengths")
                return False
            
            # Check target values (should be 1 or 2)
            unique_targets = set(targets.unique())
            expected_targets = {1, 2}
            if not unique_targets.issubset(expected_targets):
                logger.error(f"Unexpected target values: {unique_targets}")
                return False
            
            # Check for expected number of features (should be 20)
            if features.shape[1] != 20:
                logger.warning(f"Expected 20 features, got {features.shape[1]}")
            
            # Check for expected number of samples (should be 1000)
            if len(features) != 1000:
                logger.warning(f"Expected 1000 samples, got {len(features)}")
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {str(e)}")
            return False
    
    def _get_categorical_mappings(self) -> Dict:
        """Get mappings for categorical variables."""
        return {
            'checking_account_status': {
                'A11': '< 0 DM',
                'A12': '0 <= ... < 200 DM',
                'A13': '>= 200 DM',
                'A14': 'no checking account'
            },
            'credit_history': {
                'A30': 'no credits taken/all credits paid back duly',
                'A31': 'all credits at this bank paid back duly',
                'A32': 'existing credits paid back duly till now',
                'A33': 'delay in paying off in the past',
                'A34': 'critical account/other credits existing'
            },
            'purpose': {
                'A40': 'car (new)', 'A41': 'car (used)', 
                'A42': 'furniture/equipment', 'A43': 'radio/television',
                'A44': 'domestic appliances', 'A45': 'repairs',
                'A46': 'education', 'A48': 'retraining',
                'A49': 'business', 'A410': 'others'
            },
            'savings_account': {
                'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM',
                'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM',
                'A65': 'unknown/no savings account'
            },
            'employment_since': {
                'A71': 'unemployed', 'A72': '< 1 year',
                'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years',
                'A75': '>= 7 years'
            },
            'personal_status_sex': {
                'A91': 'male : divorced/separated',
                'A92': 'female : divorced/separated/married',
                'A93': 'male : single',
                'A94': 'male : married/widowed',
                'A95': 'female : single'
            },
            'other_debtors': {
                'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'
            },
            'property': {
                'A121': 'real estate',
                'A122': 'building society savings agreement/life insurance',
                'A123': 'car or other',
                'A124': 'unknown/no property'
            },
            'other_installment_plans': {
                'A141': 'bank', 'A142': 'stores', 'A143': 'none'
            },
            'housing': {
                'A151': 'rent', 'A152': 'own', 'A153': 'for free'
            },
            'job': {
                'A171': 'unemployed/unskilled - non-resident',
                'A172': 'unskilled - resident',
                'A173': 'skilled employee/official',
                'A174': 'management/self-employed/highly qualified employee/officer'
            },
            'telephone': {
                'A191': 'none', 'A192': 'yes, registered under customers name'
            },
            'foreign_worker': {
                'A201': 'yes', 'A202': 'no'
            }
        }


if __name__ == "__main__":
    # Example usage
    loader = GermanCreditDataLoader()
    features, targets = loader.load_data()
    
    print("Data Info:")
    info = loader.get_data_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Validate data
    is_valid = loader.validate_data(features, targets)
    print(f"Data is valid: {is_valid}")