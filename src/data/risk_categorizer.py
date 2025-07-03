"""
Risk categorization module to convert binary classification to multi-class.
Transforms the original binary good/bad credit classification into 5 risk categories.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yaml

logger = logging.getLogger(__name__)


class RiskCategorizer:
    """
    Converts binary credit risk classification to multi-class risk categories.
    Uses multiple approaches including probability-based and feature-based categorization.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize risk categorizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk_categories']
        self.num_classes = self.risk_config['num_classes']
        self.class_names = self.risk_config['class_names']
        self.categorization_method = self.risk_config['categorization_method']
        
        # Initialize models for probability estimation
        self.probability_model = None
        self.scaler = StandardScaler()
        
        # Risk factors and their weights
        self.risk_factors = {
            'credit_amount': 0.25,
            'duration_months': 0.20,
            'age_years': -0.15,  # Negative because older age is less risky
            'checking_account_status': 0.15,
            'credit_history': 0.10,
            'savings_account': -0.10,  # Negative because more savings is less risky
            'employment_since': -0.05,  # Negative because longer employment is less risky
            'installment_rate': 0.10
        }
    
    def create_risk_categories(self, features: pd.DataFrame, targets: pd.Series) -> pd.Series:
        """
        Create multi-class risk categories from binary targets.
        
        Args:
            features: Feature dataframe
            targets: Binary target series (1=good, 2=bad)
            
        Returns:
            Multi-class risk categories (0-4)
        """
        if self.categorization_method == "probability_based":
            return self._probability_based_categorization(features, targets)
        elif self.categorization_method == "feature_based":
            return self._feature_based_categorization(features, targets)
        else:
            raise ValueError(f"Unknown categorization method: {self.categorization_method}")
    
    def _probability_based_categorization(self, features: pd.DataFrame, targets: pd.Series) -> pd.Series:
        """
        Create risk categories based on probability scores from a trained model.
        
        Args:
            features: Feature dataframe
            targets: Binary target series
            
        Returns:
            Multi-class risk categories
        """
        logger.info("Creating risk categories using probability-based method...")
        
        # Prepare features for modeling
        X_processed = self._prepare_features_for_scoring(features)
        
        # Train a probability estimation model
        self.probability_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Convert targets: 1 (good) -> 0, 2 (bad) -> 1 for probability estimation
        y_binary = (targets == 2).astype(int)
        
        self.probability_model.fit(X_processed, y_binary)
        
        # Get probability of being "bad credit"
        bad_credit_probs = self.probability_model.predict_proba(X_processed)[:, 1]
        
        # Create risk categories based on probability thresholds and additional factors
        risk_scores = self._calculate_comprehensive_risk_scores(features, bad_credit_probs)
        risk_categories = self._assign_risk_categories(risk_scores, targets)
        
        logger.info(f"Risk category distribution: {pd.Series(risk_categories).value_counts().to_dict()}")
        
        return pd.Series(risk_categories, index=features.index)
    
    def _feature_based_categorization(self, features: pd.DataFrame, targets: pd.Series) -> pd.Series:
        """
        Create risk categories based on feature combinations and business rules.
        
        Args:
            features: Feature dataframe
            targets: Binary target series
            
        Returns:
            Multi-class risk categories
        """
        logger.info("Creating risk categories using feature-based method...")
        
        # Calculate risk scores based on individual features
        risk_scores = np.zeros(len(features))
        
        # Normalize features for consistent scoring
        features_normalized = self._normalize_features_for_scoring(features)
        
        # Calculate weighted risk score using only numerical features
        for feature, weight in self.risk_factors.items():
            if feature in features_normalized.columns:
                # Only process numerical features
                if pd.api.types.is_numeric_dtype(features_normalized[feature]):
                    try:
                        feature_values = features_normalized[feature].values
                        if np.issubdtype(feature_values.dtype, np.number):
                            risk_scores += weight * feature_values
                    except Exception as e:
                        logger.warning(f"Error processing feature {feature}: {str(e)}")
        
        # Adjust scores based on categorical variables
        risk_scores = self._adjust_for_categorical_features(features, risk_scores)
        
        # Assign risk categories
        risk_categories = self._assign_risk_categories(risk_scores, targets)
        
        logger.info(f"Risk category distribution: {pd.Series(risk_categories).value_counts().to_dict()}")
        
        return pd.Series(risk_categories, index=features.index)
    
    def _prepare_features_for_scoring(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for probability model training."""
        X = features.copy()
        
        # Handle categorical variables with simple encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Simple label encoding for model training
            X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X
    
    def _normalize_features_for_scoring(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for risk scoring."""
        features_norm = features.copy()
        
        # Only normalize numerical features to 0-1 scale
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        risk_numerical_cols = ['credit_amount', 'duration_months', 'age_years', 'installment_rate']
        
        # Only process numerical columns that exist in the dataset
        for col in risk_numerical_cols:
            if col in numerical_cols and col in features_norm.columns:
                min_val = features_norm[col].min()
                max_val = features_norm[col].max()
                if max_val > min_val:  # Avoid division by zero
                    features_norm[col] = (features_norm[col] - min_val) / (max_val - min_val)
                else:
                    features_norm[col] = 0.0
        
        return features_norm
    
    def _calculate_comprehensive_risk_scores(self, features: pd.DataFrame, bad_credit_probs: np.ndarray) -> np.ndarray:
        """Calculate comprehensive risk scores combining probability and feature-based scoring."""
        # Start with probability scores (60% weight)
        risk_scores = 0.6 * bad_credit_probs
        
        # Add feature-based risk factors (40% weight)
        features_normalized = self._normalize_features_for_scoring(features)
        
        feature_risk = np.zeros(len(features))
        
        # Only use numerical features for mathematical operations
        for feature, weight in self.risk_factors.items():
            if feature in features_normalized.columns:
                # Check if the feature is numerical
                if pd.api.types.is_numeric_dtype(features_normalized[feature]):
                    try:
                        feature_values = features_normalized[feature].values
                        # Ensure we have numerical values
                        if np.issubdtype(feature_values.dtype, np.number):
                            feature_risk += weight * feature_values
                        else:
                            logger.warning(f"Skipping non-numerical feature {feature} in risk calculation")
                    except Exception as e:
                        logger.warning(f"Error processing feature {feature}: {str(e)}")
                else:
                    logger.debug(f"Skipping categorical feature {feature} in numerical risk calculation")
        
        # Normalize feature risk to 0-1 scale
        if feature_risk.std() > 0:
            feature_risk = (feature_risk - feature_risk.min()) / (feature_risk.max() - feature_risk.min())
        
        risk_scores += 0.4 * feature_risk
        
        return risk_scores
    
    def _adjust_for_categorical_features(self, features: pd.DataFrame, risk_scores: np.ndarray) -> np.ndarray:
        """Adjust risk scores based on categorical features."""
        adjusted_scores = risk_scores.copy()
        
        # Checking account status adjustments
        if 'checking_account_status' in features.columns:
            checking_account = features['checking_account_status']
            adjusted_scores += np.where(checking_account == 'A14', 0.1, 0)  # No checking account
            adjusted_scores += np.where(checking_account == 'A11', 0.2, 0)  # < 0 DM
            adjusted_scores -= np.where(checking_account == 'A13', 0.1, 0)  # >= 200 DM
        
        # Credit history adjustments
        if 'credit_history' in features.columns:
            credit_history = features['credit_history']
            adjusted_scores += np.where(credit_history == 'A34', 0.3, 0)  # Critical account
            adjusted_scores += np.where(credit_history == 'A33', 0.2, 0)  # Delay in past
            adjusted_scores -= np.where(credit_history == 'A30', 0.1, 0)  # No credits/all paid
        
        # Savings account adjustments
        if 'savings_account' in features.columns:
            savings = features['savings_account']
            adjusted_scores += np.where(savings == 'A65', 0.1, 0)  # Unknown/no savings
            adjusted_scores += np.where(savings == 'A61', 0.1, 0)  # < 100 DM
            adjusted_scores -= np.where(savings == 'A64', 0.2, 0)  # >= 1000 DM
        
        # Employment adjustments
        if 'employment_since' in features.columns:
            employment = features['employment_since']
            adjusted_scores += np.where(employment == 'A71', 0.3, 0)  # Unemployed
            adjusted_scores += np.where(employment == 'A72', 0.1, 0)  # < 1 year
            adjusted_scores -= np.where(employment == 'A75', 0.1, 0)  # >= 7 years
        
        # Purpose adjustments
        if 'purpose' in features.columns:
            purpose = features['purpose']
            adjusted_scores += np.where(purpose == 'A40', 0.1, 0)  # New car (high amount)
            adjusted_scores += np.where(purpose == 'A49', 0.1, 0)  # Business (risky)
            adjusted_scores -= np.where(purpose == 'A46', 0.05, 0)  # Education (investment)
        
        return adjusted_scores
    
    def _assign_risk_categories(self, risk_scores: np.ndarray, targets: pd.Series) -> np.ndarray:
        """
        Assign risk categories based on risk scores and original binary classification.
        
        Args:
            risk_scores: Calculated risk scores (0-1)
            targets: Original binary targets (1=good, 2=bad)
            
        Returns:
            Risk categories (0-4)
        """
        risk_categories = np.zeros(len(targets), dtype=int)
        
        # Convert targets to boolean for easier handling
        is_bad_credit = (targets == 2)
        
        # For good credit customers (original target = 1)
        good_credit_mask = ~is_bad_credit
        good_scores = risk_scores[good_credit_mask]
        
        if len(good_scores) > 0:
            # Divide good credit customers into Very Low (0) and Low Risk (1)
            good_threshold = np.percentile(good_scores, 70)
            risk_categories[good_credit_mask] = np.where(
                good_scores <= good_threshold, 0, 1  # 0: Very Low Risk, 1: Low Risk
            )
        
        # For bad credit customers (original target = 2)
        bad_credit_mask = is_bad_credit
        bad_scores = risk_scores[bad_credit_mask]
        
        if len(bad_scores) > 0:
            # Divide bad credit customers into Medium (2), High (3), and Very High Risk (4)
            bad_threshold_low = np.percentile(bad_scores, 33)
            bad_threshold_high = np.percentile(bad_scores, 67)
            
            risk_categories[bad_credit_mask] = np.where(
                bad_scores <= bad_threshold_low, 2,  # Medium Risk
                np.where(bad_scores <= bad_threshold_high, 3, 4)  # High Risk, Very High Risk
            )
        
        # Handle edge cases and refinement
        risk_categories = self._refine_risk_categories(risk_categories, risk_scores, targets)
        
        return risk_categories
    
    def _refine_risk_categories(self, risk_categories: np.ndarray, risk_scores: np.ndarray, 
                              targets: pd.Series) -> np.ndarray:
        """Refine risk categories based on additional business rules."""
        refined_categories = risk_categories.copy()
        
        # Ensure some good credit customers can be medium risk if their scores are high
        good_credit_mask = (targets == 1)
        high_risk_good = (risk_scores > 0.7) & good_credit_mask
        refined_categories[high_risk_good] = 2  # Upgrade to Medium Risk
        
        # Ensure some bad credit customers can be low/medium risk if their scores are very low
        bad_credit_mask = (targets == 2)
        low_risk_bad = (risk_scores < 0.3) & bad_credit_mask
        refined_categories[low_risk_bad] = np.maximum(refined_categories[low_risk_bad] - 1, 1)
        
        return refined_categories
    
    def get_risk_category_descriptions(self) -> Dict[int, Dict[str, str]]:
        """Get detailed descriptions for each risk category."""
        return {
            0: {
                'name': 'Very Low Risk',
                'description': 'Excellent credit profile with strong financial indicators',
                'characteristics': 'High savings, stable employment, good credit history, low debt-to-income ratio',
                'recommendation': 'Approve with standard terms'
            },
            1: {
                'name': 'Low Risk',
                'description': 'Good credit profile with minor concerns',
                'characteristics': 'Adequate savings, stable employment, mostly positive credit history',
                'recommendation': 'Approve with standard or slightly adjusted terms'
            },
            2: {
                'name': 'Medium Risk',
                'description': 'Moderate credit profile requiring careful evaluation',
                'characteristics': 'Mixed financial indicators, some credit history issues',
                'recommendation': 'Approve with modified terms, higher interest rates, or additional collateral'
            },
            3: {
                'name': 'High Risk',
                'description': 'Poor credit profile with significant concerns',
                'characteristics': 'Limited savings, unstable employment, negative credit history',
                'recommendation': 'Approve with strict terms, high interest rates, and strong collateral requirements'
            },
            4: {
                'name': 'Very High Risk',
                'description': 'Critical credit profile with major red flags',
                'characteristics': 'Poor financial indicators across multiple dimensions',
                'recommendation': 'Consider rejection or require significant risk mitigation measures'
            }
        }
    
    def analyze_risk_distribution(self, risk_categories: pd.Series) -> Dict:
        """Analyze the distribution of risk categories."""
        distribution = risk_categories.value_counts().sort_index()
        percentages = (distribution / len(risk_categories) * 100).round(2)
        
        analysis = {
            'total_samples': len(risk_categories),
            'distribution': distribution.to_dict(),
            'percentages': percentages.to_dict(),
            'class_names': {int(k): v for k, v in self.class_names.items()}
        }
        
        return analysis
    
    def get_feature_risk_contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Get risk contributions from each feature."""
        contributions = []
        
        features_normalized = self._normalize_features_for_scoring(features)
        
        for feature, weight in self.risk_factors.items():
            if feature in features_normalized.columns:
                # Only process numerical features
                if pd.api.types.is_numeric_dtype(features_normalized[feature]):
                    try:
                        feature_values = features_normalized[feature]
                        if np.issubdtype(feature_values.dtype, np.number):
                            contribution = weight * feature_values
                            contributions.append({
                                'feature': feature,
                                'weight': weight,
                                'mean_contribution': contribution.mean(),
                                'std_contribution': contribution.std(),
                                'min_contribution': contribution.min(),
                                'max_contribution': contribution.max()
                            })
                    except Exception as e:
                        logger.warning(f"Error calculating contribution for {feature}: {str(e)}")
        
        return pd.DataFrame(contributions)


if __name__ == "__main__":
    # Example usage
    from data_loader import GermanCreditDataLoader
    
    # Load data
    loader = GermanCreditDataLoader()
    features, targets = loader.load_data()
    
    # Create risk categorizer
    categorizer = RiskCategorizer()
    
    # Generate risk categories
    risk_categories = categorizer.create_risk_categories(features, targets)
    
    # Analyze distribution
    analysis = categorizer.analyze_risk_distribution(risk_categories)
    print("Risk Category Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Get risk descriptions
    descriptions = categorizer.get_risk_category_descriptions()
    print("\nRisk Category Descriptions:")
    for category, info in descriptions.items():
        print(f"\nClass {category} - {info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Recommendation: {info['recommendation']}")