#!/usr/bin/env python3
"""
Test script to debug XGBoost issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_xgboost_import():
    """Test XGBoost import and basic functionality."""
    try:
        import xgboost as xgb
        print(f"✅ XGBoost imported successfully: {xgb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
        return False

def test_xgboost_basic():
    """Test basic XGBoost functionality."""
    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, n_classes=5, 
                                 n_informative=8, random_state=42)
        
        # Test XGBoost native interface
        print("Testing XGBoost native interface...")
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'max_depth': 3,
            'learning_rate': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        preds = model.predict(dtrain)
        print(f"✅ Native interface works: predictions shape {preds.shape}")
        
        # Test scikit-learn interface
        print("Testing XGBoost scikit-learn interface...")
        from xgboost import XGBClassifier
        
        clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        clf.fit(X, y)
        sklearn_preds = clf.predict(X)
        print(f"✅ Scikit-learn interface works: predictions shape {sklearn_preds.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_with_project_data():
    """Test XGBoost with actual project data."""
    try:
        from data.data_loader import GermanCreditDataLoader
        from data.risk_categorizer import RiskCategorizer
        from data.preprocessor import DataPreprocessor
        
        print("Testing XGBoost with project data...")
        
        # Load data
        loader = GermanCreditDataLoader()
        features, targets = loader.load_data()
        
        # Create risk categories
        categorizer = RiskCategorizer()
        risk_categories = categorizer.create_risk_categories(features, targets)
        
        # Preprocess
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(features, risk_categories)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, risk_categories, test_size=0.2, random_state=42, stratify=risk_categories
        )
        
        # Test XGBoost with native interface
        print("Testing native XGBoost interface...")
        import xgboost as xgb
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'max_depth': 6,
            'learning_rate': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=10, verbose_eval=False)
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
        pred_labels = predictions.argmax(axis=1)
        accuracy = (pred_labels == y_test).mean()
        
        print(f"✅ XGBoost native interface works: accuracy = {accuracy:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ XGBoost project test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all XGBoost tests."""
    print("🔍 Testing XGBoost Installation and Functionality")
    print("=" * 50)
    
    # Test 1: Import
    import_ok = test_xgboost_import()
    if not import_ok:
        print("❌ XGBoost import failed. Please install: pip install xgboost")
        return
    
    # Test 2: Basic functionality
    basic_ok = test_xgboost_basic()
    if not basic_ok:
        print("❌ XGBoost basic functionality failed.")
        return
    
    # Test 3: Project integration
    project_ok = test_xgboost_with_project_data()
    if not project_ok:
        print("❌ XGBoost project integration failed.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All XGBoost tests passed!")
    print("XGBoost should work in the main training pipeline.")

if __name__ == "__main__":
    main()