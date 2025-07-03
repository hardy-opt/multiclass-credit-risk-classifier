#!/usr/bin/env python3
"""
Debug script to test each model individually.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_model(model_name, model_class):
    """Test a specific model."""
    print(f"\n🧪 Testing {model_name}...")
    print("=" * 50)
    
    try:
        # Load data
        from data.data_loader import GermanCreditDataLoader
        from data.risk_categorizer import RiskCategorizer
        from data.preprocessor import DataPreprocessor
        
        print("📊 Loading data...")
        loader = GermanCreditDataLoader()
        features, targets = loader.load_data()
        
        print("🎯 Creating risk categories...")
        categorizer = RiskCategorizer()
        risk_categories = categorizer.create_risk_categories(features, targets)
        
        print("⚙️ Preprocessing...")
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(features, risk_categories)
        
        print("✂️ Splitting data...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, risk_categories, test_size=0.2, random_state=42, stratify=risk_categories
        )
        
        print(f"🤖 Training {model_name}...")
        model = model_class()
        
        # Train without hyperparameter tuning for speed
        model.train(X_train, y_train)
        
        print("📈 Evaluating...")
        evaluation = model.evaluate(X_test, y_test)
        
        print(f"✅ {model_name} SUCCESS!")
        print(f"   Accuracy: {evaluation['accuracy']:.4f}")
        print(f"   F1-Macro: {evaluation['f1_macro']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {model_name} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all models individually."""
    print("🔍 DEBUGGING INDIVIDUAL MODELS")
    print("=" * 60)
    
    models_to_test = [
        ("Logistic Regression", "LogisticRegressionCreditRiskModel"),
        ("XGBoost", "XGBoostCreditRiskModel"),
        ("Random Forest", "RandomForestCreditRiskModel"),
    ]
    
    results = {}
    
    for model_name, class_name in models_to_test:
        try:
            if class_name == "LogisticRegressionCreditRiskModel":
                from models.logistic_regression import LogisticRegressionCreditRiskModel
                model_class = LogisticRegressionCreditRiskModel
            elif class_name == "XGBoostCreditRiskModel":
                from models.xgboost_model import XGBoostCreditRiskModel
                model_class = XGBoostCreditRiskModel
            elif class_name == "RandomForestCreditRiskModel":
                from models.random_forest import RandomForestCreditRiskModel
                model_class = RandomForestCreditRiskModel
            
            success = test_model(model_name, model_class)
            results[model_name] = success
            
        except ImportError as e:
            print(f"❌ {model_name} IMPORT FAILED: {str(e)}")
            results[model_name] = False
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name}: {status}")
    
    successful_models = sum(results.values())
    total_models = len(results)
    
    print(f"\nTotal: {successful_models}/{total_models} models working")
    
    if successful_models < total_models:
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Check the failed models' error messages above")
        print("2. Verify all dependencies are installed correctly")
        print("3. Check config/config.yaml for enabled models")

if __name__ == "__main__":
    main()
