#!/usr/bin/env python3
"""
Quick start script for credit risk assessment.
Demonstrates basic usage of the framework.
"""

import sys
import os
sys.path.insert(0, 'src')

from data.data_loader import GermanCreditDataLoader
from data.risk_categorizer import RiskCategorizer
from data.preprocessor import DataPreprocessor
from models.xgboost_model import XGBoostCreditRiskModel

def main():
    print("🚀 Credit Risk Assessment - Quick Start")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    loader = GermanCreditDataLoader()
    features, targets = loader.load_data()
    print(f"   Loaded {features.shape[0]} samples with {features.shape[1]} features")
    
    # Create risk categories
    print("🎯 Creating risk categories...")
    categorizer = RiskCategorizer()
    risk_categories = categorizer.create_risk_categories(features, targets)
    analysis = categorizer.analyze_risk_distribution(risk_categories)
    print(f"   Created {analysis['total_samples']} samples across 5 risk categories")
    
    # Preprocess data
    print("⚙️ Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_processed, y_resampled = preprocessor.fit_resample(features, risk_categories)
    print(f"   Processed shape: {X_processed.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Train model
    print("🤖 Training XGBoost model...")
    model = XGBoostCreditRiskModel()
    model.train(X_train, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    evaluation = model.evaluate(X_test, y_test)
    
    print("\n🎉 Results:")
    print(f"   Accuracy: {evaluation['accuracy']:.4f}")
    print(f"   F1-Macro: {evaluation['f1_macro']:.4f}")
    print(f"   F1-Weighted: {evaluation['f1_weighted']:.4f}")
    
    print("\n✅ Quick start completed successfully!")
    print("   Check the notebooks/ directory for detailed analysis.")

if __name__ == "__main__":
    main()
