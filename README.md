# Multi-Class Credit Risk Assessment

A comprehensive machine learning project that transforms the classic German Credit dataset from binary classification to a sophisticated **multi-class credit risk assessment** system, categorizing loan applicants into 5 distinct risk categories using advanced techniques.

##  Project Overview

This project extends the traditional binary good/bad credit classification into a nuanced 5-class risk assessment system:

- **Class 0 - Very Low Risk**: Excellent credit profile with strong financial indicators
- **Class 1 - Low Risk**: Good credit profile with minor concerns
- **Class 2 - Medium Risk**: Moderate credit profile requiring careful evaluation
- **Class 3 - High Risk**: Poor credit profile with significant concerns
- **Class 4 - Very High Risk**: Critical credit profile with major red flags

## 📁 Project Structure

credit-risk-multiclass/
├── data/                    # Data storage
│   ├── raw/                   # Original UCI dataset
│   ├── processed/             # Preprocessed data splits
│   └── external/              # Additional data sources
├── src/                     # Source code
│   ├── data/                  # Data handling modules
│   ├── models/                # ML model implementations
│   ├── utils/                 # Utility functions
│   └── training/              # Training orchestration
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Execution scripts
├── results/                 # Output results
│   ├── models/                # Saved trained models
│   ├── plots/                 # Visualizations
│   ├── reports/               # Generated reports
│   └── metrics/               # Performance metrics
├── config/                  # Configuration files
├── tests/                   # Unit tests


##  Key Features

### Advanced Machine Learning Models
- **Logistic Regression**: With L1/L2 regularization and feature selection
- **XGBoost**: Gradient boosting with hyperparameter optimization
- **Random Forest**: Ensemble method with feature importance analysis
- **Ensemble Model**: Combines multiple models for enhanced performance

### Sophisticated Preprocessing Pipeline
- Advanced feature engineering and selection
- Multiple categorical encoding strategies (target, one-hot, ordinal)
- Handling class imbalance with SMOTE/ADASYN
- Robust scaling and normalization techniques

### Model Interpretability & Analysis
- SHAP (SHapley Additive exPlanations) values for feature importance
- LIME (Local Interpretable Model-agnostic Explanations)
- Comprehensive feature importance rankings
- Cost-sensitive learning with custom cost matrices

### Comprehensive Evaluation Framework
- Multi-class confusion matrices and classification reports
- ROC curves and Precision-Recall curves for each class
- Cross-validation with stratified sampling
- Cost-aware evaluation metrics

##  Dataset Information

**Source**: UCI Machine Learning Repository - German Credit Data  
**Original Size**: 1,000 samples, 20 features  
**Original Task**: Binary classification (Good/Bad credit)  
**Enhanced Task**: 5-class risk categorization  

### Key Features Include:
- Checking account status and credit history
- Loan amount, duration, and purpose
- Employment status and savings account information
- Personal demographics and housing situation
- Existing credit obligations and guarantors



<!-- 
## Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd credit-risk-multiclass
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download and prepare data**
```bash
python scripts/download_data.py
python scripts/preprocess_data.py
```

5. **Train models**
```bash
# Train all models with hyperparameter tuning
python scripts/train_models.py

# Train specific models without tuning
python scripts/train_models.py --models xgboost logistic_regression --no-tune
```

6. **Evaluate and visualize results**
```bash
python scripts/evaluate_models.py
```

## 📁 Project Structure

```
credit-risk-multiclass/
├── 📊 data/                    # Data storage
│   ├── raw/                   # Original UCI dataset
│   ├── processed/             # Preprocessed data splits
│   └── external/              # Additional data sources
├── 🧠 src/                     # Source code
│   ├── data/                  # Data handling modules
│   ├── models/                # ML model implementations
│   ├── utils/                 # Utility functions
│   └── training/              # Training orchestration
├── 📓 notebooks/               # Jupyter notebooks for analysis
├── 🎯 scripts/                 # Execution scripts
├── 📈 results/                 # Output results
│   ├── models/                # Saved trained models
│   ├── plots/                 # Visualizations
│   ├── reports/               # Generated reports
│   └── metrics/               # Performance metrics
├── ⚙️ config/                  # Configuration files
├── 🧪 tests/                   # Unit tests
└── 📚 docs/                    # Documentation
```

## 🔧 Configuration

The project uses a comprehensive YAML configuration system. Key configuration areas:

### Data Configuration
```yaml
data:
  test_size: 0.2
  random_state: 42
  stratify: true
```

### Risk Categories
```yaml
risk_categories:
  num_classes: 5
  categorization_method: "probability_based"  # or "feature_based"
```

### Model Parameters
```yaml
models:
  xgboost:
    enabled: true
    params:
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
```

## 🚀 Usage Examples

### Basic Model Training
```python
from src.training.trainer import ModelTrainingPipeline

# Initialize and run complete pipeline
pipeline = ModelTrainingPipeline("config/config.yaml")
pipeline.run_complete_pipeline()
```

### Individual Model Usage
```python
from src.models.xgboost_model import XGBoostCreditRiskModel
from src.data.data_loader import GermanCreditDataLoader

# Load data
loader = GermanCreditDataLoader()
X, y = loader.load_data()

# Initialize and train model
model = XGBoostCreditRiskModel()
model.tune_hyperparameters(X_train, y_train)
model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Risk Category Analysis
```python
from src.data.risk_categorizer import RiskCategorizer

categorizer = RiskCategorizer()
risk_categories = categorizer.create_risk_categories(features, targets)
analysis = categorizer.analyze_risk_distribution(risk_categories)
```

## 📊 Model Performance

### Expected Performance Metrics
- **Accuracy**: 75-85% (varies by model)
- **F1-Macro Score**: 0.70-0.80
- **F1-Weighted Score**: 0.75-0.85

### Feature Importance Insights
Top contributing features typically include:
1. Checking account status
2. Credit history
3. Credit amount and duration
4. Savings account balance
5. Employment status

## 🎨 Visualizations

The project generates comprehensive visualizations:

- **Confusion Matrices**: Multi-class prediction accuracy
- **ROC Curves**: Performance for each risk class
- **Feature Importance**: Model-specific feature rankings
- **SHAP Plots**: Individual prediction explanations
- **Risk Distribution**: Analysis of category assignments

## 🧪 Model Interpretability

### SHAP Integration
```python
# Setup SHAP explainer
model.setup_shap_explainer(X_background)
shap_values = model.get_shap_values(X_test)

# Generate SHAP plots
visualizer.plot_shap_summary(shap_values, X_test, model_name)
```

### Feature Contribution Analysis
```python
# Get feature importance
importance_df = model.get_feature_importance(top_n=20)

# Analyze risk contributions
risk_contributions = categorizer.get_feature_risk_contributions(features)
```

## 🔬 Advanced Features

### Hyperparameter Optimization
- **Optuna-based** optimization with 100+ trials
- **Cross-validation** for robust parameter selection
- **Early stopping** to prevent overfitting

### Cost-Sensitive Learning
- Custom cost matrices reflecting business impact
- Class weighting for imbalanced data
- SMOTE/ADASYN for synthetic sample generation

### Ensemble Methods
- **Voting classifiers** (hard/soft voting)
- **Stacking** with meta-learners
- **Model averaging** with optimized weights

## 📈 Business Impact

### Risk Assessment Benefits
1. **Granular Risk Categorization**: 5 distinct risk levels vs. binary classification
2. **Improved Decision Making**: Tailored loan terms for each risk category
3. **Reduced Default Risk**: Better identification of high-risk applicants
4. **Optimized Pricing**: Risk-based pricing strategies

### Cost-Benefit Analysis
- **Cost Matrix Integration**: Reflects true business costs of misclassification
- **ROI Optimization**: Maximize profit while minimizing risk
- **Regulatory Compliance**: Transparent and explainable credit decisions

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_models.py -v
```

## 📊 Monitoring & Evaluation

### Model Performance Tracking
- Automated cross-validation scoring
- Performance drift detection
- A/B testing framework for model comparison

### Evaluation Metrics
```python
# Comprehensive evaluation
evaluation = model.evaluate(X_test, y_test)
print(f"Accuracy: {evaluation['accuracy']:.4f}")
print(f"F1-Macro: {evaluation['f1_macro']:.4f}")
print(f"F1-Weighted: {evaluation['f1_weighted']:.4f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Use meaningful commit messages

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for providing the German Credit dataset
- **Prof. Hans Hofmann** for the original dataset creation
- **Scikit-learn, XGBoost, and SHAP** communities for excellent ML libraries
- **Optuna** team for hyperparameter optimization framework

## 📞 Support & Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the [documentation](docs/) for detailed guides
- Review the [FAQ](docs/FAQ.md) for common questions

## 🚀 Future Enhancements

### Planned Features
- [ ] Deep learning models (Neural Networks)
- [ ] AutoML integration with FLAML/AutoGluon
- [ ] Real-time prediction API
- [ ] Model monitoring dashboard
- [ ] Fairness and bias detection tools
- [ ] Integration with MLflow for experiment tracking

### Research Directions
- [ ] Federated learning for privacy-preserving credit scoring
- [ ] Explainable AI techniques for regulatory compliance
- [ ] Time-series analysis for dynamic risk assessment
- [ ] Graph neural networks for relationship modeling

---

**Made with ❤️ for better credit risk assessment** -->