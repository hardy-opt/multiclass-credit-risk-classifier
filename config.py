# Configuration file for Multi-Class Credit Risk Assessment

# Data Configuration
data:
  raw_data_path: "data/raw/"
  processed_data_path: "data/processed/"
  external_data_path: "data/external/"
  
  # UCI Dataset ID for German Credit Data
  uci_dataset_id: 144
  
  # Train-test split parameters
  test_size: 0.2
  random_state: 42
  stratify: true

# Risk Categories Configuration
risk_categories:
  num_classes: 5
  class_names:
    0: "Very Low Risk"
    1: "Low Risk" 
    2: "Medium Risk"
    3: "High Risk"
    4: "Very High Risk"
  
  # Thresholds for converting binary to multi-class
  # Based on probability scores and additional features
  categorization_method: "probability_based"  # or "feature_based"
  
  # Cost matrix for imbalanced learning
  cost_matrix:
    # Cost of misclassifying actual class i as predicted class j
    # [actual][predicted]
    very_low: [0, 1, 3, 7, 10]
    low: [1, 0, 2, 5, 8]
    medium: [2, 1, 0, 3, 6]
    high: [5, 3, 2, 0, 2]
    very_high: [10, 8, 6, 3, 0]

# Preprocessing Configuration
preprocessing:
  # Feature scaling
  scaling_method: "standard"  # standard, minmax, robust
  
  # Categorical encoding
  categorical_encoding: "target"  # onehot, target, ordinal
  
  # Feature selection
  feature_selection:
    enabled: true
    method: "mutual_info"  # mutual_info, chi2, f_classif
    k_best: 15
  
  # Handling class imbalance
  resampling:
    enabled: true
    method: "smote"  # smote, adasyn, random_oversample
    
# Model Configuration
models:
  logistic_regression:
    enabled: true
    params:
      penalty: "elasticnet"
      l1_ratio: 0.5
      C: 1.0
      class_weight: "balanced"
      solver: "saga"
      max_iter: 1000
      multi_class: "ovr"
  
  xgboost:
    enabled: true
    params:
      objective: "multi:softprob"
      num_class: 5
      max_depth: 6
      learning_rate: 0.1
      n_estimators: 100
      subsample: 0.8
      colsample_bytree: 0.8
      eval_metric: "mlogloss"
      tree_method: "hist"
  
  random_forest:
    enabled: true
    params:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
      min_samples_leaf: 2
      class_weight: "balanced"
      oob_score: true
  
  ensemble:
    enabled: true
    method: "voting"  # voting, stacking
    voting_type: "soft"  # hard, soft

# Training Configuration
training:
  # Cross-validation
  cv_folds: 5
  cv_scoring: "f1_macro"
  
  # Hyperparameter tuning
  hyperparameter_tuning:
    enabled: true
    method: "optuna"  # optuna, grid_search, random_search
    n_trials: 100
    timeout: 3600  # seconds
  
  # Early stopping for tree-based models
  early_stopping:
    enabled: true
    rounds: 10
    
# Evaluation Configuration
evaluation:
  metrics:
    - "accuracy"
    - "precision_macro"
    - "recall_macro"
    - "f1_macro"
    - "f1_weighted"
    - "roc_auc_ovr"
    - "cohen_kappa"
  
  # Model interpretability
  interpretability:
    shap_enabled: true
    lime_enabled: true
    feature_importance: true
    partial_dependence: true

# Visualization Configuration
visualization:
  figure_size: [12, 8]
  dpi: 300
  style: "seaborn-v0_8"
  color_palette: "Set2"
  
  # Plot types to generate
  plots:
    - "confusion_matrix"
    - "classification_report"
    - "roc_curve"
    - "precision_recall_curve"
    - "feature_importance"
    - "shap_summary"
    - "risk_distribution"

# Output Configuration
output:
  models_path: "results/models/"
  plots_path: "results/plots/"
  reports_path: "results/reports/"
  metrics_path: "results/metrics/"
  
  # Model saving format
  model_format: "pickle"  # pickle, joblib
  
  # Report formats
  report_formats: 
    - "html"
    - "pdf"
    - "json"

# Logging Configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/credit_risk_assessment.log"