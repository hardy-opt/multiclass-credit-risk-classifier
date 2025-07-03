#!/usr/bin/env python3
"""
Project setup script for multi-class credit risk assessment.
Creates necessary directories and files, downloads data, and validates installation.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create the necessary directory structure."""
    logger.info("Creating directory structure...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'results/models',
        'results/plots/confusion_matrix',
        'results/plots/roc_curve',
        'results/plots/feature_importance',
        'results/plots/shap_summary',
        'results/plots/risk_distribution',
        'results/reports',
        'results/metrics',
        'logs',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python modules."""
    logger.info("Creating __init__.py files...")
    
    init_files = {
        'src/__init__.py': '''"""
Multi-class Credit Risk Assessment Package
"""

__version__ = "1.0.0"
__author__ = "Credit Risk Assessment Team"
''',
        'src/data/__init__.py': '''"""
Data handling modules for credit risk assessment.
"""

from .data_loader import GermanCreditDataLoader
from .preprocessor import DataPreprocessor
from .risk_categorizer import RiskCategorizer

__all__ = [
    'GermanCreditDataLoader',
    'DataPreprocessor', 
    'RiskCategorizer'
]
''',
        'src/models/__init__.py': '''"""
Machine learning models for credit risk assessment.
"""

from .logistic_regression import LogisticRegressionCreditRiskModel
from .xgboost_model import XGBoostCreditRiskModel
from .random_forest import RandomForestCreditRiskModel
from .ensemble_model import EnsembleCreditRiskModel

__all__ = [
    'LogisticRegressionCreditRiskModel',
    'XGBoostCreditRiskModel',
    'RandomForestCreditRiskModel',
    'EnsembleCreditRiskModel'
]
''',
        'src/utils/__init__.py': '''"""
Utility modules for evaluation and visualization.
"""

from .evaluation import ModelEvaluator
from .visualization import ModelVisualizer

__all__ = [
    'ModelEvaluator',
    'ModelVisualizer'
]
''',
        'src/training/__init__.py': '''"""
Training pipeline modules.
"""

__all__ = []
''',
        'tests/__init__.py': '''"""
Test modules for credit risk assessment.
"""
'''
    }
    
    for filepath, content in init_files.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"Created: {filepath}")

def create_gitignore():
    """Create .gitignore file."""
    logger.info("Creating .gitignore file...")
    
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Data files
data/raw/*.csv
data/raw/*.data
data/processed/*.csv
!data/processed/.gitkeep

# Model files
results/models/*.pkl
results/models/*.joblib
!results/models/.gitkeep

# Logs
logs/*.log
!logs/.gitkeep

# OS
.DS_Store
Thumbs.db

# Configuration (if contains sensitive data)
# config/secrets.yaml

# Results
results/plots/*.png
results/plots/*.jpg
results/plots/*.pdf
results/reports/*.html
results/reports/*.pdf
!results/plots/.gitkeep
!results/reports/.gitkeep
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    logger.info("Created .gitignore file")

def create_placeholder_files():
    """Create placeholder files to maintain directory structure in git."""
    logger.info("Creating placeholder files...")
    
    placeholder_dirs = [
        'data/raw',
        'data/processed',
        'data/external',
        'results/models',
        'results/plots',
        'results/reports',
        'results/metrics',
        'logs'
    ]
    
    for directory in placeholder_dirs:
        placeholder_file = os.path.join(directory, '.gitkeep')
        with open(placeholder_file, 'w') as f:
            f.write('# Placeholder file to maintain directory structure\n')

def validate_dependencies():
    """Validate that required dependencies are installed."""
    logger.info("Validating dependencies...")
    
    # Map of import names to their actual package names
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',  # sklearn is the import name, scikit-learn is pip name
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'shap': 'shap',
        'optuna': 'optuna',
        'imblearn': 'imbalanced-learn',  # imblearn is the import name
        'category_encoders': 'category-encoders',  # category_encoders is the import name
        'yaml': 'pyyaml',  # yaml is the import name, pyyaml is pip name
        'joblib': 'joblib',
        'ucimlrepo': 'ucimlrepo'
    }
    
    missing_packages = []
    
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {pip_name} - OK")
        except ImportError:
            missing_packages.append(pip_name)
            logger.warning(f"❌ {pip_name} - MISSING")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        logger.info("✅ All dependencies are installed!")
        return True

def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        # Add src to path for testing
        sys.path.insert(0, 'src')
        from data.data_loader import GermanCreditDataLoader
        
        # Initialize data loader
        loader = GermanCreditDataLoader()
        
        # Test data loading
        features, targets = loader.load_data()
        
        logger.info(f"✅ Data loaded successfully: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Validate data
        is_valid = loader.validate_data(features, targets)
        if is_valid:
            logger.info("✅ Data validation passed")
        else:
            logger.warning("⚠️ Data validation failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {str(e)}")
        return False

def create_sample_scripts():
    """Create sample usage scripts."""
    logger.info("Creating sample scripts...")
    
    # Quick start script
    quick_start_content = '''#!/usr/bin/env python3
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
    
    print("\\n🎉 Results:")
    print(f"   Accuracy: {evaluation['accuracy']:.4f}")
    print(f"   F1-Macro: {evaluation['f1_macro']:.4f}")
    print(f"   F1-Weighted: {evaluation['f1_weighted']:.4f}")
    
    print("\\n✅ Quick start completed successfully!")
    print("   Check the notebooks/ directory for detailed analysis.")

if __name__ == "__main__":
    main()
'''
    
    with open('scripts/quick_start.py', 'w') as f:
        f.write(quick_start_content)
    
    # Make script executable
    os.chmod('scripts/quick_start.py', 0o755)
    logger.info("Created scripts/quick_start.py")

def main():
    """Main setup function."""
    print("🏗️  Setting up Multi-Class Credit Risk Assessment Project")
    print("=" * 60)
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Create __init__.py files
        create_init_files()
        
        # Create .gitignore
        create_gitignore()
        
        # Create placeholder files
        create_placeholder_files()
        
        # Create sample scripts
        create_sample_scripts()
        
        # Validate dependencies
        deps_ok = validate_dependencies()
        
        if deps_ok:
            # Test data loading
            data_ok = test_data_loading()
            
            if data_ok:
                print("\\n" + "=" * 60)
                print("🎉 PROJECT SETUP COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print("\\n📋 Next Steps:")
                print("1. 📓 Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
                print("2. 🚀 Quick start: python scripts/quick_start.py")
                print("3. 🏋️  Full training: python scripts/train_models.py")
                print("4. 📊 Check results in the results/ directory")
                print("\\n🔧 Configuration:")
                print("   - Edit config/config.yaml to customize settings")
                print("   - Check requirements.txt for dependencies")
                print("\\n📚 Documentation:")
                print("   - README.md: Project overview and usage")
                print("   - docs/: Detailed documentation")
            else:
                print("\\n⚠️  Setup completed with warnings. Check data loading issues.")
        else:
            print("\\n❌ Setup incomplete. Please install missing dependencies.")
            return 1
            
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)