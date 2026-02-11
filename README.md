# MCH-Guard: Machine Learning Framework for Microhemorrhage Prediction and Analysis

A comprehensive machine learning framework for predicting, analyzing, and monitoring microhemorrhage (MCH) progression in neuroimaging studies. This research project implements multiple model families to address different clinical questions related to MCH occurrence, timing, and progression.

NOTE: THIS REPOSITORY IS STILL UNDER CONSTRUCTION 

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Families](#model-families)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Statistical Analysis](#statistical-analysis)
- [Results and Visualizations](#results-and-visualizations)
- [Requirements](#requirements)
- [License](#license)

## Overview

MCH-Guard is a research framework designed to predict and analyze microhemorrhage (MCH) events using longitudinal neuroimaging and clinical data. The project implements four distinct model families, each addressing different aspects of MCH prediction:

- **Classification (CLS)**: Binary classification to predict MCH occurrence
- **Regression (RG)**: Predict time-to-MCH or duration of MCH events
- **Survival Analysis (SRV)**: Cox proportional hazards models for time-to-event analysis
- **Switch Prediction (SW)**: Predict medication switches related to MCH progression

Each model family includes three variants (M1, M2, M3) representing different feature set sizes, allowing for comparison of model performance across varying levels of feature complexity.

## Features

- **Multi-model Framework**: Four distinct model families for comprehensive MCH analysis
- **Progressive Model Sizes**: Three variants (M1, M2, M3) per family for feature complexity analysis
- **Automated Hyperparameter Tuning**: Optuna-based optimization for optimal model performance
- **Statistical Validation**: Comprehensive statistical tests including bootstrap confidence intervals, McNemar's test, and permutation tests
- **Reproducible Research**: Group-based cross-validation to prevent data leakage
- **Publication-Ready Visualizations**: Automated generation of ROC curves, confusion matrices, feature importance plots, and statistical analysis figures
- **Intelligent Training Mode**: Progressive training with automatic Optuna trial adjustment for statistical significance

## Model Families

### 1. Classification Models (CLS)
**Purpose**: Predict binary MCH occurrence (MCH_pos = 0/1)

- **M1**: Small feature set model
- **M2**: Medium feature set model  
- **M3**: Large feature set model

**Algorithms**: Random Forest Classifier with Optuna hyperparameter optimization

**Metrics**: Accuracy, ROC-AUC, F1-score, Precision, Recall

**Outputs**: 
- Trained models: `models/cls_m1_model.joblib`, `cls_m2_model.joblib`, `cls_m3_model.joblib`
- Test datasets: `processed/CLS_small_test.csv`, `CLS_medium_test.csv`, `CLS_large_test.csv`
- Visualizations: ROC curves, confusion matrices, feature importance plots

### 2. Regression Models (RG)
**Purpose**: Predict time-to-MCH or duration of MCH events (in years)

- **M1**: Small feature set model
- **M2**: Medium feature set model
- **M3**: Large feature set model

**Algorithms**: Extra Trees Regressor with RandomizedSearchCV hyperparameter optimization

**Metrics**: R² Score, Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)

**Outputs**:
- Trained models: `models/rg_m1_model.joblib`, `rg_m2_model.joblib`, `rg_m3_model.joblib`
- Test datasets: `processed/RG_small_test.csv`, `RG_medium_test.csv`, `RG_large_test.csv`
- Visualizations: Feature importance plots, prediction scatter plots

### 3. Survival Analysis Models (SRV)
**Purpose**: Time-to-event analysis using Cox proportional hazards models

- **M1**: Small feature set model
- **M2**: Medium feature set model
- **M3**: Large feature set model

**Algorithms**: Cox Proportional Hazards (CoxPH) from lifelines library

**Metrics**: Concordance Index (C-index, higher is better), AIC (Akaike Information Criterion, lower is better)

**Outputs**:
- Trained models: `models/srv_m1_coxph.joblib`, `srv_m2_coxph.joblib`, `srv_m3_coxph.joblib`
- Test datasets: `processed/SRV_small_test.csv`, `SRV_medium_test.csv`, `SRV_large_test.csv`
- Visualizations: Kaplan-Meier curves, hazard ratio plots

### 4. Switch Prediction Models (SW)
**Purpose**: Predict medication switches related to MCH progression

- **M1**: Small feature set model
- **M2**: Medium feature set model
- **M3**: Large feature set model

**Algorithms**: XGBoost Classifier with optional Optuna hyperparameter optimization

**Metrics**: Accuracy, ROC-AUC, F1-score, Precision, Recall

**Outputs**:
- Trained models: `models/sw_m1_model.joblib`, `sw_m2_model.joblib`, `sw_m3_model.joblib`
- Test datasets: `processed/SW_small_test.csv`, `SW_medium_test.csv`, `SW_large_test.csv`
- Visualizations: ROC curves, confusion matrices, feature importance plots

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MCH-Guard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, optuna, lifelines; print('Dependencies installed successfully')"
   ```

## Project Structure

```
MCH-Guard/
├── models/                          # Trained model files (.joblib)
│   ├── cls_m1_model.joblib
│   ├── cls_m2_model.joblib
│   ├── cls_m3_model.joblib
│   ├── rg_m1_model.joblib
│   ├── rg_m2_model.joblib
│   ├── rg_m3_model.joblib
│   ├── srv_m1_coxph.joblib
│   ├── srv_m2_coxph.joblib
│   ├── srv_m3_coxph.joblib
│   └── ...
│
├── notebooks/                       # Training and analysis scripts
│   ├── CLS/                        # Classification models
│   │   ├── CLS_M1_Train.py
│   │   ├── CLS_M2_Train.py
│   │   ├── CLS_M3_Train.py
│   │   ├── CLS_Statistical_Analysis.py
│   │   └── create_publication_figure.py
│   ├── RG/                         # Regression models
│   │   ├── RG_M1_Train.py
│   │   ├── RG_M2_Train.py
│   │   ├── RG_M3_Train.py
│   │   ├── RG_Statistical_Analysis.py
│   │   └── create_publication_figure.py
│   ├── SRV/                        # Survival analysis models
│   │   ├── SRV_M1_Train.py
│   │   ├── SRV_M2_Train.py
│   │   ├── SRV_M3_Train.py
│   │   ├── SRV_Statistical_Analysis.py
│   │   └── create_publication_figure.py
│   ├── SW/                         # Switch prediction models
│   │   ├── SW_M1_Train.py
│   │   ├── SW_M2_Train.py
│   │   ├── SW_M3_Train.py
│   │   ├── SW_Statistical_Analysis.py
│   │   └── create_publication_figure.py
│   ├── EDA/                        # Exploratory data analysis
│   │   ├── MCH.py
│   │   └── ...
│   └── describe_cohort.py          # Cohort description utilities
│
├── process_py/                      # Data processing scripts
│   ├── data_process/                # Individual data source processors
│   │   ├── adsp_process.py
│   │   ├── apoe_process.py
│   │   ├── biomarker_process.py
│   │   ├── cdr_process.py
│   │   ├── demo_process.py
│   │   ├── dxsum_process.py
│   │   ├── med_hist_process.py
│   │   ├── nfl_process.py
│   │   ├── recmed_process.py
│   │   ├── t2_process.py
│   │   └── wmh_process.py
│   └── merge_data.py                # Main data merging script
│
├── processed/                       # Processed datasets
│   ├── classification_small.csv
│   ├── classification_medium.csv
│   ├── classification_large.csv
│   ├── worsening_small.csv
│   ├── worsening_medium.csv
│   ├── worsening_large.csv
│   ├── longitudinal_progression.csv
│   └── ...
│
├── viz/                             # Generated visualizations
│   ├── classification_results/
│   ├── regression_results/
│   ├── survival_results/
│   └── switch_results/
│
├── train_all.py                     # Main training orchestration script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Usage

### Data Processing

Before training models, process the raw data:

```bash
cd process_py
python merge_data.py
```

This will generate the processed datasets in the `processed/` directory:
- `classification_small.csv`, `classification_medium.csv`, `classification_large.csv` (for CLS models)
- `worsening_small.csv`, `worsening_medium.csv`, `worsening_large.csv` (for RG and SRV models)
- `longitudinal_progression.csv` (for SW models)

### Model Training

#### Option 1: Train All Models (Standard Mode)

Train all model families and sizes:

```bash
python train_all.py
```

Train specific families:

```bash
python train_all.py --families CLS RG
```

Train specific sizes:

```bash
python train_all.py --sizes M1 M2
```

Train in parallel:

```bash
python train_all.py --parallel --max-workers 4
```

#### Option 2: Intelligent Training Mode (CLS Only)

Progressive training with automatic Optuna trial adjustment for statistical significance:

```bash
python train_all.py --intelligent --families CLS
```

This mode:
- Trains models sequentially (M1 → M2 → M3)
- Automatically increases Optuna n_trials if statistical significance is not achieved
- Ensures each model size shows statistically significant improvement over the previous

**Intelligent Mode Parameters**:
```bash
python train_all.py --intelligent \
    --initial-n-trials 100 \
    --n-trials-increment 50 \
    --max-n-trials 300 \
    --min-improvement 0.02
```

#### Option 3: Train Individual Models

Train a specific model directly:

```bash
# Classification M1
python notebooks/CLS/CLS_M1_Train.py

# Regression M2
python notebooks/RG/RG_M2_Train.py

# Survival M3
python notebooks/SRV/SRV_M3_Train.py

# Switch M2 with hyperparameter optimization
python notebooks/SW/SW_M2_Train.py --hpo
```

### Statistical Analysis

Run statistical comparisons between model variants:

```bash
# Classification models
python notebooks/CLS/CLS_Statistical_Analysis.py

# Regression models
python notebooks/RG/RG_Statistical_Analysis.py

# Survival models
python notebooks/SRV/SRV_Statistical_Analysis.py

# Switch models
python notebooks/SW/SW_Statistical_Analysis.py
```

Statistical analyses include:
- Bootstrap confidence intervals
- McNemar's test for paired comparisons
- Permutation tests
- Friedman test for multiple related samples
- Post-hoc Nemenyi test for pairwise comparisons

### Generate Publication Figures

Create publication-ready figures:

```bash
# Classification
python notebooks/CLS/create_publication_figure.py

# Regression
python notebooks/RG/create_publication_figure.py

# Survival
python notebooks/SRV/create_publication_figure.py
```

## Data Processing

The data processing pipeline integrates multiple data sources:

### Data Sources

- **Demographics**: Age, gender, education, race/ethnicity
- **APOE Genotype**: APOE ε4 allele status
- **Biomarkers**: Neurofilament light chain (NFL), other biomarkers
- **Clinical Assessments**: CDR (Clinical Dementia Rating), cognitive scores
- **Medical History**: Comorbidities (psychiatric, neurological, cardiac, etc.)
- **Medications**: Categorized medication classes (AD/dementia, lipid-lowering, blood pressure, etc.)
- **Neuroimaging**: T2-weighted images, white matter hyperintensities (WMH)
- **MCH Data**: Microhemorrhage counts and positive flags

### Processing Steps

1. **Individual Data Processing**: Each data source is processed independently
   - Missing value handling
   - Data type conversion
   - Feature engineering
   - Normalization/scaling

2. **Data Merging**: All sources are merged by RID (Research ID) and SCANDATE
   - Temporal alignment
   - Handling of missing visits
   - Feature selection based on model size (M1/M2/M3)

3. **Dataset Creation**:
   - **Classification datasets**: For predicting MCH occurrence
   - **Worsening datasets**: For predicting time-to-MCH or duration
   - **Longitudinal progression**: For switch prediction

## Model Training

### Hyperparameter Optimization

Models use different hyperparameter optimization strategies:

- **CLS (M2, M3)**: Optuna with configurable n_trials (default: 50-300)
- **RG (M2, M3)**: RandomizedSearchCV
- **SRV**: Built-in CoxPH optimization
- **SW (M2, M3)**: Optuna (optional, enabled with `--hpo` flag)

### Cross-Validation

All models use **GroupKFold** cross-validation to prevent data leakage:
- Groups are defined by RID (Research ID)
- Ensures no patient appears in both training and validation sets
- Maintains temporal relationships

### Class Imbalance Handling

Classification models (CLS, SW) use **SMOTE** (Synthetic Minority Oversampling Technique) to handle class imbalance.

## Statistical Analysis

Each model family includes comprehensive statistical analysis:

### Classification Models
- Bootstrap confidence intervals for accuracy, AUC, F1-score
- McNemar's test for paired model comparisons
- Permutation tests for significance testing
- Friedman test for multiple model comparison

### Regression Models
- Bootstrap confidence intervals for R², MSE, RMSE, MAE
- Paired t-tests or Wilcoxon signed-rank tests
- Effect size calculations (Cohen's d)

### Survival Models
- Bootstrap confidence intervals for concordance index
- Log-rank tests for survival curve comparisons
- Hazard ratio analysis

## Results and Visualizations

All results are saved in the `viz/` directory:

### Classification Results (`viz/classification_results/`)
- ROC curves for all model variants
- Confusion matrices
- Feature importance plots (Random Forest)
- Model comparison plots
- Statistical analysis reports

### Regression Results (`viz/regression_results/`)
- Feature importance plots
- Prediction scatter plots
- Residual analysis plots

### Survival Results (`viz/survival_results/`)
- Kaplan-Meier survival curves
- Hazard ratio plots
- Concordance index comparisons

### Switch Results (`viz/switch_results/`)
- ROC curves
- Confusion matrices
- Feature importance plots

## Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:

- `pandas>=2.0.0`: Data manipulation
- `scikit-learn>=1.3.0`: Machine learning algorithms
- `optuna`: Hyperparameter optimization
- `lifelines>=0.27.0`: Survival analysis
- `xgboost`: Gradient boosting for switch models
- `matplotlib>=3.7.0`: Plotting
- `seaborn`: Statistical visualizations
- `plotly>=5.15.0`: Interactive plots
- `joblib>=1.3.0`: Model serialization
- `imbalanced-learn`: SMOTE for class imbalance

## Environment Variables

The training scripts support several environment variables:

- `OPTUNA_N_TRIALS`: Number of Optuna trials for hyperparameter search (default: 50)
- `HYPERPARAMETER_SEARCH`: Enable/disable hyperparameter search for CLS/RG models (default: false for M1, true for M2/M3)
- `LOKY_MAX_CPU_COUNT`: Maximum CPU cores for parallel processing (default: 4)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mch_guard,
  title = {MCH-Guard: Machine Learning Models for Microhemorrhage Prediction and Analysis},
  author = {[Author Names]},
  year = {2025},
  url = {[Repository URL]}
}
```

## Acknowledgments

This research uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The data processing and model development framework was designed to support reproducible research in neuroimaging and clinical prediction.

## Contact

For questions or issues, please open an issue on the repository or contact the research team.

---

**Note**: This is a research codebase. Ensure proper data handling and compliance with data use agreements when working with clinical data.
