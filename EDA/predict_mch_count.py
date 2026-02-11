import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_recall_curve,
                            confusion_matrix, classification_report, roc_curve, auc, f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import time
import logging
import os
import platform
import subprocess
from imblearn.over_sampling import SMOTE

# Import Optuna for hyperparameter optimization
import optuna

# Set the number of CPU cores to use (adjust the number based on your system)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set to number of cores you want to use

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to log and save Optuna results
def log_optuna_results(study, model_name, output_dir="viz/classification_results/classification_results/optuna"):
    """Log and save Optuna study results for later analysis.
    
    Args:
        study: The Optuna study object
        model_name: Name of the model being optimized
        output_dir: Directory to save results
        
    Returns:
        pandas.DataFrame: DataFrame containing all trial results
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with all trial results
    study_results = []
    for trial in study.trials:
        # Extract parameters and results
        params = trial.params.copy()
        params['trial_number'] = trial.number
        params['value'] = trial.value if trial.value is not None else float('nan')
        params['state'] = trial.state.name
        params['datetime'] = trial.datetime_start.strftime('%Y-%m-%d %H:%M:%S')
        params['duration'] = (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
        
        study_results.append(params)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(study_results)
    # csv_path = f"{output_dir}/Large_{model_name}_optuna_results.csv"
    # results_df.to_csv(csv_path, index=False)
    # logger.info(f"Saved Optuna study results to {csv_path}")
    
    # Log summary statistics
    if len(study_results) > 0:
        logger.info(f"Optuna study summary for {model_name}:")
        logger.info(f"  Number of trials: {len(study.trials)}")
        logger.info(f"  Best value: {study.best_value:.4f}")
        logger.info(f"  Best parameters:")
        for param, value in study.best_params.items():
            logger.info(f"    {param}: {value}")
    
    return results_df


def visualize_optuna_results(study, model_name, output_dir="viz/classification_results/classification_results/optuna"):
    """Create additional visualizations for Optuna hyperparameter optimization results.
    
    Args:
        study: The Optuna study object
        model_name: Name of the model being optimized
        output_dir: Directory to save visualizations
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get a DataFrame of the study results
        df_results = pd.DataFrame()
        for trial in study.trials:
            if trial.state.is_finished() and trial.value is not None:
                params = trial.params.copy()
                params["value"] = trial.value
                params["trial"] = trial.number
                df_results = pd.concat([df_results, pd.DataFrame([params])], ignore_index=True)
        
        if len(df_results) < 2:
            logger.warning("Not enough completed trials for visualization")
            return
        
        # 1. Create a parallel coordinate plot to visualize parameter relationships
        plt.figure(figsize=(12, 8))
        pd.plotting.parallel_coordinates(
            df_results, 'trial', 
            cols=[col for col in df_results.columns if col not in ['trial', 'value']]
        )
        plt.title(f"{model_name} - Parameter Parallel Coordinates")
        plt.tight_layout()
        #plt.savefig(f"{output_dir}/{model_name}_parallel_coordinates_large.png")
        plt.close()
        
        # 2. Create pairplots for key parameters and objective value
        # Select the most important parameters (top 3-4)
        if len(df_results.columns) > 5:
            # Try to get parameter importance from study
            try:
                importance = optuna.importance.get_param_importances(study)
                top_params = list(importance.keys())[:4]  # Get top 4 parameters
            except:
                # If importance calculation fails, just take the first few params
                top_params = [col for col in df_results.columns 
                             if col not in ['trial', 'value']][:4]
        else:
            top_params = [col for col in df_results.columns if col not in ['trial', 'value']]
        
        # Include the value column and create a scatter matrix
        plot_cols = top_params + ['value']
        
        # Using seaborn for better visualization
        sns.set(style="ticks")
        g = sns.pairplot(
            df_results[plot_cols], 
            height=2.5,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
            diag_kws={'fill': True}
        )
        g.fig.suptitle(f"{model_name} - Parameter Relationships", y=1.02, fontsize=16)
        #g.savefig(f"{output_dir}/{model_name}_parameter_relationships_large.png")
        plt.close()
        
        # 3. Create learning curves (trial number vs score)
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['trial'], df_results['value'], 'o-', alpha=0.7)
        plt.axhline(y=study.best_value, color='r', linestyle='--', 
                   label=f'Best value: {study.best_value:.4f}')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title(f'{model_name} - Optimization History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
       # plt.savefig(f"{output_dir}/{model_name}_learning_curve_large.png")
        plt.close()
        
        logger.info(f"Created additional visualizations for {model_name} in {output_dir}")
        
    except Exception as e:
        logger.warning(f"Error creating additional visualizations: {str(e)}")


# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Define color schemes for consistent visualization
COLORS = {
    'rf': '#1f77b4',     # blue
    'lr': '#d62728',     # red
    'svm': '#9467bd',    # purple
    'knn': '#8c564b',    # brown
    
    # Also add keys with capital letters for different access patterns
    'RF': '#1f77b4',     
    'Random Forest': '#1f77b4'
}

def gpu_available():
    """Check if GPU is available for computation."""
    # Check for NVIDIA GPU on Windows
    if platform.system() == 'Windows':
        try:
            # Check using nvidia-smi
            subprocess.check_output('nvidia-smi')
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    # Check for NVIDIA GPU on Linux/macOS
    else:
        try:
            subprocess.check_output(['nvidia-smi'])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

def load_and_preprocess_data(filepath="./processed/classification_large.csv"):
    """Load and preprocess the dataset."""
    logger.info("Loading and preprocessing data...")
    
    data = pd.read_csv(filepath)
    
    # Check for missing values
    missing_values = data.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_values[missing_values > 0]}")
    
    # Fill missing values with appropriate strategies
    # For numeric columns: median
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            
    # Drop rows with any remaining missing values
    original_rows = len(data)
    data = data.dropna()
    dropped_rows = original_rows - len(data)
    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with missing values")
    print(data.head())
    # Only get rows with MCH_pos = 1
    #data = data[data['MCH_pos'] == 1]
    # Group MCH_count into 4 classes: 0 (No MCH), 1 (MCH=1), 2-4 (MCH=2-4), >4 (MCH>4)
    def group_mch_count(mch_count):
        if mch_count == 0:
            return 0
        elif mch_count == 1:
            return 1  # MCH=1
        elif 2 <= mch_count <= 3:
            return 2  # MCH=2-3
        else:
            return 3 # MCH>3

    data['MCH_count'] = data['MCH_count'].apply(group_mch_count)
    # Split features and target
    X = data.drop(columns=['MCH_pos','MCH_count', 'RID', 'SCANDATE'])
    y = data['MCH_count']
    groups = data['RID']
    
    # Check class distribution
    class_dist = y.value_counts(normalize=True)
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Class distribution: {class_dist.to_dict()}")
    
    return X, y, groups

def split_and_scale_data(X, y, groups=None):
    """Split the data into training, validation and test sets and scale the features."""
    logger.info(f"Original dataset shape: {X.shape}")
    
    # First split: separate test set
    if groups is not None:
        X_temp, X_test, y_temp, y_test, groups_temp, groups_test = train_test_split(
            X, y, groups, test_size=0.1, random_state=123, stratify=y)
        
        # Second split: create validation set from remaining data
        X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
            X_temp, y_temp, groups_temp, test_size=0.1, random_state=123, stratify=y_temp)
    else:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=123, stratify=y)
        
        # Second split: create validation set from remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1, random_state=123, stratify=y_temp)
        groups_train = None

   
    # Store feature names before conversion to numpy
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    
    
    # Convert to numpy arrays if they are not already
    X_train_np = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    X_val_np = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
    X_test_np = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_val_scaled = scaler.transform(X_val_np)
    X_test_scaled = scaler.transform(X_test_np)

    X_train_res = X_train_scaled
    y_train_res = y_train
        
    # Save test data for statistical analysis
    

    
    return X_train_res, y_train_res, X_val_scaled, y_val, X_test_scaled, y_test, feature_names, groups_train

def train_random_forest(X_train, y_train, X_val, y_val, groups=None, perform_hyperparameter_search=False):
    """Train and evaluate Random Forest model using Optuna for hyperparameter optimization."""
    logger.info("Training Random Forest model...")
    
    # Create a directory for Optuna study visualizations
    os.makedirs("viz/classification_results/optuna", exist_ok=True)
    
    if perform_hyperparameter_search:
        logger.info("Starting Optuna hyperparameter optimization for Random Forest...")
        
        # Define the objective function for Optuna
        def objective(trial):
            # Hyperparameter search space
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                
                # Class weights to handle imbalance
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
                
                # Fixed parameters
                'random_state': 123,
                'n_jobs': -1  # Use all available cores
            }
            
            # Create the model with the suggested hyperparameters
            
            model = RandomForestClassifier(**param)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_val_pred_proba = model.predict_proba(X_val)
            val_roc_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr', average='weighted')
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            #logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
            
            return val_accuracy
        
        # Create the Optuna study
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            study_name="random_forest_optimization"
        )
        
        # Start hyperparameter optimization
        start_time = time.time()
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        training_time = time.time() - start_time
        
        # Log the optimization results to console and save to file
        results_df = log_optuna_results(study, "random_forest")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best Accuracy: {study.best_value:.4f}")
        

        
        # Instantiate the best model
        best_params = study.best_params
        best_params.update({
            'random_state': 123,
            'n_jobs': -1  # Use all available cores
        })
        
        best_rf = RandomForestClassifier(**best_params)
        best_rf.fit(X_train, y_train)
    else:
        # Train base model with default parameters
        logger.info("Training base Random Forest model with default parameters...")
        rf = RandomForestClassifier(
            random_state=123,
            class_weight='balanced',  # Adjust weights inversely proportional to class frequencies
            n_estimators=500,         # More trees for better performance
            max_depth=10,             # Limit tree depth to prevent overfitting
            min_samples_leaf=5,       # Require more samples in leaf nodes
            min_samples_split=10,     # Require more samples to split
            n_jobs=-1                 # Use all available cores
        )
        
        start_time = time.time()
        best_rf = rf
        best_rf.fit(X_train, y_train)
        training_time = time.time() - start_time
    
    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    y_val_pred_proba = best_rf.predict_proba(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr', average='weighted')
    
    # Cross-validation scores (handle potential SMOTE-induced length mismatch)
    use_group_cv = groups is not None
    if use_group_cv:
        try:
            if len(groups) != len(y_train):
                logger.warning(f"Groups length ({len(groups)}) != y_train length ({len(y_train)}). Falling back to standard 5-fold CV.")
                use_group_cv = False
        except Exception:
            use_group_cv = False
    if use_group_cv:
        cv = GroupKFold(n_splits=5)
        cv_scores = cross_val_score(best_rf, X_train, y_train, groups=groups, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')
    
    # Log results
    logger.info(f"RF Training Time: {training_time:.2f} seconds")
    logger.info(f"RF Validation Accuracy: {val_accuracy:.3f}")
    logger.info(f"RF Validation ROC AUC: {val_roc_auc:.3f}")
    logger.info(f"RF Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")
    
    # Feature importance visualization
    # if hasattr(best_rf, 'feature_importances_') and len(X_train) > 0:
    #     try:
    #         # Get number of features
    #         n_features = best_rf.n_features_in_
            
    #         # Create feature importance plot
    #         plt.figure(figsize=(10, 8))
    #         importances = best_rf.feature_importances_
    #         indices = np.argsort(importances)[::-1]
            
    #         # Plot top 20 features or all features if less than 20
    #         top_n = min(20, n_features)
    #         plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    #         plt.bar(range(top_n), importances[indices[:top_n]], align='center', color=COLORS['RF'])
            
    #         # If feature names are available, use them for x-axis labels
    #         if 'feature_names' in locals() and feature_names is not None and len(feature_names) == n_features:
    #             plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    #         else:
    #             plt.xticks(range(top_n), [f'Feature {i}' for i in indices[:top_n]], rotation=90)
                
    #         plt.tight_layout()
    #         #plt.savefig('viz/classification_results/rf_feature_importance_large.png')
    #         plt.close()
    #         logger.info("Saved Random Forest feature importance plot to viz/classification_results/rf_feature_importance.png")
    #     except Exception as e:
    #         logger.warning(f"Could not create feature importance plot: {e}")
    
    return best_rf


def evaluate_model(model, X_test, y_test, model_name, optimize_threshold=False):
    """Evaluate a model and return metrics with optional threshold optimization."""
    logger.info(f"Evaluating {model_name}...")
    
    # Get probability predictions
    y_pred_proba_full = model.predict_proba(X_test)
    y_pred_proba = y_pred_proba_full  # Keep all class probabilities for multi-class
    
    # Default threshold prediction
    y_pred_default = model.predict(X_test)
    default_accuracy = accuracy_score(y_test, y_pred_default)
    
    # For multi-class classification, we don't optimize thresholds
    # Just use the default prediction
    y_pred = y_pred_default
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    logger.info(f"{model_name} - Test accuracy: {accuracy:.3f}")
    logger.info(f"{model_name} - Test AUC: {auc:.3f}")
    logger.info(f"{model_name} - Test F1: {f1:.3f}")
    logger.info(f"{model_name} - Test precision: {precision:.3f}")
    logger.info(f"{model_name} - Test recall: {recall:.3f}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models (multi-class)."""
    logger.info("Plotting ROC curves for multi-class classification...")
    
    plt.figure(figsize=(12, 8))
    
    for i, (model_name, result) in enumerate(results.items()):
        # Get probability predictions
        y_proba = result['y_pred_proba']
        
        # For multi-class, we need to compute ROC for each class
        n_classes = y_proba.shape[1]
        colors = ['blue', 'red', 'green', 'orange', 'purple'][:n_classes]
        
        for class_idx in range(n_classes):
            # Convert to binary problem: current class vs rest
            y_binary = (y_test == class_idx).astype(int)
            y_proba_binary = y_proba[:, class_idx]
            
            fpr, tpr, _ = roc_curve(y_binary, y_proba_binary)
            roc_auc = auc(fpr, tpr)
            
            # Use the COLORS dictionary with clear fallback colors
            model_color = COLORS.get(model_name, COLORS.get(model_name.lower(), f'C{i}'))
            
            plt.plot(fpr, tpr, lw=2, color=model_color, alpha=0.7,
                     label=f'{model_name} Class {class_idx} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    #plt.savefig("viz/classification_results/roc_curves_large.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results, y_test=None, output_dir="viz/classification_results"):
    """Plot and save confusion matrices for all models.
    Automatically uses the correct labels for the 4-class MCH grouping.
    """
    logger.info("Plotting confusion matrices...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine class indices and labels
    if y_test is not None:
        classes = sorted(np.unique(y_test))
    else:
        # Fallback: infer from predictions of the first model
        any_result = next(iter(results.values()))
        classes = sorted(np.unique(any_result['y_pred']))

    # Map known labels for our grouping
    label_map = {0: 'No MCH', 1: 'MCH=1', 2: 'MCH=2-3', 3: 'MCH>3'}
    class_labels = [label_map.get(int(c), f'Class {int(c)}') for c in classes]

    # Calculate number of rows and columns for subplots
    n_models = len(results)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Convert to 2D array if there's only 1 plot
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (model_name, result) in enumerate(results.items()):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Calculate confusion matrix with fixed class order
        if y_test is not None:
            cm = confusion_matrix(y_test, result['y_pred'], labels=classes)
            accuracy = accuracy_score(y_test, result['y_pred'])
        else:
            logger.warning(f"y_test not provided - creating placeholder confusion matrix for {model_name}")
            cm = np.zeros((len(classes), len(classes)), dtype=int)
            accuracy = float('nan')

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels
        )

        # Set titles and labels
        ax.set_title(f"{model_name}\nAccuracy={accuracy:.3f}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)

    # Hide any unused subplots
    for i in range(n_models, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "multiclass_confusion_matrices.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrices to {save_path}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name, top_n=15):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for sorting
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance and get top n
        top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                align='center', color=COLORS.get(model_name.lower(), 'blue'), alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features - {model_name}')
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        #plt.savefig(f"viz/classification_results/{model_name.lower()}_feature_importance_large.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_features
    else:
        logger.warning(f"{model_name} does not have feature_importances_ attribute.")
        return None

def compare_models(results_dict):
    """Compare model performance metrics."""
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': list(results_dict.keys()),
        'Accuracy': [results['accuracy'] for results in results_dict.values()],
        'F1 Score': [results['f1'] for results in results_dict.values()]
    })
    
    # Display comparison
    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Bar chart
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', 
           color=[COLORS.get(model.lower(), 'blue') for model in comparison_df['Model']], alpha=0.7)
    plt.bar(x + width/2, comparison_df['F1 Score'], width, label='F1 Score', 
           color=[COLORS.get(model.lower(), 'blue') for model in comparison_df['Model']], alpha=0.9)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, comparison_df['Model'])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(comparison_df['Accuracy']):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(comparison_df['F1 Score']):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    #plt.savefig("viz/classification_results/model_comparison_large.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df


def plot_learning_curve_curves(estimator, X, y, groups, title='Learning curve', cv_splits=5):
    """Plot group-aware learning curves to diagnose data vs signal limitation."""
    cv = GroupKFold(n_splits=cv_splits) if groups is not None else 5
    train_sizes = np.linspace(0.1, 1.0, 6)
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        groups=groups,
        train_sizes=train_sizes,
        cv=cv,
        scoring='accuracy',
        shuffle=True,
        random_state=123,
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    # Ensure directory exists
    os.makedirs('viz/classification_results', exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.plot(sizes, train_mean, 'o-', label='Train Accuracy')
    plt.plot(sizes, val_mean, 'o-', label='Validation Accuracy')
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('viz/classification_results/learning_curve_large.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curve to viz/classification_results/learning_curve_large.png")
    return sizes, train_mean, val_mean


def label_shuffle_baseline(estimator, X, y, groups, cv_splits=5):
    """Compute CV accuracy after shuffling labels to check leakage and baseline."""
    cv = GroupKFold(n_splits=cv_splits) if groups is not None else 5
    y_shuffled = sk_shuffle(y, random_state=123)
    scores = cross_val_score(estimator, X, y_shuffled, groups=groups, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std()


def capacity_sweep_random_forest(X_train, y_train, X_val, y_val, groups=None):
    """Sweep RF capacity to detect model limitation vs. signal limitation."""
    configs = [
        {'max_depth': 4,  'min_samples_leaf': 10},
        {'max_depth': 8,  'min_samples_leaf': 5},
        {'max_depth': 12, 'min_samples_leaf': 2},
        {'max_depth': None, 'min_samples_leaf': 1},
    ]
    results = []
    for cfg in configs:
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=123,
            n_jobs=-1,
            **cfg
        )
        rf.fit(X_train, y_train)
        acc_val = accuracy_score(y_val, rf.predict(X_val))
        results.append((cfg, acc_val))
    return results



def feature_selection(X_train, y_train, threshold=0.01):
    """Select important features using a trained Random Forest."""
    logger.info("Performing feature selection...")
    
    # Train a Random Forest for feature selection
    feat_selector = RandomForestClassifier(
        n_estimators=200, 
        random_state=123,
        class_weight='balanced'
    )
    feat_selector.fit(X_train, y_train)
    
    # Get feature importances and names
    importances = feat_selector.feature_importances_
    
    # Create a DataFrame of features and importances
    if hasattr(X_train, 'columns'):
        features = X_train.columns
    else:
        features = np.array([f'feature_{i}' for i in range(X_train.shape[1])])
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Select features above the threshold
    selected_features = feature_importance[feature_importance['importance'] > threshold]
    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    logger.info(f"Top 10 features: {selected_features.head(10)['feature'].tolist()}")
    
    return selected_features['feature'].tolist()

def main(hyperparameter_search=True):
    """Main pipeline function.
    
    Args:
        hyperparameter_search (bool): Whether to perform hyperparameter search. 
                                     If False, train base models with default parameters.
    """
    # Start timing
    start_time = time.time()
    
    # Create visualization directory if it doesn't exist
    os.makedirs("viz", exist_ok=True)
    
    # Log hyperparameter search setting
    if hyperparameter_search:
        logger.info("Hyperparameter search is enabled.")
    else:
        logger.info("Hyperparameter search is disabled. Training base models with default parameters.")
    
    # Load and preprocess data
    X, y, groups = load_and_preprocess_data()
    
    # Split and scale data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, groups_train = split_and_scale_data(X, y, groups)
    logger.info("Dataset shapes summary:")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Feature count: {len(feature_names)}")

 
    # Diagnostics: learning curve, label-shuffle, capacity sweep
    estimator_for_curve = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        max_depth=10,
        min_samples_leaf=5,
        min_samples_split=10,
        random_state=123,
        n_jobs=-1,
    )

    # Train Random Forest model
    best_rf = train_random_forest(X_train, y_train, X_val, y_val, groups_train, perform_hyperparameter_search=hyperparameter_search)
    
    # Standard evaluation
    results = {}
    results['RF'] = evaluate_model(best_rf, X_test, y_test, "Random Forest")
    
    # Visualizations
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)
    
    # Feature importance
    rf_top_features = plot_feature_importance(best_rf, feature_names, "Random Forest")
    
    # Compare models
    comparison = compare_models(results)
    
    # Total runtime
    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f} seconds")
    


    
    
    # Return results
    return {
        'models': {
            'rf': best_rf
        },
        'results': results,
        'feature_importance': {
            'rf': rf_top_features
        },
        'comparison': comparison,
    }

if __name__ == "__main__":
    main()