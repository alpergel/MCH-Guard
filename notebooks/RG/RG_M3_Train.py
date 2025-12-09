
"""
RG_M3_Train.py

This script trains an ExtraTrees Regressor model on the m3 worsening dataset
to predict the duration (in months) based on input features. It includes
automated hyperparameter tuning using RandomizedSearchCV.

Author: ARIA-Guard Team
Date: April 2025
"""

# Standard library imports
import os
import time

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             explained_variance_score)
import joblib
from scipy.stats import randint, uniform

COLORS = {
    'Extra Trees Regressor': '#d62728',     # red
}
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset.
        
    Returns:
        tuple: X and y data for modeling.
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    
    # Convert Duration from days to years
    data['Duration'] = round(data['Duration'] / 365)
    #data['Duration'] = data['Duration'] / 365
    # Drop unnecessary columns
    data = data.drop(columns=['RID', 'SCANDATE', 'MCH_pos', 'MCH_count', 'SWITCH_STATUS'])
    
    # Prepare the data
    X = data.drop(columns=['Duration'])
    y = data['Duration']
    

    print(f"Data loaded successfully. Shape: {X.shape}")
    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    ev_score = explained_variance_score(y_test, y_pred)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}") 
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Explained Variance Score: {ev_score:.4f}")
    
    return {
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'ev_score': ev_score
    }





def plot_feature_importance(model, feature_names, model_name, top_n=15, X_train=None, y_train=None):
    """Plot feature importance for tree-based models with confidence intervals and directional effects."""
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Calculate confidence intervals from individual trees
        if hasattr(model, 'estimators_'):
            tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
            importances_std = np.std(tree_importances, axis=0)
            importances_ci_lower = importances - 1.96 * importances_std
            importances_ci_upper = importances + 1.96 * importances_std
        else:
            # If we can't get tree-level importances, use bootstrap estimate
            importances_std = np.zeros_like(importances)
            importances_ci_lower = importances
            importances_ci_upper = importances
        
        # Create DataFrame for sorting
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'std': importances_std,
            'ci_lower': importances_ci_lower,
            'ci_upper': importances_ci_upper
        })
        
        # Determine direction of effect for each feature (correlation with outcome)
        feature_directions = {}
        if X_train is not None and y_train is not None:
            print("Computing feature directions...")
            for i, feat in enumerate(feature_names):
                # Get feature values
                if isinstance(X_train, pd.DataFrame):
                    feat_values = X_train[feat].values
                else:
                    feat_values = X_train[:, i]
                
                # Calculate correlation with outcome
                try:
                    # Check for zero variance to avoid division by zero
                    if np.std(feat_values) < 1e-10 or np.std(y_train) < 1e-10:
                        feature_directions[feat] = 'unknown'
                    else:
                        # Check for NaN values
                        if np.any(np.isnan(feat_values)) or np.any(np.isnan(y_train)):
                            feature_directions[feat] = 'unknown'
                        else:
                            corr = np.corrcoef(feat_values, y_train)[0, 1]
                            # Check if correlation is NaN (can happen with constant features)
                            if np.isnan(corr):
                                feature_directions[feat] = 'unknown'
                            else:
                                feature_directions[feat] = 'positive' if corr > 0 else 'negative'
                except Exception as e:
                    print(f"Could not compute correlation for {feat}: {e}")
                    feature_directions[feat] = 'unknown'
        else:
            # Default to unknown if training data not provided
            for feat in feature_names:
                feature_directions[feat] = 'unknown'
        
        feature_importance['direction'] = feature_importance['feature'].map(feature_directions)
        
        # Sort by importance and get top n
        top_features = feature_importance.sort_values('importance', ascending=True).tail(top_n)
        
        # Create the plot with error bars
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.5)))
        
        y_pos = np.arange(len(top_features))
        
        # Calculate error bar sizes
        lower_err = top_features['importance'] - top_features['ci_lower']
        upper_err = top_features['ci_upper'] - top_features['importance']
        
        # Color code by direction (for regression: positive = longer duration, negative = shorter duration)
        colors = []
        for direction in top_features['direction']:
            if direction == 'positive':
                colors.append('#d93a3a')  # Red for positive correlation (longer duration)
            elif direction == 'negative':
                colors.append('#2ca02c')  # Green for negative correlation (shorter duration)
            else:
                colors.append('#1f77b4')  # Blue for unknown
        
        # Plot horizontal bars with error bars
        ax.barh(y_pos, top_features['importance'], 
                xerr=[lower_err.values, upper_err.values],
                align='center', alpha=0.7, 
                color=colors, capsize=5, ecolor='black', linewidth=1.5)
        
        # Create interpretable y-axis labels with direction indicators
        new_labels = []
        for idx, row in top_features.iterrows():
            feat = row['feature']
            importance = row['importance']
            direction = row['direction']
            
            # Binary variables (0/1)
            if feat.startswith('MED_') or feat in ['PSYCH', 'NEURL', 'HEAD', 'CARD', 'RESP', 
                                                    'HEPAT', 'DERM', 'MUSCL', 'ENDO', 'GAST', 
                                                    'HEMA', 'RENA', 'ALLE', 'ALCH']:
                if direction == 'positive':
                    new_labels.append(f"{feat} [Yes → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [Yes → ↓duration]")
                else:
                    new_labels.append(f"{feat}")
            
            # e4_GENOTYPE: ordinal (0, 1, 2 copies)
            elif feat == 'e4_GENOTYPE':
                if direction == 'positive':
                    new_labels.append(f"{feat} [+1 copy → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [+1 copy → ↓duration]")
                else:
                    new_labels.append(f"{feat}")
            
            # PTGENDER: categorical
            elif feat == 'PTGENDER':
                if direction == 'positive':
                    new_labels.append(f"{feat} [Female → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [Male → ↑duration]")
                else:
                    new_labels.append(f"{feat}")
            
            # PTEDUCAT: binary (0, 1)
            elif feat == 'PTEDUCAT':
                if direction == 'positive':
                    new_labels.append(f"{feat} [Higher edu → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [Higher edu → ↓duration]")
                else:
                    new_labels.append(f"{feat}")
            
            # RACE_ETHNICITY: categorical
            elif feat == 'RACE_ETHNICITY':
                if direction == 'positive':
                    new_labels.append(f"{feat} [Higher value → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [Lower value → ↑duration]")
                else:
                    new_labels.append(f"{feat}")
            
            # Continuous variables (age, biomarkers, imaging)
            else:
                if direction == 'positive':
                    new_labels.append(f"{feat} [↑value → ↑duration]")
                elif direction == 'negative':
                    new_labels.append(f"{feat} [↑value → ↓duration]")
                else:
                    new_labels.append(f"{feat}")
        
        # Add importance values as text annotations
        for i, (idx, row) in enumerate(top_features.iterrows()):
            importance = row['importance']
            direction = row['direction']
            
            # Color code the text
            if direction == 'positive':
                text_color = 'darkred'
            elif direction == 'negative':
                text_color = 'darkgreen'
            else:
                text_color = 'darkblue'
            
            ax.text(importance + 0.005, i, f'{importance:.4f}', 
                   va='center', ha='left', fontsize=9, color=text_color, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=text_color, alpha=0.7))
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(new_labels, fontsize=9)
        ax.set_xlabel('Feature Importance (with 95% CI)', fontsize=12, weight='bold')
        ax.set_title(f'Top {top_n} Features with Effect Direction - {model_name} (M3 Dataset)', 
                    fontsize=14, pad=20, weight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d93a3a', edgecolor='darkred', 
                  label='↑ feature value → ↑ duration', linewidth=2, alpha=0.7),
            Patch(facecolor='#2ca02c', edgecolor='darkgreen', 
                  label='↑ feature value → ↓ duration', linewidth=2, alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
                 framealpha=0.9, title='Interpretation Guide:', title_fontsize=9)
        
        # Set x-axis limits with padding
        max_val = top_features['ci_upper'].max()
        ax.set_xlim(0, max_val * 1.15)
        
        plt.tight_layout()
        plt.savefig(f"viz/regression_results/rg_feature_importance_m3.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved enhanced feature importance plot with confidence intervals to viz/regression_results/rg_feature_importance_m3.png")
        
        return top_features
    else:
        print(f"{model_name} does not have feature_importances_ attribute.")
        return None


def plot_true_vs_pred(y_true, y_pred, model_name="Model", save_path="viz/regression_results/rg_true_vs_pred_m3.png"):
    """Plot predicted vs true duration values and save the figure.

    Args:
        y_true (array-like): Ground truth durations.
        y_pred (array-like): Predicted durations by the model.
        model_name (str): Name of the model for labeling.
        save_path (str): File path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color=COLORS.get(model_name.lower(), "#1f77b4"), alpha=0.6)

    # Identity line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)

    plt.xlabel(f"True Duration (365 days)")
    plt.ylabel(f"Predicted Duration (365 days)")
    plt.title(f"True vs Predicted Duration - {model_name}")
    plt.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """
    Main function to run the entire training and evaluation pipeline.
    """
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # File path
    file_path = "./processed/worsening_m3.csv"
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    
    # Save test data for statistical analysis
    test_data = pd.DataFrame(X_test, columns=X.columns)
    test_data['Duration'] = y_test
    test_csv_path = "./processed/RG_m3_test.csv"
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)
    test_data.to_csv(test_csv_path, index=False)
    print(f"Test data saved to {test_csv_path}")
    
    print("\n=== Baseline Model ===")
    # Create and train the baseline model
    baseline_model = ExtraTreesRegressor(n_estimators=500, random_state=123)
    baseline_model.fit(X_train, y_train)
    # Evaluate the baseline model
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)

    # Plot predicted vs true durations
    y_pred = baseline_model.predict(X_test)
    plot_true_vs_pred(y_test, y_pred, "Extra Trees Regressor")
    
    # Get feature names from the original dataset
    feature_names = X.columns.tolist()

    rf_top_features = plot_feature_importance(baseline_model, feature_names, "Extra Trees Regressor", 
                                              top_n=15, X_train=X_train, y_train=y_train)
    
    # Save the tuned model
    model_path = os.path.join("models", "rg_m3_model.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(baseline_model, model_path)
    print(f"\nTuned model saved to {model_path}")



if __name__ == "__main__":
    main()

