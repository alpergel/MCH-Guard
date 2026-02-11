"""
Create Publication-Ready Multi-Panel Figure for Classification Models

This script generates a comprehensive figure combining:
1. ROC curves for all model sizes
2. Feature importance comparison across models
3. Model performance metrics
4. Feature descriptions

Author: MCH-Guard Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
import joblib
import os
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Set publication-quality style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Define colors for models
MODEL_COLORS = {
    'M3': '#d62728',   # Red
    'M2': '#2ca02c',  # Green  
    'M1': '#1f77b4'    # Blue
}

# Feature name mapping to readable labels
FEATURE_LABELS = {
    'PTAGE': 'Age',
    'PTGENDER': 'Sex',
    'PTEDUCAT': 'Education',
    'RACE_ETHNICITY': 'Race/Ethnicity',
    'e4_GENOTYPE': 'APOE ε4 Alleles',
    'ptau_ab_ratio_csf': 'CSF p-tau/Aβ42',
    'PLASMA_NFL': 'Plasma NfL',
    'embedding_scalar': 'Phenotypic Embedding',
    'NORM_WMH': 'White Matter Hyperintensities',
    'NORM_GRAY': 'Gray Matter Volume',
    
    # Cognitive/Clinical scores
    'CDRSB': 'CDR Sum of Boxes',
    'PHC_LAN': 'Language Score',
    'PHC_MEM': 'Memory Score',
    'PHC_EXF': 'Executive Function',
    'PHC_VSP': 'Visuospatial Score',
    'PHC_Smoker': 'Smoking Status',
    'PHC_BMI': 'BMI',
    'CDGLOBAL': 'CDR Global Score',
    'MMSE': 'MMSE Score',
    'FAQ': 'Functional Assessment',
    
    # Medical History
    'PSYCH': 'Psychiatric History',
    'NEURL': 'Neurological History',
    'HEAD': 'Head Injury',
    'CARD': 'Cardiovascular Disease',
    'RESP': 'Respiratory Disease',
    'HEPAT': 'Hepatic Disease',
    'DERM': 'Dermatologic Condition',
    'MUSCL': 'Musculoskeletal Disorder',
    'ENDO': 'Endocrine Disorder',
    'GAST': 'Gastrointestinal Disease',
    'HEMA': 'Hematologic Disorder',
    'RENA': 'Renal Disease',
    'ALLE': 'Allergies',
    'ALCH': 'Alcohol Use',
    
    # Medications (shortened for space)
    'MED_Anti_Thrombotic': 'Antithrombotic',
    'MED_AD_and_Dementia': 'AD/Dementia Med',
    'MED_Lipid_Lowering': 'Lipid-Lowering',
    'MED_Blood_Pressure': 'Antihypertensive',
    'MED_Thyroid': 'Thyroid Med',
    'MED_Vitamins/Minerals': 'Vitamins/Minerals',
    'MED_NSAIDs': 'NSAIDs',
    'MED_Herbal_Supplements': 'Herbal Supplements',
    'MED_Analgesics': 'Analgesics',
    'MED_Antidepressants': 'Antidepressants',
    'MED_Diabetes_Oral': 'Oral Antidiabetic',
    'MED_Other_GI': 'GI Medication',
    'MED_General_Urological': 'Urological Med',
    'MED_Antibiotics': 'Antibiotics',
    'MED_Osteoporosis': 'Osteoporosis Med',
    'MED_Inhalers': 'Inhalers',
    'MED_Sleep_Aids': 'Sleep Aids',
    'MED_Hormone_Replacement': 'Hormone Therapy',
    'MED_GERD/PPI': 'PPI/GERD Med',
    'MED_Anticonvulsants': 'Anticonvulsants',
    'MED_Antihistamines': 'Antihistamines',
    'MED_H2_Blockers': 'H2 Blockers',
    'MED_Nasal_Sprays': 'Nasal Sprays',
    'MED_Diabetes_Injectable': 'Injectable Antidiabetic',
    'MED_Corticosteroids': 'Corticosteroids',
    'MED_Benzodiazepines': 'Benzodiazepines',
    'MED_Eye_Supplements': 'Eye Supplements',
    'MED_Other_Respiratory': 'Respiratory Med',
    'MED_Antiviral': 'Antiviral',
    'MED_Anti_Arrythmia': 'Antiarrhythmic',
}

def get_readable_name(feature):
    """Convert technical feature name to readable label."""
    return FEATURE_LABELS.get(feature, feature)

def load_model_data(size):
    """Load model, test data, and compute predictions."""
    import warnings
    
    # Load model
    model_path = f"models/cls_{size.lower()}_model.joblib"
    model = joblib.load(model_path)
    
    # Load test data
    test_path = f"processed/CLS_{size.lower()}_test.csv"
    test_data = pd.read_csv(test_path)
    
    X_test = test_data.drop(columns=['MCH_pos'])
    y_test = test_data['MCH_pos']
    
    # Store feature names before converting to numpy
    feature_names = X_test.columns.tolist()
    
    # Convert to numpy array to avoid sklearn warning about feature names
    X_test_array = X_test.values
    
    # Get predictions using numpy array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_proba = model.predict_proba(X_test_array)[:, 1]
        y_pred = model.predict(X_test_array)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Calculate confidence intervals from trees
    if hasattr(model, 'estimators_'):
        tree_importances = np.array([tree.feature_importances_ for tree in model.estimators_])
        importances_std = np.std(tree_importances, axis=0)
        importances_ci_lower = importances - 1.96 * importances_std
        importances_ci_upper = importances + 1.96 * importances_std
    else:
        importances_std = np.zeros_like(importances)
        importances_ci_lower = importances
        importances_ci_upper = importances
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'ci_lower': importances_ci_lower,
        'ci_upper': importances_ci_upper
    })
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_importance': feature_importance,
        'X_test': X_test  # Keep DataFrame for feature correlation calculations
    }

def plot_roc_panel(ax, data_dict):
    """Plot ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='Chance')
    
    for size, color in MODEL_COLORS.items():
        data = data_dict[size]
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, linewidth=2.5, 
                label=f'{size} Model (AUC={roc_auc:.3f})', alpha=0.8)
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves', fontsize=13, fontweight='bold', loc='left')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, 
             edgecolor='black', fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')

def plot_feature_comparison_panel(ax, data_dict, top_n=15):
    """Plot feature importance comparison across models."""
    # Get union of top features from all models
    all_top_features = set()
    for size in MODEL_COLORS.keys():
        top_feats = data_dict[size]['feature_importance'].nlargest(top_n, 'importance')['feature']
        all_top_features.update(top_feats)
    
    # Focus on M3 model's top features for main figure
    m3_top = data_dict['M3']['feature_importance'].nlargest(top_n, 'importance')['feature'].tolist()
    
    # Prepare data for plotting
    y_positions = np.arange(len(m3_top))
    bar_height = 0.25
    
    for idx, size in enumerate(['M1', 'M2', 'M3']):  # Reversed order for better stacking
        data = data_dict[size]['feature_importance']
        
        importances = []
        ci_lowers = []
        ci_uppers = []
        
        for feat in m3_top:
            if feat in data['feature'].values:
                row = data[data['feature'] == feat].iloc[0]
                importances.append(row['importance'])
                ci_lowers.append(row['ci_lower'])
                ci_uppers.append(row['ci_upper'])
            else:
                importances.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
        
        importances = np.array(importances)
        ci_lowers = np.array(ci_lowers)
        ci_uppers = np.array(ci_uppers)
        
        # Calculate error bars
        lower_err = importances - ci_lowers
        upper_err = ci_uppers - importances
        
        # Offset positions for each model
        offset = (idx - 1) * bar_height
        
        ax.barh(y_positions + offset, importances, bar_height, 
                color=MODEL_COLORS[size], alpha=0.7, 
                label=f'{size} Model', edgecolor='black', linewidth=0.8)
        
        # Add error bars for M3 model only to avoid clutter
        if size == 'M3':
            ax.errorbar(importances, y_positions + offset, 
                       xerr=[lower_err, upper_err],
                       fmt='none', ecolor='black', capsize=2, 
                       linewidth=1, alpha=0.5)
    
    # Set labels with readable names
    readable_labels = [get_readable_name(f) for f in m3_top]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(readable_labels, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title('B) Top 15 Features Comparison', fontsize=13, fontweight='bold', loc='left')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
             edgecolor='black', fontsize=10)
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.set_xlim(0, None)
    ax.invert_yaxis()  # Most important at top

def plot_performance_metrics_panel(ax, data_dict):
    """Plot performance metrics comparison with bootstrap confidence intervals."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.utils import resample
    
    def bootstrap_metric(y_true, y_pred, y_pred_proba, metric_name, n_bootstrap=1000):
        """Calculate bootstrap confidence interval for a metric."""
        scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = resample(range(n_samples), n_samples=n_samples)
            y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_pred_boot = y_pred[indices]
            y_pred_proba_boot = y_pred_proba[indices]
            
            # Calculate metric
            try:
                if metric_name == 'Accuracy':
                    score = accuracy_score(y_true_boot, y_pred_boot)
                elif metric_name == 'Precision':
                    score = precision_score(y_true_boot, y_pred_boot, zero_division=0)
                elif metric_name == 'Recall':
                    score = recall_score(y_true_boot, y_pred_boot, zero_division=0)
                elif metric_name == 'F1-Score':
                    score = f1_score(y_true_boot, y_pred_boot, zero_division=0)
                elif metric_name == 'ROC-AUC':
                    score = roc_auc_score(y_true_boot, y_pred_proba_boot)
                scores.append(score)
            except:
                continue
        
        # Calculate 95% CI
        lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)
        mean = np.mean(scores)
        
        return mean, lower, upper
    
    print("  Computing bootstrap confidence intervals...")
    metrics_data = []
    for size in ['M3', 'M2', 'M1']:
        data = data_dict[size]
        row = {'Model': size}
        
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            mean, lower, upper = bootstrap_metric(
                data['y_test'], data['y_pred'], data['y_pred_proba'], metric, n_bootstrap=500
            )
            row[metric] = mean
            row[f'{metric}_lower'] = lower
            row[f'{metric}_upper'] = upper
        
        metrics_data.append(row)
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(metrics)) * 2.2  # Increase spacing between metric groups
    width = 0.60  # Wider bars for better visibility
    
    annotations = []  # (x_pos, y_top, value)
    for idx, size in enumerate(['M3', 'M2', 'M1']):
        row = df_metrics[df_metrics['Model'] == size].iloc[0]
        values = [row[m] for m in metrics]
        errors_lower = [row[m] - row[f'{m}_lower'] for m in metrics]
        errors_upper = [row[f'{m}_upper'] - row[m] for m in metrics]
        
        offset = (idx - 1) * width
        ax.bar(x + offset, values, width, label=f'{size} Model',
               color=MODEL_COLORS[size], alpha=0.7, edgecolor='black', linewidth=0.8,
               yerr=[errors_lower, errors_upper], capsize=3, 
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.6})
        # Store annotation positions for each metric bar
        for i, val in enumerate(values):
            x_pos = x[i] + offset
            y_top = val + errors_upper[i]
            annotations.append((x_pos, y_top, val))
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics (95% CI)', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=11)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
             edgecolor='black', fontsize=10)
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add numeric value annotations above CI caps
    y_margin = 0.02  # Fixed margin for spacing above error bars
    
    # Calculate max annotation height to set appropriate y-limit
    max_annotation_y = max(y_top + y_margin for _, y_top, _ in annotations)
    
    # Set y-limit with extra headroom for annotations (10% extra space above max annotation)
    ax.set_ylim([0, max_annotation_y * 1.10])
    
    for x_pos, y_top, val in annotations:
        ax.text(x_pos, y_top + y_margin, f"{val:.3f}", ha='center', va='bottom',
                fontsize=9, fontweight='bold')

def plot_feature_descriptions_panel(ax, data_dict):
    """Plot feature category descriptions."""
    ax.axis('off')
    
    # Create text description
    description = """
Feature Categories:

Demographics & Genetics:
• Age, Sex, Education, Race/Ethnicity, APOE ε4 genotype

Biomarkers:
• CSF p-tau/Aβ42 ratio, Plasma neurofilament light (NfL)
• Cognitive embedding score

Neuroimaging:
• White matter hyperintensities (normalized)
• Gray matter volume (normalized)

Medical History:
• Psychiatric, neurological, cardiovascular, and other
  system-based medical conditions

Medications:
• 28 medication categories including antithrombotics,
  AD/dementia medications, cardiovascular drugs, etc.

Note: Feature importance values represent the relative 
contribution of each variable to MCH prediction. Error bars 
indicate 95% confidence intervals calculated from individual 
decision trees.
    """
    
    ax.text(0.05, 0.95, description, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='Arial',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_title('D) Feature Descriptions', fontsize=13, fontweight='bold', loc='left')

def plot_individual_feature_importance(ax, data, model_name, top_n=15):
    """Plot feature importance as a forest plot styled like SRV.
    Positive values indicate increased MCH risk; negative indicate decreased risk.
    """
    # Get top features
    top_features = data['feature_importance'].nlargest(top_n, 'importance')

    # Determine feature directions (correlation with outcome)
    X_test = data['X_test']
    y_test = data['y_test']

    feature_directions = {}
    for feat in top_features['feature']:
        if feat in X_test.columns:
            feat_values = X_test[feat].values
            try:
                if np.std(feat_values) < 1e-10 or np.std(y_test) < 1e-10:
                    feature_directions[feat] = 'unknown'
                elif np.any(np.isnan(feat_values)) or np.any(np.isnan(y_test)):
                    feature_directions[feat] = 'unknown'
                else:
                    corr = np.corrcoef(feat_values, y_test)[0, 1]
                    if np.isnan(corr):
                        feature_directions[feat] = 'unknown'
                    else:
                        feature_directions[feat] = 'positive' if corr > 0 else 'negative'
            except:
                feature_directions[feat] = 'unknown'
        else:
            feature_directions[feat] = 'unknown'

    # Add direction to dataframe and sort (keep original relative style: ascending)
    top_features = top_features.copy()
    top_features['direction'] = top_features['feature'].map(feature_directions)
    top_features = top_features.sort_values('importance', ascending=True)

    y_positions = np.arange(len(top_features))

    # Labels (match SRV style: indicate Yes/↑value mapping) 
    med_history_like = {'PSYCH','NEURL','HEAD','CARD','RESP','HEPAT','DERM','MUSCL','ENDO','GAST','HEMA','RENA','ALLE','ALCH','PHC_Smoker'}
    readable_labels = []
    for _, row in top_features.iterrows():
        feat = row['feature']
        direction = row['direction']
        base_name = get_readable_name(feat)
        if feat.startswith('MED_') or feat in med_history_like:
            # Binary: Yes/No
            if direction == 'positive':
                readable_labels.append(f"{base_name} [Yes → ↑risk]")
            elif direction == 'negative':
                readable_labels.append(f"{base_name} [Yes → ↓risk]")
            else:
                readable_labels.append(base_name)
        elif feat == 'e4_GENOTYPE':
            if direction == 'positive':
                readable_labels.append(f"{base_name} [+1 copy → ↑risk]")
            elif direction == 'negative':
                readable_labels.append(f"{base_name} [+1 copy → ↓risk]")
            else:
                readable_labels.append(base_name)
        elif feat == 'PTGENDER':
            if direction == 'positive':
                readable_labels.append(f"{base_name} [Female → ↑risk]")
            elif direction == 'negative':
                readable_labels.append(f"{base_name} [Male → ↑risk]")
            else:
                readable_labels.append(base_name)
        else:
            # Continuous
            if direction == 'positive':
                readable_labels.append(f"{base_name} [↑value → ↑risk]")
            elif direction == 'negative':
                readable_labels.append(f"{base_name} [↑value → ↓risk]")
            else:
                readable_labels.append(base_name)

    # Compute signed point estimates and CIs
    signed_points = []
    signed_low = []
    signed_high = []
    for _, row in top_features.iterrows():
        imp = row['importance']
        ci_l = row['ci_lower']
        ci_u = row['ci_upper']
        direction = row['direction']
        if direction == 'positive':
            point = imp
            low = ci_l
            high = ci_u
        elif direction == 'negative':
            point = -imp
            low = -ci_u
            high = -ci_l
        else:
            point = imp
            low = ci_l
            high = ci_u
        signed_points.append(point)
        signed_low.append(low)
        signed_high.append(high)

    signed_points = np.array(signed_points)
    signed_low = np.array(signed_low)
    signed_high = np.array(signed_high)

    # Prepare error extents for errorbar API
    err_lower = signed_points - signed_low
    err_upper = signed_high - signed_points

    # Draw forest plot using SRV-like errorbar markers
    ax.errorbar(signed_points, y_positions,
                xerr=[err_lower, err_upper],
                fmt='o', markersize=7, capsize=4, capthick=1.5,
                color='#d93a3a', ecolor='#d93a3a',
                markeredgecolor='darkred', markeredgewidth=1.2,
                linewidth=1.8, alpha=0.8)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    # Labels and formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(readable_labels, fontsize=8.5)
    ax.set_xlabel('Signed Importance (95% CI)', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name} Model', fontsize=11, fontweight='bold', loc='left')
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)

    # Symmetric x-limits around 0 for visual balance
    max_abs = 0.0
    if signed_low.size > 0 and signed_high.size > 0:
        max_abs = max(np.max(np.abs(signed_low)), np.max(np.abs(signed_high)))
    ax.set_xlim(-1.05 * max_abs, 1.05 * max_abs if max_abs > 0 else 1)

    # Legend for direction (SRV style)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='darkred', label='Positive effect (↑risk)', linewidth=1.5),
        Patch(facecolor='white', edgecolor='darkgreen', label='Negative effect (↓risk)', linewidth=1.5)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
              framealpha=0.9, edgecolor='black', title='Interpretation:', title_fontsize=7)

    # Add numeric value labels (color-coded)
    # Compute max_abs for offsets
    max_abs = 0.0
    if signed_low.size > 0 and signed_high.size > 0:
        max_abs = max(np.max(np.abs(signed_low)), np.max(np.abs(signed_high)))
    offset_unit = 0.03 * (max_abs if max_abs > 0 else 1.0)
    for i, val in enumerate(signed_points):
        if np.isnan(val):
            continue
        if val >= 0:
            ha = 'left'
            dx = offset_unit
            color = 'darkred'
        else:
            ha = 'right'
            dx = -offset_unit
            color = 'darkgreen'
        ax.text(val + dx, i, f'{val:.3f}', va='center', ha=ha, fontsize=7.5,
                color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.7, linewidth=1))

    # Sample info
    n_samples = len(data['y_test'])
    n_pos = sum(data['y_test'])
    ax.text(0.98, 0.92, f'n={n_samples}\nMCH+={n_pos}',
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

def create_publication_figure():
    """Create the main publication figure."""
    print("Loading model data...")
    data_dict = {}
    for size in ['M3', 'M2', 'M1']:
        print(f"  Loading {size} model...")
        data_dict[size] = load_model_data(size)
    
    print("\nCreating publication figure...")
    
    # Create figure with 2 columns: left for ROC/metrics, right for feature importance
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35,
                          left=0.06, right=0.97, top=0.96, bottom=0.05,
                          height_ratios=[1.2, 1, 1])
    
    # Left column
    # A) ROC curves (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_roc_panel(ax1, data_dict)
    
    # B) Performance metrics (middle and bottom left combined)
    ax2 = fig.add_subplot(gs[1:, 0])
    plot_performance_metrics_panel(ax2, data_dict)
    
    # Right column - Feature importance for each model
    # C) M3 model feature importance (top right)
    ax3 = fig.add_subplot(gs[0, 1])
    plot_individual_feature_importance(ax3, data_dict['M3'], 'M3', top_n=15)
    
    # D) M2 model feature importance (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_individual_feature_importance(ax4, data_dict['M2'], 'M2', top_n=15)
    
    # E) M1 model feature importance (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    plot_individual_feature_importance(ax5, data_dict['M1'], 'M1', top_n=15)
    
    # Add panel labels
    panels = [(ax1, 'A'), (ax2, 'B'), (ax3, 'C'), (ax4, 'D'), (ax5, 'E')]
    for ax, label in panels:
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')
    
    # Save figure
    output_path = "viz/classification_results/publication_figure_cls.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    # Also save as high-res PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
    
    print("\n[SUCCESS] Publication figure created successfully!")

if __name__ == "__main__":
    create_publication_figure()

