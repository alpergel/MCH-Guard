"""
Create Publication-Ready Multi-Panel Figure for Survival Models

This script generates a comprehensive figure combining:
1. Forest plots showing hazard ratios for all model sizes
2. Kaplan-Meier survival curves stratified by risk groups
3. Model performance metrics (Concordance Index, Partial AIC)

Author: MCH-Guard Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os
import warnings
from lifelines import KaplanMeierFitter

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Risk group colors
RISK_COLORS = {
    'Low Risk': '#2ca02c',     # Green
    'M2 Risk': '#ff7f0e',  # Orange
    'High Risk': '#d62728'     # Red
}

# Feature name mapping
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
    'CDRSB': 'CDR Sum of Boxes',
    'PHC_LAN': 'Language Score',
    'PHC_MEM': 'Memory Score',
    'PHC_EXF': 'Executive Function',
    'PHC_VSP': 'Visuospatial Score',
    'PHC_Smoker': 'Smoking Status',
    'PHC_BMI': 'BMI',
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
}

def get_readable_name(feature):
    """Convert technical feature name to readable label."""
    # Remove MED_ prefix if present
    if feature.startswith('MED_'):
        feature_clean = feature.replace('MED_', '').replace('_', ' ').title()
        return FEATURE_LABELS.get(feature, feature_clean)
    return FEATURE_LABELS.get(feature, feature)

def load_model_data(size):
    """Load Cox model and held-out test data."""
    # Load model
    model_path = f"models/srv_{size.lower()}_coxph.joblib"
    model = joblib.load(model_path)
    
    # Load held-out test data
    test_data_path = f"processed/SRV_{size.lower()}_test.csv"
    
    # Check if test data exists, otherwise fall back to full worsening data
    if os.path.exists(test_data_path):
        # Load test data (has RID as index)
        test_df = pd.read_csv(test_data_path, index_col=0)
        print(f"    Using held-out test data: {len(test_df)} samples")
    else:
        print(f"    Warning: Test data not found at {test_data_path}")
        print(f"    Falling back to full worsening data")
        # Load full worsening data
        data_path = f"processed/worsening_{size.lower()}.csv"
        df = pd.read_csv(data_path, parse_dates=['SCANDATE'])
        
        # Create test set similar to training script
        DATECOL = "SCANDATE"
        IDCOL = "RID"
        EVENTCOL = "MCH_pos"
        
        first = df.sort_values([IDCOL, DATECOL]).groupby(IDCOL).first()
        baseline_date = df.groupby(IDCOL)[DATECOL].min()
        event_date = df[df[EVENTCOL]==1].groupby(IDCOL)[DATECOL].min()
        last_date = df.groupby(IDCOL)[DATECOL].max()
        
        test_df = pd.DataFrame({
            "baseline": baseline_date,
            "event_date": event_date,
            "last": last_date
        })
        test_df["event"] = test_df["event_date"].notna().astype(int)
        test_df["time"] = ((test_df["event_date"].fillna(test_df["last"])) - test_df["baseline"]).dt.days/365.25
        
        # Get covariates from model
        covariate_cols = list(model.summary.index)
        available_cols = [col for col in covariate_cols if col in first.columns]
        test_df = test_df.join(first[available_cols])
        test_df = test_df.dropna()
    
    # Get metrics
    concordance = model.concordance_index_
    partial_aic = model.AIC_partial_
    
    return {
        'model': model,
        'test_data': test_df,  # Renamed from 'data' to be clear it's test data
        'concordance': concordance,
        'partial_aic': partial_aic,
        'summary': model.summary
    }

def plot_forest_plot(ax, data, model_name, top_n=12):
    """Plot forest plot (hazard ratios with CIs) matching SRV_Train style."""
    summary = data['summary']
    
    # Get top features by absolute coefficient value
    summary['abs_coef'] = summary['coef'].abs()
    top_features = summary.nlargest(top_n, 'abs_coef')
    
    # Sort by coefficient value for plotting
    top_features = top_features.sort_values('coef')
    
    # Get coefficients and confidence intervals
    coefs = top_features['coef']
    ci_lower = top_features['coef lower 95%']
    ci_upper = top_features['coef upper 95%']
    
    # Calculate error bar sizes
    lower_err = coefs - ci_lower
    upper_err = ci_upper - coefs
    
    y_pos = np.arange(len(coefs))
    
    # Plot points with error bars
    ax.errorbar(coefs.values, y_pos, 
                xerr=[lower_err.values, upper_err.values],
                fmt='o', markersize=7, capsize=4, capthick=1.5,
                color='#d93a3a', ecolor='#d93a3a', 
                markeredgecolor='darkred', markeredgewidth=1.2,
                linewidth=1.8, alpha=0.8)
    
    # Add coefficient value labels with color coding
    for i, (idx, coef) in enumerate(coefs.items()):
        if coef > 0:
            ha = 'left'
            offset = 0.03
            color = 'darkred'
        else:
            ha = 'right'
            offset = -0.03
            color = 'darkgreen'
        
        # Add coefficient value
        ax.text(coef + offset, i, f'{coef:.3f}', 
                va='center', ha=ha, fontsize=6.5, color=color, weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=color, alpha=0.7, linewidth=1))
    
    # Create interpretable y-axis labels with risk direction
    new_labels = []
    for idx, coef in coefs.items():
        # Get readable base name
        base_name = get_readable_name(idx)
        
        # Add direction based on variable type
        if idx.startswith('MED_') or idx in ['PSYCH', 'NEURL', 'HEAD', 'CARD', 'RESP', 
                                              'HEPAT', 'DERM', 'MUSCL', 'ENDO', 'GAST', 
                                              'HEMA', 'RENA', 'ALLE', 'ALCH']:
            # Binary: Yes/No
            if coef > 0:
                new_labels.append(f"{base_name} [Yes → ↑risk]")
            else:
                new_labels.append(f"{base_name} [Yes → ↓risk]")
        
        elif idx == 'e4_GENOTYPE':
            if coef > 0:
                new_labels.append(f"{base_name} [+1 copy → ↑risk]")
            else:
                new_labels.append(f"{base_name} [+1 copy → ↓risk]")
        
        elif idx == 'PTGENDER':
            if coef > 0:
                new_labels.append(f"{base_name} [Female → ↑risk]")
            else:
                new_labels.append(f"{base_name} [Male → ↑risk]")
        
        # Continuous variables
        else:
            if coef > 0:
                new_labels.append(f"{base_name} [↑value → ↑risk]")
            else:
                new_labels.append(f"{base_name} [↑value → ↓risk]")
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(new_labels, fontsize=8)
    ax.set_xlabel('Log Hazard Ratio (95% CI)', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name} Model', fontsize=11, fontweight='bold', loc='left')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='darkred', label='Positive coef (↑risk)', linewidth=1.5),
        Patch(facecolor='white', edgecolor='darkgreen', label='Negative coef (↓risk)', linewidth=1.5)
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, 
             framealpha=0.9, edgecolor='black',
             title='Interpretation:', title_fontsize=7)
    

    
    # Set x-axis limits with padding
    all_values = np.concatenate([ci_lower.values, ci_upper.values])
    max_abs = max(abs(all_values.min()), abs(all_values.max()))
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

def plot_kaplan_meier(ax, data, model_name):
    """Plot Kaplan-Meier curves stratified by risk groups using held-out test data."""
    test_df = data['test_data']  # Use held-out test data
    model = data['model']
    
    # Test data already has time, event, and covariates
    # Get covariate columns from model
    covariate_cols = list(model.summary.index)
    available_cols = [col for col in covariate_cols if col in test_df.columns]
    
    if not available_cols or 'time' not in test_df.columns or 'event' not in test_df.columns:
        ax.text(0.5, 0.5, f'Insufficient data for {model_name}', 
               transform=ax.transAxes, ha='center', va='center')
        return
    
    # Use test data directly (already prepared)
    surv_df = test_df[['time', 'event'] + available_cols].copy()
    surv_df = surv_df.dropna()
    
    # Calculate risk scores and create deterministic tertiles (Low < Med < High)
    try:
        risk_scores = model.predict_log_partial_hazard(surv_df[available_cols])
        # Ensure 1D Series aligned to surv_df
        if isinstance(risk_scores, pd.DataFrame):
            risk_scores = risk_scores.iloc[:, 0]
        risk_scores = pd.Series(np.asarray(risk_scores).reshape(-1), index=surv_df.index)

        # Prefer qcut on scores; fall back to rank-based if duplicates collapse bins
        try:
            q_idx = pd.qcut(risk_scores, q=3, labels=False, duplicates='drop')
        except Exception:
            ranks = risk_scores.rank(method='average')
            q_idx = pd.qcut(ranks, q=3, labels=False, duplicates='drop')

        # If we couldn't form 3 distinct bins, bail out gracefully
        if pd.Series(q_idx).nunique() < 3:
            ax.text(0.5, 0.5, f'Insufficient risk stratification for {model_name}',
                    transform=ax.transAxes, ha='center', va='center')
            return

        label_map = {0: 'Low Risk', 1: 'M2 Risk', 2: 'High Risk'}
        risk_groups = pd.Series([label_map[int(i)] for i in q_idx], index=risk_scores.index, dtype='category')
        
        # Plot Kaplan-Meier curves with confidence intervals
        kmf = KaplanMeierFitter()
        
        for group in ['Low Risk', 'M2 Risk', 'High Risk']:
            mask = risk_groups == group
            if mask.sum() > 0:
                kmf.fit(surv_df.loc[mask, 'time'], 
                       surv_df.loc[mask, 'event'], 
                       label=f'{group} (n={mask.sum()})')
                
                # Plot with confidence intervals
                kmf.plot_survival_function(ax=ax, ci_show=True, 
                                          color=RISK_COLORS[group],
                                          linewidth=2.5, alpha=0.9)
        
        ax.set_xlabel('Time (years)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name} Model', fontsize=11, fontweight='bold', loc='left')
        ax.legend(loc='lower left', fontsize=8, frameon=True, 
                 framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, None])
        
        # Add concordance index annotation
        concordance = data['concordance']
        ax.text(0.98, 0.98, f'C-index = {concordance:.3f}', 
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.9, edgecolor=MODEL_COLORS[model_name], linewidth=2))
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', 
               transform=ax.transAxes, ha='center', va='center', fontsize=9)
        print(f"Warning: Could not create KM plot for {model_name}: {e}")

def plot_model_comparison(ax, data_dict):
    """Plot concordance index comparison across models."""
    models = ['M3', 'M2', 'M1']
    concordances = [data_dict[m]['concordance'] for m in models]
    partial_aics = [data_dict[m]['partial_aic'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Stats for scaling and deltas
    max_aic = max(partial_aics)
    min_aic = min(partial_aics)
    max_c = max(concordances)
    min_c = min(concordances)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, concordances, width, label='Concordance Index',
                   color=[MODEL_COLORS[m] for m in models], alpha=0.7,
                   edgecolor='black', linewidth=1.2)
    
    # Create twin axis for AIC
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, partial_aics, width, label='Partial AIC',
                   color=[MODEL_COLORS[m] for m in models], alpha=0.4,
                   edgecolor='black', linewidth=1.2, hatch='///')
    
    # Add value labels with deltas (Δ vs best)
    best_c = max_c
    best_aic = min_aic
    c_annotations = []
    aic_annotations = []
    
    for i, (c, aic) in enumerate(zip(concordances, partial_aics)):
        c_text_y = c + (max_c - min_c) * 0.05 + 0.002
        aic_text_y = aic + (max_aic - min_aic) * 0.06 + 5
        
        ax.text(
            i - width/2,
            c_text_y,
            f"{c:.3f}\nΔ={c - best_c:+.3f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )
        ax2.text(
            i + width/2,
            aic_text_y,
            f"{aic:.0f}\nΔ={aic - best_aic:+.0f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold'
        )
        
        # Store max annotation positions (approximate height of 2-line text)
        c_annotations.append(c_text_y + 0.008)  # Add approximate text height
        aic_annotations.append(aic_text_y + 15)  # Add approximate text height
    
    # Formatting
    ax.set_ylabel('Concordance Index (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Partial AIC (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison (Δ vs best)', fontsize=13, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    
    # Calculate dynamic margins based on annotation heights
    max_c_annotation = max(c_annotations)
    max_aic_annotation = max(aic_annotations)
    
    # Set limits with 8% extra headroom above annotations
    c_upper = min(1.0, max_c_annotation * 1.08)
    aic_upper = max_aic_annotation * 1.08
    
    c_margin = max(0.02, (max_c - min_c) * 0.25)
    aic_margin = max(20, (max_aic - min_aic) * 0.25)
    
    ax.set_ylim([max(0, min_c - c_margin), c_upper])
    ax2.set_ylim([max(0, min_aic - aic_margin), aic_upper])
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Legends: show both metrics and model color mapping
    from matplotlib.patches import Patch

    # Metrics legend (filled vs hatched)
    metric_handles = [
        Patch(facecolor='lightgray', edgecolor='black', label='Concordance Index', linewidth=1.2, alpha=0.8),
        Patch(facecolor='lightgray', edgecolor='black', label='Partial AIC', linewidth=1.2, hatch='///', alpha=0.6)
    ]
    legend_metrics = ax.legend(
        handles=metric_handles,
        loc='upper right',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fontsize=10,
        title='Metrics:',
        title_fontsize=10
    )
    ax.add_artist(legend_metrics)

    # Model color mapping legend
    model_handles = [
        Patch(facecolor=MODEL_COLORS[m], edgecolor='black', label=m, linewidth=1.2) for m in models
    ]
    ax.legend(
        handles=model_handles,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor='black',
        fontsize=10,
        title='Models:',
        title_fontsize=10,
        ncol=1
    )

def create_publication_figure():
    """Create the main publication figure for survival models."""
    print("Loading survival model data...")
    data_dict = {}
    for size in ['M3', 'M2', 'M1']:
        print(f"  Loading {size} model...")
        try:
            data_dict[size] = load_model_data(size)
            print(f"    Concordance: {data_dict[size]['concordance']:.3f}")
            print(f"    Partial AIC: {data_dict[size]['partial_aic']:.1f}")
        except Exception as e:
            print(f"    Error loading {size}: {e}")
    
    if len(data_dict) == 0:
        print("Error: No models loaded successfully")
        return
    
    print("\nCreating publication figure...")
    
    # Create figure with 2 columns
    # Left column: Forest plots (3 rows)
    # Right column: KM curves (top 3 rows) + Model comparison (bottom)
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35,
                          left=0.08, right=0.96, top=0.96, bottom=0.05,
                          height_ratios=[1, 1, 1, 1])
    
    # Left column - Forest plots
    ax1 = fig.add_subplot(gs[0, 0])  # M3
    ax2 = fig.add_subplot(gs[1, 0])  # M2
    ax3 = fig.add_subplot(gs[2, 0])  # M1
    
    forest_axes = [(ax1, 'M3'), (ax2, 'M2'), (ax3, 'M1')]
    for ax, size in forest_axes:
        if size in data_dict:
            plot_forest_plot(ax, data_dict[size], size, top_n=12)
    
    # Right column - Kaplan-Meier curves
    ax4 = fig.add_subplot(gs[0, 1])  # M3
    ax5 = fig.add_subplot(gs[1, 1])  # M2
    ax6 = fig.add_subplot(gs[2, 1])  # M1
    
    km_axes = [(ax4, 'M3'), (ax5, 'M2'), (ax6, 'M1')]
    for ax, size in km_axes:
        if size in data_dict:
            plot_kaplan_meier(ax, data_dict[size], size)
    
    # Bottom left - Model comparison metrics
    ax7 = fig.add_subplot(gs[3, :])
    plot_model_comparison(ax7, data_dict)
    
    # Add panel labels
    panel_info = [
        (ax1, 'A1'), (ax2, 'A2'), (ax3, 'A3'),
        (ax4, 'B1'), (ax5, 'B2'), (ax6, 'B3'),
        (ax7, 'C')
    ]
    for ax, label in panel_info:
        ax.text(-0.15, 1.08, label, transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')
    
    # Save figure
    output_path = "viz/survival_results/publication_figure_srv.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    # Also save as high-res PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.close()
    
    print("\n[SUCCESS] Survival publication figure created successfully!")

if __name__ == "__main__":
    create_publication_figure()

