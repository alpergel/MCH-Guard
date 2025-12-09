import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# ------------------------------------------------------------------
# 0. Load and basic cleaning
# ------------------------------------------------------------------
FILE = "./processed/worsening_m1.csv"      # adjust path
DATECOL = "SCANDATE"
IDCOL   = "RID"
EVENTCOL= "MCH_pos"                     # 1 = event of interest
df = pd.read_csv(FILE, parse_dates=[DATECOL])

# ensure binary 0/1
df[EVENTCOL] = df[EVENTCOL].astype(int)


# static covariates (deduplicated)
demographics = [
    "PTGENDER",
    "PTAGE",
    "PTEDUCAT",
    "RACE_ETHNICITY",
    "MED_Anti_Thrombotic",
    "MED_AD_and_Dementia",
    "MED_Lipid_Lowering",
    "MED_Blood_Pressure",
    "MED_Thyroid",
    "MED_Vitamins/Minerals",
    "MED_NSAIDs",
    "MED_Herbal_Supplements",
    "MED_Glaucoma",
    "MED_Analgesics",
    "MED_Antidepressants",
    "MED_Diabetes_Oral",
    "MED_Other_GI",
    "MED_General_Urological",
    "MED_Antibiotics",
    "MED_Osteoporosis",
    "MED_Inhalers",
    "MED_Sleep_Aids",
    "MED_Hormone_Replacement",
    "MED_GERD/PPI",
    "MED_Anticonvulsants",
    "MED_Antihistamines",
    "MED_H2_Blockers",
    "MED_Nasal_Sprays",
    "MED_Diabetes_Injectable",
    "MED_Corticosteroids",
    "MED_Benzodiazepines",
    "MED_Eye_Supplements",
    "MED_Other_Respiratory",
    "MED_Antiviral",
    "MED_Anti_Arrythmia",
    "MED_Parkinsons",
    "e4_GENOTYPE",
    "PSYCH",
    "NEURL",
    "HEAD",
    "CARD",
    "RESP",
    "HEPAT",
    "DERM",
    "MUSCL",
    "ENDO",
    "GAST",
    "HEMA",
    "RENA",
    "ALLE",
    "ALCH",
]

# ------------------------------------------------------------------
# 1. Helper: z-score numeric columns (fitted on first use)
# ------------------------------------------------------------------
def zscore(df_in, cols):
    sc = StandardScaler()
    df_out = df_in.copy()
    df_out[cols] = sc.fit_transform(df_out[cols])
    return df_out

# ------------------------------------------------------------------
# Helper: prune near-zero-variance and highly correlated predictors
# ------------------------------------------------------------------
def prune(df_in, thresh: float = 0.9, var_thresh: float = 0.01):
    """Drop columns with ~zero variance and one of any pair with |ρ|>thresh.
    
    Args:
        thresh: Correlation threshold (default 0.9) - removes highly correlated features
        var_thresh: Variance threshold (default 0.01) - removes low-variance features
                    For binary columns, checks if proportion of one value > 99% or < 1%
    """
    # Only consider numeric columns for std/corr calculations
    numeric_cols = df_in.select_dtypes(include=['number', 'float', 'int']).columns

    # Check for low variance: use higher threshold to catch problematic columns
    std_vals = df_in[numeric_cols].std()
    std_mask = std_vals > var_thresh
    
    # Also check for binary columns with extreme proportions (mostly 0s or mostly 1s)
    # This catches columns that pass std threshold but are still problematic
    cols_to_remove = []
    for col in numeric_cols:
        if col in std_mask.index and std_mask[col]:  # Column passed std threshold
            unique_vals = df_in[col].dropna().unique()
            if len(unique_vals) <= 2:  # Binary or near-binary column
                prop = df_in[col].dropna().mean()
                if prop < 0.01 or prop > 0.99:  # Less than 1% or more than 99% are 1s
                    cols_to_remove.append(col)
                    std_mask[col] = False
    
    removed_low_var_cols = list(std_vals.index[~std_mask])
    if removed_low_var_cols:
        print(f"[prune] Removed {len(removed_low_var_cols)} low-variance columns: {removed_low_var_cols}")

    df_numeric = df_in[numeric_cols].loc[:, std_mask]
    # Prepare output: all columns, but we will drop high correlation from numeric only
    non_numeric_cols = [col for col in df_in.columns if col not in numeric_cols]

    if df_numeric.shape[1] < 2:
        # Return original with only low-variance numeric columns removed
        result = df_in.drop(columns=removed_low_var_cols)
        return result

    # Correlation calculation
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > thresh)]
    if drop:
        print(f"[prune] Removed {len(drop)} highly correlated columns: {drop}")

    # Drop from numeric
    pruned_numeric = df_numeric.drop(columns=drop)
    # Return all non-numeric columns plus remaining numeric cols, preserving original order
    keep_cols = non_numeric_cols + list(pruned_numeric.columns)
    # Remove duplicates in keep_cols while preserving order
    seen = set()
    final_cols = [x for x in keep_cols if not (x in seen or seen.add(x))]
    result = df_in.loc[:, final_cols]
    return result

# ------------------------------------------------------------------
# 2. DATASET A – baseline only (one row / patient)
# ------------------------------------------------------------------
first = df.sort_values([IDCOL, DATECOL]).groupby(IDCOL).first()
baseline_date = df.groupby(IDCOL)[DATECOL].min()
event_date    = df[df[EVENTCOL]==1].groupby(IDCOL)[DATECOL].min()
last_date     = df.groupby(IDCOL)[DATECOL].max()

A = pd.DataFrame({
    "baseline": baseline_date,
    "event_date": event_date,
    "last": last_date
})
A["event"] = A["event_date"].notna().astype(int)
A["time"]  = ((A["event_date"].fillna(A["last"])) - A["baseline"]).dt.days/365.25
A = A.join(first[demographics])
A = zscore(A, ["PTAGE"])

# ------------------------------------------------------------------
# 4. Train-Test Split
# ------------------------------------------------------------------
from sklearn.model_selection import train_test_split

print("=== MODEL A  (baseline Cox) ===")
num_cols = ["time", "event"] + demographics
A_model  = A[num_cols].dropna().copy()

# make sure categoricals are numeric
A_model["PTGENDER"] = A_model["PTGENDER"].astype(int)

# Split data (80/20 train/test)
A_train, A_test = train_test_split(A_model, test_size=0.2, random_state=42, stratify=A_model["event"])
print(f"Training set: {len(A_train)} samples, Test set: {len(A_test)} samples")

# -------------------------------------------------------------
# Hyperparameter tuning for baseline Cox model using Optuna
# -------------------------------------------------------------
import optuna
from lifelines.utils import k_fold_cross_validation


# Keep only the selected columns in test set
A_test = A_test[A_train.columns]

# prune redundant predictors then fit
A_train = prune(A_train)
A_test = A_test[A_train.columns]

def objective(trial):
    penalizer = trial.suggest_float("penalizer", 1e-4, 10.0, log=True)
    l1_ratio  = trial.suggest_float("l1_ratio", 0.0, 1.0)
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        # lifelines returns list of c-index for each fold
        c_indexes = k_fold_cross_validation(cph, A_train, duration_col="time", event_col="event", k=10)
        return -np.mean(c_indexes)   # Optuna minimizes, so negate
    except Exception:
        # in case of convergence failure give poor score
        return 1.0

print("Running Optuna hyperparameter search (baseline model)...")
study = optuna.create_study()
study.optimize(objective, n_trials=50, show_progress_bar=True)
print("Best params:", study.best_params)

best_cph = CoxPHFitter(penalizer=study.best_params["penalizer"], l1_ratio=study.best_params["l1_ratio"])
best_cph.fit(A_train, duration_col="time", event_col="event")

print("\n=== Training Set Performance ===")
print(best_cph.summary[["coef","p"]])
print(f"Concordance (train): {best_cph.concordance_index_:.3f}  pAIC: {best_cph.AIC_partial_:.1f}")

# Evaluate on test set
test_concordance = best_cph.concordance_index_
test_concordance_actual = best_cph.score(A_test, scoring_method="concordance_index")
print(f"\n=== Test Set Performance ===")
print(f"Concordance (test): {test_concordance_actual:.3f}\n")

# ------------------------------------------------------------------
# Save the model and test data
# ------------------------------------------------------------------
import joblib
joblib.dump(best_cph, 'models/srv_m1_coxph.joblib')
print("Model saved to models/srv_m1_coxph.joblib")

# Save test data for publication figures
A_test.to_csv('processed/SRV_m1_test.csv', index=True)
print("Test data saved to processed/SRV_m1_test.csv")

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
print("Generating visualizations...")

# 2. Forest Plot of Coefficients with Confidence Intervals
fig, ax = plt.subplots(figsize=(10, 12))
summary = best_cph.summary

# Sort by coefficient value
summary_sorted = summary.sort_values('coef')

# Get coefficients and confidence intervals
coefs = summary_sorted['coef']
ci_lower = summary_sorted['coef lower 95%']
ci_upper = summary_sorted['coef upper 95%']

# Calculate error bar sizes (distance from coefficient to CI bounds)
lower_err = coefs - ci_lower
upper_err = ci_upper - coefs

y_pos = np.arange(len(coefs))

# Plot points with error bars
ax.errorbar(coefs.values, y_pos, 
            xerr=[lower_err.values, upper_err.values],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='#d93a3a', ecolor='#d93a3a', 
            markeredgecolor='darkred', markeredgewidth=1.5,
            linewidth=2, alpha=0.8)

# Add coefficient value labels and risk direction
for i, (idx, coef) in enumerate(coefs.items()):
    # Determine risk direction
    if coef > 0:
        risk_label = "↑ value → ↑ risk"
        ha = 'left'
        offset = 0.05
        color = 'darkred'
    else:
        risk_label = "↑ value → ↓ risk"
        ha = 'right'
        offset = -0.05
        color = 'darkgreen'
    
    # Add coefficient value
    ax.text(coef + offset, i, f'{coef:.3f}', 
            va='center', ha=ha, fontsize=9, color=color, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.7))

# Create interpretable y-axis labels with risk direction
new_labels = []
for idx, coef in coefs.items():
    # Determine variable type and create appropriate label

    # Binary variables (0/1)
    if idx.startswith('MED_') or idx in ['PSYCH', 'NEURL', 'HEAD', 'CARD', 'RESP', 
                                          'HEPAT', 'DERM', 'MUSCL', 'ENDO', 'GAST', 
                                          'HEMA', 'RENA', 'ALLE', 'ALCH']:
        # Positive coef: 1/Yes increases risk; Negative: 1/Yes decreases risk
        if coef > 0:
            new_labels.append(f"{idx} [Yes → ↑risk]")
        else:
            new_labels.append(f"{idx} [Yes → ↓risk]")
    
    # e4_GENOTYPE: ordinal categorical (0, 1, 2 copies of APOE e4)
    elif idx == 'e4_GENOTYPE':
        if coef > 0:
            new_labels.append(f"{idx} [+1 copy → ↑risk]")
        else:
            new_labels.append(f"{idx} [+1 copy → ↓risk]")
    
    # PTGENDER: categorical (1=Male, 2=Female typically)
    elif idx == 'PTGENDER':
        if coef > 0:
            new_labels.append(f"{idx} [2 → ↑risk]")
        else:
            new_labels.append(f"{idx} [1 → ↑risk]")
    
    # PTEDUCAT: binary (0, 1)
    elif idx == 'PTEDUCAT':
        if coef > 0:
            new_labels.append(f"{idx} [1 → ↑risk]")
        else:
            new_labels.append(f"{idx} [0 → ↑risk]")
    
    # RACE_ETHNICITY: categorical
    elif idx == 'RACE_ETHNICITY':
        if coef > 0:
            new_labels.append(f"{idx} [Higher → ↑risk]")
        else:
            new_labels.append(f"{idx} [Lower → ↑risk]")
    
    # Continuous variables (age, biomarkers, imaging)
    else:
        if coef > 0:
            new_labels.append(f"{idx} [↑value → ↑risk]")
        else:
            new_labels.append(f"{idx} [↑value → ↓risk]")

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(new_labels, fontsize=9)
ax.set_xlabel('Log Hazard Ratio (95% CI)', fontsize=12, weight='bold')
ax.set_title('Feature Effects on Risk with 95% Confidence Intervals - M1 Dataset', 
             fontsize=14, pad=20, weight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.5, label='No effect')
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend explaining interpretation
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='white', edgecolor='darkred', label='Positive coefficient (↑risk)', linewidth=2),
    Patch(facecolor='white', edgecolor='darkgreen', label='Negative coefficient (↓risk)', linewidth=2)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9,
         title='Interpretation Guide:', title_fontsize=9)

# Set x-axis limits with some padding
all_values = np.concatenate([ci_lower.values, ci_upper.values])
max_abs = max(abs(all_values.min()), abs(all_values.max()))
ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

plt.tight_layout()
plt.savefig('viz/survival_results/forest_plot_m1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 3. Kaplan-Meier Risk Curves (on TEST data)
# Calculate risk scores on test set and create risk groups
risk_scores_test = best_cph.predict_partial_hazard(A_test)
risk_groups_test = pd.qcut(risk_scores_test, q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])

# Plot Kaplan-Meier curves for each risk group
fig, ax = plt.subplots(figsize=(10, 6))
kmf = KaplanMeierFitter()

for group in ['Low Risk', 'Medium Risk', 'High Risk']:
    mask = risk_groups_test == group
    kmf.fit(A_test.loc[mask, 'time'], 
            A_test.loc[mask, 'event'], 
            label=f'{group} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Survival Probability')
ax.set_title('Kaplan-Meier Curves by Risk Group - M1 Dataset (Test Set)')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('viz/survival_results/kaplan_meier_m1.png', dpi=300)
plt.close()

print("Visualizations saved:")
print("- forest_plot_m1.png")
print("- kaplan_meier_m1.png")

