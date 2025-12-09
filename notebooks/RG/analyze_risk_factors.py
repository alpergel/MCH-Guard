"""
Analyze regression model feature importance to identify factors
that influence MCH progression duration.

RG Model: Predicts Duration (continuous) - time until MCH worsening
"""

import pandas as pd
import joblib
import sys
import numpy as np

# Load the trained model
MODEL_PATH = "models/rg_large_model.joblib"
DATA_PATH = "processed/worsening_large.csv"
MAPPING_PATH = "processed/subclass_mapping.csv"

try:
    model = joblib.load(MODEL_PATH)
except:
    print(f"Error: Could not load model from {MODEL_PATH}")
    print("Please run the training script first (RG_Large_Train.py)")
    sys.exit(1)

# Load medication mapping
try:
    med_mapping = pd.read_csv(MAPPING_PATH)
    med_dict = dict(zip(med_mapping['Encoding'], med_mapping['Subclass']))
except:
    print(f"Warning: Could not load medication mapping from {MAPPING_PATH}")
    med_dict = {}

# Load data to get feature names
try:
    data = pd.read_csv(DATA_PATH)
    drop_cols = [c for c in ['RID', 'SCANDATE', 'MCH_pos', 'MCH_count', 'SWITCH_STATUS', 'Duration'] if c in data.columns]
    data = data.drop(columns=drop_cols)
    data = data.select_dtypes(include=['number'])
    feature_names = data.columns.tolist()
except:
    print(f"Error: Could not load data from {DATA_PATH}")
    sys.exit(1)

# Get feature importances
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("=" * 80)
print("RISK FACTOR ANALYSIS - What Influences MCH Progression Speed?")
print("=" * 80)
print("\nModel: Extra Trees Regression")
print("Target: Duration (Time until MCH worsening in years)")
print("Note: Higher importance = more influential in predicting progression speed")
print()

# Categorize factors
non_modifiable = ['PTGENDER', 'PTAGE', 'RACE_ETHNICITY', 'e4_GENOTYPE', 'GENOTYPE_encoded', 'DIAGNOSIS']
medications = [f for f in feature_names if f.startswith('MED_')]
health_conditions = [f for f in feature_names if f in ['PSYCH', 'NEURL', 'HEAD', 'CARD', 'RESP', 
                                                         'HEPAT', 'DERM', 'MUSCL', 'ENDO', 'GAST', 
                                                         'HEMA', 'RENA', 'ALLE', 'ALCH']]
clinical_measures = [f for f in feature_names if f.startswith('PHC_') or f == 'CDRSB']
biomarkers = ['ptau_ab_ratio_csf', 'PLASMA_NFL']
imaging = ['NORM_WMH', 'embedding_scalar', 'NORM_GRAY']
other_factors = [f for f in feature_names if f not in non_modifiable + medications + health_conditions + clinical_measures + biomarkers + imaging]

print("\n" + "=" * 80)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 80)
for idx, row in feature_importance_df.head(15).iterrows():
    feat = row['feature']
    imp = row['importance']
    percentage = imp * 100
    print(f"{percentage:5.2f}% - {feat}")

print("\n" + "=" * 80)
print("1. NON-MODIFIABLE FACTORS (Cannot be changed)")
print("=" * 80)
non_mod_df = feature_importance_df[feature_importance_df['feature'].isin(non_modifiable)]
if len(non_mod_df) > 0:
    for idx, row in non_mod_df.iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        
        if feat in ['e4_GENOTYPE', 'GENOTYPE_encoded']:
            interpretation = "APOE e4 allele status"
        elif feat == 'PTAGE':
            interpretation = "Age"
        elif feat == 'PTGENDER':
            interpretation = "Gender"
        elif feat == 'RACE_ETHNICITY':
            interpretation = "Race/ethnicity"
        else:
            interpretation = feat
        
        print(f"\n{feat}:")
        print(f"  Importance: {percentage:.2f}%")
        print(f"  Interpretation: {interpretation}")
        print(f"  Rank: #{list(feature_importance_df['feature']).index(feat) + 1} out of {len(feature_names)}")
else:
    print("  None in top features")

print("\n" + "=" * 80)
print("2. MEDICATIONS (Potentially Modifiable - Consult Doctor)")
print("=" * 80)
med_df = feature_importance_df[feature_importance_df['feature'].isin(medications)]
if len(med_df) > 0:
    print("\nTop medication factors:")
    for idx, row in med_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        rank = list(feature_importance_df['feature']).index(feat) + 1
        
        # Decode medication name
        med_name = feat.replace('MED_', '').replace('_', ' ')
        if '/' in med_name:
            med_name = med_name  # Keep original for multi-word names
        
        significance = "***" if percentage > 2.0 else "**" if percentage > 1.0 else "*" if percentage > 0.5 else ""
        print(f"  {med_name}: {percentage:.2f}% importance (Rank #{rank}) {significance}")
else:
    print("  None in model")

print("\n" + "=" * 80)
print("3. HEALTH CONDITIONS (Potentially Preventable/Manageable)")
print("=" * 80)

condition_names = {
    'PSYCH': 'Psychiatric conditions',
    'NEURL': 'Neurological conditions',
    'HEAD': 'Head/headache conditions',
    'CARD': 'Cardiovascular conditions',
    'RESP': 'Respiratory conditions',
    'HEPAT': 'Hepatic/liver conditions',
    'DERM': 'Dermatological conditions',
    'MUSCL': 'Musculoskeletal conditions',
    'ENDO': 'Endocrine conditions',
    'GAST': 'Gastrointestinal conditions',
    'HEMA': 'Hematological conditions',
    'RENA': 'Renal/kidney conditions',
    'ALLE': 'Allergies',
    'ALCH': 'Alcohol use/history'
}

cond_df = feature_importance_df[feature_importance_df['feature'].isin(health_conditions)]
if len(cond_df) > 0:
    print("\nTop health condition factors:")
    for idx, row in cond_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        rank = list(feature_importance_df['feature']).index(feat) + 1
        
        cond_name = condition_names.get(feat, feat)
        significance = "***" if percentage > 2.0 else "**" if percentage > 1.0 else "*" if percentage > 0.5 else ""
        print(f"  {cond_name}: {percentage:.2f}% importance (Rank #{rank}) {significance}")
else:
    print("  None in model")

print("\n" + "=" * 80)
print("4. CLINICAL MEASURES (Targets for Management)")
print("=" * 80)
clinical_df = feature_importance_df[feature_importance_df['feature'].isin(clinical_measures)]
if len(clinical_df) > 0:
    print("\nTop clinical measure factors:")
    for idx, row in clinical_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        rank = list(feature_importance_df['feature']).index(feat) + 1
        
        if feat == 'PHC_BMI':
            description = "Body Mass Index"
        elif feat == 'PHC_Hypertension':
            description = "Hypertension status"
        elif feat == 'PHC_Diabetes':
            description = "Diabetes status"
        elif feat == 'PHC_Heart':
            description = "Heart disease"
        elif feat == 'PHC_Stroke':
            description = "Stroke history"
        elif feat == 'PHC_Smoker':
            description = "Smoking status"
        elif feat == 'CDRSB':
            description = "CDR Sum of Boxes (cognitive decline)"
        elif feat.startswith('PHC_'):
            description = feat.replace('PHC_', '')
        else:
            description = feat
        
        significance = "***" if percentage > 2.0 else "**" if percentage > 1.0 else "*" if percentage > 0.5 else ""
        print(f"  {description}: {percentage:.2f}% importance (Rank #{rank}) {significance}")
else:
    print("  None in model")

print("\n" + "=" * 80)
print("5. BIOMARKERS")
print("=" * 80)
bio_df = feature_importance_df[feature_importance_df['feature'].isin(biomarkers)]
if len(bio_df) > 0:
    print("\nTop biomarker factors:")
    for idx, row in bio_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        rank = list(feature_importance_df['feature']).index(feat) + 1
        
        significance = "***" if percentage > 2.0 else "**" if percentage > 1.0 else "*" if percentage > 0.5 else ""
        print(f"  {feat}: {percentage:.2f}% importance (Rank #{rank}) {significance}")
else:
    print("  None in model")

print("\n" + "=" * 80)
print("6. IMAGING MEASURES")
print("=" * 80)
img_df = feature_importance_df[feature_importance_df['feature'].isin(imaging)]
if len(img_df) > 0:
    print("\nTop imaging factors:")
    for idx, row in img_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        percentage = imp * 100
        rank = list(feature_importance_df['feature']).index(feat) + 1
        
        significance = "***" if percentage > 2.0 else "**" if percentage > 1.0 else "*" if percentage > 0.5 else ""
        print(f"  {feat}: {percentage:.2f}% importance (Rank #{rank}) {significance}")
else:
    print("  None in model")

