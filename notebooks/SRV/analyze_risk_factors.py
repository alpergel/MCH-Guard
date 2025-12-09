"""
Analyze survival model coefficients to identify modifiable risk factors
that patients can potentially act on to reduce their risk.
"""

import pandas as pd
import joblib
import sys

# Load the trained model
MODEL_PATH = "models/srv_large_coxph.joblib"
MAPPING_PATH = "processed/subclass_mapping.csv"

try:
    model = joblib.load(MODEL_PATH)
except:
    print(f"Error: Could not load model from {MODEL_PATH}")
    print("Please run the training script first (SRV_Train_M1.py)")
    sys.exit(1)

# Load medication mapping
try:
    med_mapping = pd.read_csv(MAPPING_PATH)
    med_dict = dict(zip(med_mapping['Encoding'], med_mapping['Subclass']))
except:
    print(f"Warning: Could not load medication mapping from {MAPPING_PATH}")
    med_dict = {}

# Get coefficients summary
summary = model.summary.copy()
summary = summary.sort_values('coef')

print("=" * 80)
print("RISK FACTOR ANALYSIS - What Can Patients Do to Lower Risk?")
print("=" * 80)
print()

# Categorize factors
non_modifiable = ['PTGENDER', 'PTAGE', 'RACE_ETHNICITY', 'e4_GENOTYPE', 'GENOTYPE_encoded']
medications = [col for col in summary.index if col.startswith('MED_')]
# Medical history columns
potentially_modifiable_conditions = [col for col in summary.index if col in ['PSYCH', 'NEURL', 'HEAD', 'CARD', 'RESP', 
                                                                               'HEPAT', 'DERM', 'MUSCL', 'ENDO', 'GAST', 
                                                                               'HEMA', 'RENA', 'ALLE', 'ALCH']]
other_factors = [col for col in summary.index if col not in non_modifiable + 
                potentially_modifiable_conditions + medications]



print("\n" + "=" * 80)
print("1. NON-MODIFIABLE FACTORS (Cannot be changed)")
print("=" * 80)
for factor in non_modifiable:
    if factor in summary.index:
        coef = summary.loc[factor, 'coef']
        p_val = summary.loc[factor, 'p']
        hr = summary.loc[factor, 'exp(coef)']
        
        if factor == 'e4_GENOTYPE' or factor == 'GENOTYPE_encoded':
            interpretation = "Each additional APOE e4 allele"
        elif factor == 'PTAGE':
            interpretation = "Age (per standard deviation increase)"
        elif factor == 'PTGENDER':
            # In most medical datasets: 1=Male, 2=Female
            if coef > 0:
                interpretation = "Female vs. Male"
            else:
                interpretation = "Male vs. Female"
        elif factor == 'RACE_ETHNICITY':
            interpretation = "Race/ethnicity category"
        else:
            interpretation = factor
        
        risk_direction = "INCREASES" if coef > 0 else "DECREASES"
        significance_note = "" if p_val < 0.05 else " (NOT statistically significant)"
        
        print(f"\n{factor}:")
        print(f"  Coefficient: {coef:.4f} (p={p_val:.4f})")
        print(f"  Hazard Ratio: {hr:.3f}")
        
        if factor == 'PTGENDER':
            print(f"  Interpretation: {interpretation} {risk_direction} risk by {abs((hr-1)*100):.1f}%{significance_note}")
        else:
            print(f"  Interpretation: {interpretation} {risk_direction} risk by {abs((hr-1)*100):.1f}% per unit increase{significance_note}")

print("\n" + "=" * 80)
print("2. MEDICATIONS (Potentially Modifiable - Consult Doctor)")
print("=" * 80)
print("\nMedications that DECREASE risk (protective):")
protective_meds = []
risky_meds = []

for med in medications:
    if med in summary.index:
        coef = summary.loc[med, 'coef']
        p_val = summary.loc[med, 'p']
        hr = summary.loc[med, 'exp(coef)']
        
        # Decode medication name from encoding
        try:
            med_code = int(med.replace('MED_', ''))
            med_name = med_dict.get(med_code, f"Unknown Med ({med_code})")
        except:
            med_name = med.replace('MED_', '').replace('_', ' ')
        
        if coef < 0:
            protective_meds.append((med_name, coef, p_val, hr))
        else:
            risky_meds.append((med_name, coef, p_val, hr))

if protective_meds:
    protective_meds.sort(key=lambda x: x[1])  # Sort by coefficient (most protective first)
    for med_name, coef, p_val, hr in protective_meds:
        risk_reduction = (1 - hr) * 100
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  + {med_name}: {risk_reduction:.1f}% risk reduction (HR={hr:.3f}, p={p_val:.4f}) {significance}")
else:
    print("  None identified")

print("\nMedications that INCREASE risk:")
if risky_meds:
    risky_meds.sort(key=lambda x: x[1], reverse=True)  # Sort by coefficient (most risky first)
    for med_name, coef, p_val, hr in risky_meds:
        risk_increase = (hr - 1) * 100
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  - {med_name}: {risk_increase:.1f}% risk increase (HR={hr:.3f}, p={p_val:.4f}) {significance}")
else:
    print("  None identified")

print("\n" + "=" * 80)
print("3. HEALTH CONDITIONS (Potentially Preventable/Manageable)")
print("=" * 80)
print("\nConditions that INCREASE risk when present:")
risky_conditions = []
protective_conditions = []

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

for cond in potentially_modifiable_conditions:
    if cond in summary.index:
        coef = summary.loc[cond, 'coef']
        p_val = summary.loc[cond, 'p']
        hr = summary.loc[cond, 'exp(coef)']
        cond_name = condition_names.get(cond, cond)
        
        if coef > 0:
            risky_conditions.append((cond_name, coef, p_val, hr))
        else:
            protective_conditions.append((cond_name, coef, p_val, hr))

if risky_conditions:
    risky_conditions.sort(key=lambda x: x[1], reverse=True)
    for cond_name, coef, p_val, hr in risky_conditions:
        risk_increase = (hr - 1) * 100
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  - {cond_name}: {risk_increase:.1f}% risk increase (HR={hr:.3f}, p={p_val:.4f}) {significance}")
else:
    print("  None identified")

if protective_conditions:
    print("\nConditions showing protective effect (unexpected - may need further investigation):")
    protective_conditions.sort(key=lambda x: x[1])
    for cond_name, coef, p_val, hr in protective_conditions:
        risk_reduction = (1 - hr) * 100
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  ? {cond_name}: {risk_reduction:.1f}% risk reduction (HR={hr:.3f}, p={p_val:.4f}) {significance}")

print("\n" + "=" * 80)
print("4. OTHER FACTORS")
print("=" * 80)
for factor in other_factors:
    if factor in summary.index:
        coef = summary.loc[factor, 'coef']
        p_val = summary.loc[factor, 'p']
        hr = summary.loc[factor, 'exp(coef)']
        
        risk_direction = "increases" if coef > 0 else "decreases"
        change = abs((hr - 1) * 100)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"\n{factor}: {risk_direction} risk by {change:.1f}% (HR={hr:.3f}, p={p_val:.4f}) {significance}")

