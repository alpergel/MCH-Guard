import pandas as pd

# Load the CSV file
upenn_csf_biomarkers = pd.read_csv("datasets/UPENNBIOMK_ROCHE_ELECSYS_30Aug2024.csv", delimiter=",")

# Keep only the most recent replicate for replicated samples
upenn_csf_biomarkers = upenn_csf_biomarkers.sort_values(by='RUNDATE', ascending=False).drop_duplicates(subset=['RID', 'VISCODE2'], keep='first')

# Rename biomarkers with CSF tags
upenn_csf_biomarkers = upenn_csf_biomarkers.rename(columns={
    'ABETA40': 'ABETA40_csf',
    'ABETA42': 'ABETA42_csf',
    'TAU': 'TAU_csf',
    'PTAU': 'PTAU_csf'
})

# Create positivity status variables
upenn_csf_biomarkers['ABETA42_csf'] = upenn_csf_biomarkers['ABETA42_csf'].round()
upenn_csf_biomarkers['ptau_pos_csf'] = upenn_csf_biomarkers['PTAU_csf'] > 24
upenn_csf_biomarkers['amyloid_pos_csf'] = upenn_csf_biomarkers['ABETA42_csf'] < 980
upenn_csf_biomarkers['ptau_ab_ratio_csf'] = upenn_csf_biomarkers['PTAU_csf'].astype(float) / upenn_csf_biomarkers['ABETA42_csf'].astype(float)
upenn_csf_biomarkers['ad_pathology_pos_csf'] = upenn_csf_biomarkers['ptau_ab_ratio_csf'] > 0.025

# Convert all biomarker variables to numeric
cols_to_convert = ['ABETA42_csf', 'ABETA40_csf', 'TAU_csf', 'PTAU_csf', 'ptau_ab_ratio_csf']
upenn_csf_biomarkers[cols_to_convert] = upenn_csf_biomarkers[cols_to_convert].apply(pd.to_numeric, errors='coerce')



# Remove unnecessary columns
upenn_csf_biomarkers = upenn_csf_biomarkers.drop(columns=['RUNDATE','BATCH','COMMENT','update_stamp','PHASE','PTID','VISCODE2', 'ABETA42_csf', 'TAU_csf', 'PTAU_csf', 'ptau_pos_csf', 'amyloid_pos_csf', 'ad_pathology_pos_csf'])

# Save the Processed Records
output_path = "processed/upenn.csv"
upenn_csf_biomarkers.to_csv(output_path, index=False)