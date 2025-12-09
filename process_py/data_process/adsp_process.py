import pandas as pd

# Read the two datasets
cogn = pd.read_csv("datasets/ADSP_PHC_COGN_14Sep2025.csv")
cvrf = pd.read_csv("datasets/ADSP_PHC_CVRF_14Sep2025.csv")

# Ensure scandate columns are in datetime format
cogn['SCANDATE'] = pd.to_datetime(cogn['EXAMDATE'])
cvrf['SCANDATE'] = pd.to_datetime(cvrf['EXAMDATE'])

# Sort both dataframes by RID and SCANDATE for merge_asof, and reset index to ensure monotonicity
cogn = cogn.sort_values(['SCANDATE']).reset_index(drop=True)
cvrf = cvrf.sort_values(['SCANDATE']).reset_index(drop=True)

# # Merge on RID and nearest scandate within 360 days
# merged = pd.merge_asof(
#     cogn,
#     cvrf,
#     by='RID',
#     on='SCANDATE',
#     direction='nearest',
#     tolerance=pd.Timedelta(days=360),
#     suffixes=('_cogn', '_cvrf')
# )
# merged = merged.drop(columns=[
#     'PTID_cogn', 'SUBJID_cogn', 'PHASE_cogn', 'VISCODE_cogn', 'VISCODE2_cogn',
#     'EXAMDATE_cogn', 'update_stamp_cvrf', 'PHC_Visit_cogn',
#     'PHC_Diagnosis_cogn', 'PHC_Sex_cogn', 'PHC_Race_cogn',
#     'PHC_Ethnicity_cogn', 'PHC_Education_cogn','update_stamp_cogn','VISCODE_cvrf','VISCODE2_cvrf','EXAMDATE_cvrf',
#     "PHC_Diagnosis_cvrf","PHC_Sex_cvrf","PHC_Race_cvrf","PHC_Ethnicity_cvrf","PHC_Education_cvrf",
#     'SUBJID_cvrf','PHASE_cvrf','PHC_Visit_cvrf'
# ])
# merged = merged[merged['PTID_cvrf'].notna()]

# merged = merged[['RID', 'SCANDATE', 'PHC_MEM_SE','PHC_EXF','PHC_LAN','PHC_VSP','PHC_BMI','PHC_Hypertension','PHC_Diabetes','PHC_Heart','PHC_Stroke','PHC_SBP','PHC_Smoker','PHC_ASCVD_10y_FRS_Simple_Ageover30']]

# merged = merged.sort_values(['RID', 'SCANDATE']).reset_index(drop=True)

cogn = cogn[['RID', 'SCANDATE', 'PHC_MEM','PHC_EXF','PHC_LAN']]

output_path = "processed/cogn.csv"
cogn.to_csv(output_path, index=False)

cvrf = cvrf[['RID','PHC_BMI','PHC_Hypertension','PHC_Diabetes','PHC_Heart','PHC_Stroke','PHC_Smoker','PHC_ASCVD_10y_FRS_Simple_Ageover30']]
output_path = "processed/cvrf.csv"
cvrf.to_csv(output_path, index=False)