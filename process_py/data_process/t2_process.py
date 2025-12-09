import pandas as pd

# Load the CSV file
t2 = pd.read_csv("datasets/MAYOADIRL_MRI_MCH_30Aug2024.csv")

t2 = t2.drop(columns=['COLPROT','VISCODE','VISCODE2','LONI_STUDY','LONI_SERIES','LONI_IMG_ID','SERIES_NUM','CHOSEN','STUDY_QUALITY','SERIES_QUALITY','EVALDATE','SERIES_UID','update_stamp'])
"""Keep rows for TYPE == 'MCH' and rows with NOFINDINGS == 1 so that
both positive findings and clean scans are represented in the output.
"""
t2 = t2[(t2['TYPE'] == 'MCH') | (t2['NOFINDINGS'] == 1)].copy()

# Compute how many TYPE=='MCH' rows exist per (RID, SCANDATE)
mch_only = t2[t2['TYPE'] == 'MCH'].copy()
mch_counts = (
    mch_only.groupby(['RID', 'SCANDATE'], as_index=False)
            .size()
            .rename(columns={'size': 'MCH_count'})
)

# Ensure at most one row per (RID, SCANDATE) and attach MCH_count
t2 = (
    t2.sort_values(['RID', 'SCANDATE'])
      .drop_duplicates(subset=['RID', 'SCANDATE'], keep='first')
      .merge(mch_counts, on=['RID', 'SCANDATE'], how='left')
      .reset_index(drop=True)
)

# Fill missing counts with 0 and ensure integer type
t2['MCH_count'] = t2['MCH_count'].fillna(0)

# If NOFINDINGS is 1, MCH_count must be 0
nof_numeric = pd.to_numeric(t2['NOFINDINGS'], errors='coerce')
t2.loc[nof_numeric == 1, 'MCH_count'] = 0

# Finalize type
t2['MCH_count'] = t2['MCH_count'].astype(int)
t2['MCH_pos'] = (t2['NOFINDINGS'] == 0).astype(int)
t2 = t2.drop(columns=['NOFINDINGS'])

output_path = "processed/t2.csv"
t2.to_csv(output_path, index=False)