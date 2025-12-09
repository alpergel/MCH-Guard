import pandas as pd

# Load the CSV file
wmh = pd.read_csv("datasets/WMH_[ADNI1,GO,2,3].csv")

wmh = wmh[['RID', 'EXAMDATE', 'TOTAL_WMH', 'TOTAL_GRAY','CEREBRUM_TCV']]
wmh['NORM_WMH'] = wmh['TOTAL_WMH'] / wmh['CEREBRUM_TCV']
wmh['NORM_GRAY'] = wmh['TOTAL_GRAY'] / wmh['CEREBRUM_TCV']
wmh = wmh.drop(columns=['TOTAL_WMH','TOTAL_GRAY','CEREBRUM_TCV'])
output_path = "processed/wmh.csv"
wmh.to_csv(output_path, index=False)