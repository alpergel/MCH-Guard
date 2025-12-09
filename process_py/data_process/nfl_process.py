import pandas as pd
from datetime import datetime 

# Load the CSV file
nfl = pd.read_csv("datasets/ADNI_BLENNOWPLASMANFLLONG_10_03_18_09Jun2025.csv")

# Extract relevant columns: 'RID' and 'CMMED'
subset_data = nfl[['RID', 'DRAW_DATE','PLASMA_NFL']]

# Convert DOB column to date-time for machine readability
subset_data['SCANDATE'] = pd.to_datetime(subset_data['DRAW_DATE'])
subset_data = subset_data.drop(columns=['DRAW_DATE'])

# Normalize the float values in PLASMA_NFL
max_value = subset_data['PLASMA_NFL'].max()
min_value = subset_data['PLASMA_NFL'].min()
subset_data['PLASMA_NFL'] = (subset_data['PLASMA_NFL'] - min_value) / (max_value - min_value)


# Output
output_path = "processed/nfl.csv"
subset_data.to_csv(output_path, index=False)