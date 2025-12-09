import pandas as pd
from datetime import datetime 



# Load the CSV file
cdr = pd.read_csv("datasets/All_subjects_CDR_07Sep2025.csv")
cdr = cdr[["RID", "VISDATE", "CDRSB"]]
cdr = cdr.drop(columns=["VISDATE"])
output_path = "processed/cdr.csv"
cdr.to_csv(output_path, index=False)
