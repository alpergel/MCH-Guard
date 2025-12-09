import pandas as pd
from datetime import datetime 



# Load the CSV file
dx = pd.read_csv("datasets/All_Subjects_DXSUM_07Sep2025.csv")
dx = dx[["RID", "EXAMDATE", "DIAGNOSIS"]]
dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"])
dx = dx.rename(columns={"EXAMDATE": "SCANDATE"})
dx = dx.dropna()

output_path = "processed/dx.csv"
dx.to_csv(output_path, index=False)
