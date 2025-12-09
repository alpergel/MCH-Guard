import pandas as pd
from datetime import datetime 

#RACE_ETHNICITY	12. Ethnic Category	N	1		1=Hispanic or Latino; 2=Not Hispanic or Latino; 3=Unknown
#PTRACCAT	13. Racial Categories	N	1		1=American Indian or Alaskan Native; 2=Asian; 3=Native Hawaiian or Other Pacific Islander; 4=Black or African American; 5=White; 6=More than one race; 7=Unknown


# Load the CSV file
dem = pd.read_csv("datasets/All_Subjects_PTDEMOG_19Oct2024.csv")

# Extract relevant columns: 'RID' and 'CMMED'
subset_data = dem[['RID', 'PTGENDER','PTDOB','PTEDUCAT','PTNOTRT','PTETHCAT','PTRACCAT']]

# Remove all rows with -4 value Gender
subset_data = subset_data[subset_data['PTGENDER'] != -4]

# Convert DOB column to date-time for machine readability
subset_data['PTDOB'] = pd.to_datetime(subset_data['PTDOB'], format='%m/%Y')

# Fill empty data slots for gender
subset_data['PTGENDER'] = subset_data['PTGENDER'].fillna(0)

# Convert PTEDUCAT to binary: 1 = college education or higher (16+ years), 0 = less than college
subset_data['PTEDUCAT'] = subset_data['PTEDUCAT'].apply(lambda x: 1 if x >= 16 else 0 if pd.notna(x) else 0)

# Merge RACE_ETHNICITY and PTRACCAT into 4 race/ethnicity categories
def categorize_race_ethnicity(row):
    """
    Merge ethnic category and race into 4 categories:
    1. Non-hispanic White
    2. Non-hispanic Black
    3. Hispanic Latin American
    4. Other
    """
    eth = row['PTETHCAT']
    race = row['PTRACCAT']
    
    # Hispanic or Latino (regardless of race)
    if eth == 1:
        #'Hispanic Latin American'
        return 3
    # Not Hispanic or Latino
    elif eth == 2:
        if race == 5:  # White
            #'Non-hispanic White'
            return 1
        elif race == 4:  # Black or African American
            # 'Non-hispanic Black'
            return 2
        else:  # Asian, Native American, Pacific Islander, More than one race, or Unknown
           # 'Other'
            return  4
    # Unknown ethnicity or any other case
    else:
        return 4

subset_data['RACE_ETHNICITY'] = subset_data.apply(categorize_race_ethnicity, axis=1)
subset_data = subset_data.drop(columns=['PTETHCAT', 'PTRACCAT'])

# Output
output_path = "processed/dem.csv"
subset_data.to_csv(output_path, index=False)