import pandas as pd

# Load the CSV file
hist = pd.read_csv("datasets/All_Subjects_MEDHIST_31Oct2024.csv")
hist['SCANDATE'] = hist['VISDATE'].copy()
hist = hist.drop(columns=['PHASE','PTID','VISCODE','VISCODE2','VISDATE','ID','SITEID','USERDATE','USERDATE2','update_stamp'])
hist = hist[['RID', 'MHPSYCH', 'MH2NEURL', 'MH3HEAD', 'MH4CARD', 'MH5RESP', 'MH6HEPAT', 'MH7DERM', 'MH8MUSCL', 'MH9ENDO', 'MH10GAST', 'MH11HEMA', 'MH12RENA', 'MH13ALLE', 'MH14ALCH']]

# Define column mapping to remove MH prefix and numbers
column_mapping = {
    'MHPSYCH': 'PSYCH',
    'MH2NEURL': 'NEURL', 
    'MH3HEAD': 'HEAD',
    'MH4CARD': 'CARD',
    'MH5RESP': 'RESP',
    'MH6HEPAT': 'HEPAT',
    'MH7DERM': 'DERM',
    'MH8MUSCL': 'MUSCL',
    'MH9ENDO': 'ENDO',
    'MH10GAST': 'GAST',
    'MH11HEMA': 'HEMA',
    'MH12RENA': 'RENA',
    'MH13ALLE': 'ALLE',
    'MH14ALCH': 'ALCH'
}

# Rename columns
hist = hist.rename(columns=column_mapping)

output_path = "processed/hist.csv"
hist.to_csv(output_path, index=False)