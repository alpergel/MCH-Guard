import pandas as pd
from datetime import datetime 



# Load the CSV file
apoe = pd.read_csv("datasets/APOERES_13Aug2025.csv")

# Extract relevant columns: 'RID' and 'CMMED'
subset_data = apoe[['RID', 'GENOTYPE']]

# Create a mapping of unique genotypes to numeric category codes
def encode_genotype(genotype):
    if "4/4" in genotype:
        return 2  # High risk (homozygous)
    elif "/4" in genotype or "4/" in genotype:
        return 1  # M2 risk (heterozygous)
    else:
        return 0  # Low risk (no e4 allele)

subset_data['e4_GENOTYPE'] = subset_data['GENOTYPE'].apply(encode_genotype)
#genotype_mapping = {genotype: i for i, genotype in enumerate(subset_data['GENOTYPE'].unique())}

# Apply the mapping to encode the 'GENOTYPE' column
#subset_data['GENOTYPE_encoded'] = subset_data['GENOTYPE'].map(genotype_mapping)


# Output
output_path = "processed/apoe.csv"
subset_data.to_csv(output_path, index=False)