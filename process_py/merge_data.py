#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Merge Processing Script

This script merges multiple datasets from ADNI sources and creates several output datasets
for different analysis purposes including cross-sectional, longitudinal, and classification datasets.
"""

# Standard library imports
import os
import sys
from datetime import timedelta
import logging
from time import perf_counter

# Third-party imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"   # or set to whatever number of cores you want reported

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("merge_data")

script_start_time = perf_counter()

# Flag: normalize continuous variables (float columns) in merge.csv output only
NORMALIZE_CONTINUOUS_MERGE = True

def log_df_info(name, df):
    """Log a concise summary for a DataFrame."""
    try:
        rid_info = df['RID'].nunique() if 'RID' in df.columns else 'N/A'
    except Exception:
        rid_info = 'N/A'
    logger.info(f"{name}: shape={df.shape}, rows={len(df)}, cols={df.shape[1]}, unique_RIDs={rid_info}")

def perform_univariate_analysis(df, group_column='MCH_pos', alpha=0.05):
    """
    Perform univariate statistical comparisons between groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    group_column : str, default='MCH_pos'
        Column name that defines the groups for comparison
    alpha : float, default=0.05
        Significance level for statistical tests
        
    Returns:
    --------
    dict
        Dictionary containing statistical test results for each variable
    """
    if group_column not in df.columns:
        logger.warning(f"Group column '{group_column}' not found in dataset")
        return {}
    
    # Get unique groups
    groups = df[group_column].unique()
    if len(groups) != 2:
        logger.warning(f"Expected 2 groups, found {len(groups)}. Skipping univariate analysis.")
        return {}
    
    group1, group2 = groups
    logger.info(f"Performing univariate analysis comparing groups: {group1} vs {group2}")
    
    # Separate data by groups
    group1_data = df[df[group_column] == group1]
    group2_data = df[df[group_column] == group2]
    
    logger.info(f"Group {group1}: n={len(group1_data)}")
    logger.info(f"Group {group2}: n={len(group2_data)}")
    
    results = {}
    
    # Analyze each numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Exclude group column, RID, MCH_count, and SWITCH_STATUS from analysis
    exclude_cols = {group_column, 'RID', 'MCH_count', 'SWITCH_STATUS'}
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in numeric_cols:
        try:
            # Remove missing values
            group1_vals = group1_data[col].dropna()
            group2_vals = group2_data[col].dropna()
            
            if len(group1_vals) == 0 or len(group2_vals) == 0:
                continue
                
            # Descriptive statistics
            group1_mean = group1_vals.mean()
            group1_std = group1_vals.std()
            group1_median = group1_vals.median()
            
            group2_mean = group2_vals.mean()
            group2_std = group2_vals.std()
            group2_median = group2_vals.median()
            
            # Test for normality (Shapiro-Wilk test on m1er sample)
            min_n = min(len(group1_vals), len(group2_vals))
            if min_n > 3 and min_n <= 5000:  # Shapiro-Wilk is reliable for n <= 5000
                try:
                    _, p_norm1 = stats.shapiro(group1_vals.sample(min(5000, len(group1_vals))))
                    _, p_norm2 = stats.shapiro(group2_vals.sample(min(5000, len(group2_vals))))
                    normal_dist = p_norm1 > alpha and p_norm2 > alpha
                except:
                    normal_dist = False
            else:
                normal_dist = False
            
            # Choose appropriate statistical test
            if normal_dist and len(group1_vals) > 30 and len(group2_vals) > 30:
                # Use t-test for normally distributed data with m3 samples
                statistic, p_value = ttest_ind(group1_vals, group2_vals)
                test_name = "Independent t-test"
            else:
                # Use Mann-Whitney U test for non-normal or m1 samples
                statistic, p_value = mannwhitneyu(group1_vals, group2_vals, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            # Effect size (Cohen's d for t-test, rank-biserial correlation for Mann-Whitney)
            if test_name == "Independent t-test":
                pooled_std = np.sqrt(((len(group1_vals) - 1) * group1_std**2 + 
                                    (len(group2_vals) - 1) * group2_std**2) / 
                                   (len(group1_vals) + len(group2_vals) - 2))
                effect_size = abs(group1_mean - group2_mean) / pooled_std if pooled_std > 0 else 0
            else:
                # Rank-biserial correlation approximation
                effect_size = abs(statistic - (len(group1_vals) * len(group2_vals) / 2)) / (len(group1_vals) * len(group2_vals) / 2)
            
            # Interpret effect size
            if effect_size < 0.2:
                effect_interpretation = "negligible"
            elif effect_size < 0.5:
                effect_interpretation = "m1"
            elif effect_size < 0.8:
                effect_interpretation = "m2"
            else:
                effect_interpretation = "m3"
            
            results[col] = {
                'test_name': test_name,
                'group1_n': len(group1_vals),
                'group2_n': len(group2_vals),
                'group1_mean': group1_mean,
                'group1_std': group1_std,
                'group1_median': group1_median,
                'group2_mean': group2_mean,
                'group2_std': group2_std,
                'group2_median': group2_median,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': effect_size,
                'effect_interpretation': effect_interpretation,
                'normal_distribution': normal_dist
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing column {col}: {e}")
            continue
    
    # Analyze categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # Exclude group column, MCH_count, and SWITCH_STATUS from analysis
    exclude_cols = {group_column, 'MCH_count', 'SWITCH_STATUS'}
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    for col in categorical_cols:
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[col], df[group_column])
            
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                continue
                
            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramer's V for effect size
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            # Interpret Cramer's V
            if cramers_v < 0.1:
                effect_interpretation = "negligible"
            elif cramers_v < 0.3:
                effect_interpretation = "m1"
            elif cramers_v < 0.5:
                effect_interpretation = "m2"
            else:
                effect_interpretation = "m3"
            
            results[col] = {
                'test_name': "Chi-square test",
                'group1_n': contingency_table[group1].sum(),
                'group2_n': contingency_table[group2].sum(),
                'contingency_table': contingency_table.to_dict(),
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': cramers_v,
                'effect_interpretation': effect_interpretation,
                'degrees_of_freedom': dof
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing categorical column {col}: {e}")
            continue
    
    return results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_meds_within_window(rid, scandate, med_data, window_days=360):
    """
    Get medications for a given RID and SCANDATE within a specified time window.
    
    Parameters:
    -----------
    rid : int or str
        Patient identifier
    scandate : datetime
        Reference scan date
    med_data : pandas.DataFrame
        Medication data containing RID, SCANDATE, and medication information
    window_days : int, default=360
        Number of days to consider for the time window (default: 6 months)
        
    Returns:
    --------
    list
        List of medications for the patient within the specified time window
    """
    patient_meds = med_data[med_data['RID'] == rid]
    if patient_meds.empty:
        return []
    
    # Find the closest date within the specified window
    closest_date = min(patient_meds['SCANDATE'], key=lambda x: abs(x - scandate))
    if abs(closest_date - scandate) <= timedelta(days=window_days):
        return patient_meds[patient_meds['SCANDATE'] == closest_date]['PTMEDS'].iloc[0]
    return []

# Calculate PT Age
def calculate_age(birthdate, date):
    """
    Calculate age in years between birthdate and reference date.
    Accounts for month and day to determine if a full year has passed.
    
    Parameters:
    -----------
    birthdate : datetime
        Date of birth
    date : datetime
        Reference date (e.g., scan date)
        
    Returns:
    --------
    int
        Age in years
    """
    age = date.year - birthdate.year
    if (date.month, date.day) < (birthdate.month, birthdate.day):
        age -= 1
    return age
class EmbeddingToScalar:
    def __init__(self):
        self.method = None
        self.fitted_components = {}
    
    def fit_transform(self, embeddings, targets=None, method='pca'):
        """
        Convert high-dimensional embeddings to scalars
        
        Args:
            embeddings: np.array of shape (n_samples, n_dims)
            targets: np.array of shape (n_samples,) - target values for supervised methods
            method: str - conversion method to use
        """
        self.method = method
        embeddings = np.array(embeddings)
        
        if method == 'pca':
            return self._pca_method(embeddings)
            
        elif method == 'supervised':
            if targets is None:
                raise ValueError("Targets required for supervised method")
            return self._supervised_method(embeddings, targets)
            
        elif method == 'l2_norm':
            return self._l2_norm_method(embeddings)
            
        elif method == 'weighted_sum':
            return self._weighted_sum_method(embeddings)
            
        elif method == 'mean':
            return self._mean_method(embeddings)
            
        elif method == 'dominant_component':
            return self._dominant_component_method(embeddings)
        elif method == 'lle':
            return self._lle_method(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def transform(self, embeddings):
        """Transform new embeddings using fitted method"""
        embeddings = np.array(embeddings)
        
        if self.method == 'pca':
            return self.fitted_components['pca'].transform(embeddings)[:, 0]
            
        elif self.method == 'supervised':
            return self.fitted_components['regressor'].predict(embeddings)
            
        elif self.method == 'l2_norm':
            return np.linalg.norm(embeddings, axis=1)
            
        elif self.method == 'weighted_sum':
            return np.dot(embeddings, self.fitted_components['weights'])
            
        elif self.method == 'mean':
            return np.mean(embeddings, axis=1)
            
        elif self.method == 'dominant_component':
            return np.dot(embeddings, self.fitted_components['dominant_weights'])
            
    #NOTE: Locally linear embedding
    def _pca_method(self, embeddings):
        """Use first principal component as scalar"""
        pca = PCA(n_components=1)
        scalar_features = pca.fit_transform(embeddings).flatten()
        self.fitted_components['pca'] = pca
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[0]:.3f}")
        return scalar_features
    
    def _supervised_method(self, embeddings, targets):
        """Train regressor to predict target, use prediction as scalar"""
        regressor = LinearRegression()
        regressor.fit(embeddings, targets)
        scalar_features = regressor.predict(embeddings)
        self.fitted_components['regressor'] = regressor
        logger.info(f"Supervised R² score: {regressor.score(embeddings, targets):.3f}")
        return scalar_features
    
    def _l2_norm_method(self, embeddings):
        """Use L2 (Euclidean) norm of embedding vector"""
        return np.linalg.norm(embeddings, axis=1)
    
    def _weighted_sum_method(self, embeddings):
        """Weighted sum based on feature variance"""
        # Weight by inverse variance to emphasize stable features
        variances = np.var(embeddings, axis=0)
        weights = 1 / (variances + 1e-8)  # Add m1 epsilon to avoid division by zero
        weights /= np.sum(weights)  # Normalize
        self.fitted_components['weights'] = weights
        return np.dot(embeddings, weights)
    
    def _mean_method(self, embeddings):
        """Simple mean of all embedding dimensions"""
        return np.mean(embeddings, axis=1)
    
    def _dominant_component_method(self, embeddings):
        """Weight by absolute correlation with target if available, else by variance"""
        variances = np.var(embeddings, axis=0)
        # Use variance as proxy for importance
        weights = variances / np.sum(variances)
        self.fitted_components['dominant_weights'] = weights
        return np.dot(embeddings, weights)
        
    def _lle_method(self, embeddings):
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=1)
        X_reduced = lle.fit_transform(embeddings)
        return X_reduced

def medical_embedding_to_scalar(embeddings, targets=None, 
                               normalize_first=True, method='pca'):
    """
    Specialized function for medical imaging embedding to scalar conversion
    
    Args:
        embeddings: MRI embeddings (n_samples, n_dims)
        targets: Target values (optional)
        normalize_first: Whether to standardize embeddings first
        method: Conversion method
    """
    
    if normalize_first:
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
    
    converter = EmbeddingToScalar()
    
    if method == 'supervised' and targets is not None:
        scalars = converter.fit_transform(embeddings, targets, method)
        logger.info("Using supervised approach - scalar optimized for target prediction")
    else:
        scalars = converter.fit_transform(embeddings, method=method)
        logger.info(f"Using {method} approach")
    
    return scalars, converter

#----------------------------------------------------------------------------------------------------------------------
# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check for possible data directories
possible_paths = [
    os.path.join(os.path.dirname(script_dir), 'processed'),  # Up one level, then processed dir
    os.path.join(script_dir, 'processed'),                   # In same directory
    os.path.join(script_dir, '..', 'processed'),             # Normalized up one level path
    os.path.join(os.path.dirname(script_dir), 'datasets'),   # Try datasets directory instead
    os.path.join(script_dir, 'datasets'),                    # Datasets in same directory
    'processed',                                             # Direct subdirectory as fallback
]

# Find the first valid data directory
base_path = None
for path in possible_paths:
    if os.path.isdir(path):
        logger.info(f"Found data directory: {path}")
        base_path = path
        break

if base_path is None:
    logger.error("Error: Could not find a valid data directory.")
    sys.exit(1)

# ============================================================================
# LOAD DATA FILES
# ============================================================================

try:
    # Load the CSV files with proper path handling
    upenn_df = pd.read_csv(os.path.join(base_path, 'upenn.csv'))
    mayo_df = pd.read_csv(os.path.join(base_path, 't2.csv'))
    med_df = pd.read_csv(os.path.join(base_path, 'encoded_medicines.csv'))
    dem_df = pd.read_csv(os.path.join(base_path, 'dem.csv'))
    apoe_df = pd.read_csv(os.path.join(base_path, 'apoe.csv'))
    wmh_df = pd.read_csv(os.path.join(base_path, 'wmh.csv'))
    hist_df = pd.read_csv(os.path.join(base_path, 'hist.csv'))
    nfl_df = pd.read_csv(os.path.join(base_path, 'nfl.csv'))
    cdr_df = pd.read_csv(os.path.join(base_path,'cdr.csv'))
    dx_df = pd.read_csv(os.path.join(base_path,'dx.csv'))
    cogn_df = pd.read_csv(os.path.join(base_path,'cogn.csv'))
    cvrf_df = pd.read_csv(os.path.join(base_path,'cvrf.csv'))
    logger.info("All data files loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Error: Could not find a required data file: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading data files: {e}")
    sys.exit(1)


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

logger.info("Starting data preprocessing...")

# Convert date columns to datetime format
logger.info("Converting date columns to datetime format...")
try:
    upenn_df['SCANDATE'] = pd.to_datetime(upenn_df['EXAMDATE'])
    upenn_df = upenn_df.drop(columns=['EXAMDATE'])
    mayo_df['SCANDATE'] = pd.to_datetime(mayo_df['SCANDATE'])
    med_df['SCANDATE'] = pd.to_datetime(med_df['SCANDATE'])
    wmh_df['SCANDATE'] = pd.to_datetime(wmh_df['EXAMDATE'])
    wmh_df = wmh_df.drop(columns=['EXAMDATE'])
    nfl_df['SCANDATE'] = pd.to_datetime(nfl_df['SCANDATE'])
    dx_df['SCANDATE'] = pd.to_datetime(dx_df['SCANDATE'])
    cogn_df['SCANDATE'] = pd.to_datetime(cogn_df['SCANDATE'])
    logger.info("Date conversion completed successfully")
except Exception as e:
    logger.error(f"Error converting date columns: {e}")
    sys.exit(1)

# ============================================================================
# LOAD VECTOR ENCODINGS EARLY (for RID filtering and embedding scalar)
# ============================================================================

logger.info("Loading vector encoding data for RID filtering and embeddings...")
vec = pd.DataFrame()
vec_processing_enabled = False
try:
    # Try to locate the vector encoding data file
    vec_file = pd.read_csv(os.path.join(os.path.dirname(script_dir), 'datasets', 'adni_udip_update.csv'))
    if vec_file is None:
        logger.warning("Vector encoding file not found; downstream vector-dependent datasets may be skipped")
    else:
        vec_processing_enabled = True
        vec_file['SCANDATE'] = pd.to_datetime(vec_file['ScanDate'])
        vec_file = vec_file.sort_values(['SCANDATE'])
        vec_file = vec_file.drop(columns=['T1Code'])
        # Keep UDIP encoding and expand to numeric columns
        encoding_column = 'UDIP'  # Options: 'VectorEncoding' or 'UDIP'
        vec = vec_file.copy()
        if encoding_column in vec.columns and len(vec) > 0:
            vec[encoding_column] = vec[encoding_column].apply(lambda x: eval(x) if isinstance(x, str) else x)
            if len(vec) > 0 and isinstance(vec.iloc[0][encoding_column], list):
                vector_length = len(vec.iloc[0][encoding_column])
                vector_df = pd.DataFrame(
                    vec[encoding_column].tolist(), index=vec.index,
                    columns=[f'Vec_{i}' for i in range(vector_length)]
                )
                vec = pd.concat([vec, vector_df], axis=1)
                vec = vec.drop(columns=[encoding_column])
            else:
                logger.warning(f"{encoding_column} column does not contain proper vector data")
        else:
            logger.warning(f"No {encoding_column} data available in vector file")

        # Compute embedding scalar early so downstream merges can use it directly
        vector_cols = [col for col in vec.columns if col.startswith('Vec_')]
        if len(vector_cols) > 0:
            embeddings = vec[vector_cols].values
            try:
                scalar_features, _ = medical_embedding_to_scalar(
                    embeddings, normalize_first=False, method='lle'
                )
                vec['embedding_scalar'] = scalar_features
            except Exception as e:
                logger.warning(f"Failed to compute embedding scalar via LLE; vectors will be available without scalar. Error: {e}")
        else:
            logger.warning("No Vec_* columns found after expansion; embeddings unavailable")
except Exception as e:
    logger.warning(f"Issue while loading/processing vectors: {e}")


# ============================================================================
# DATA CLEANING
# ============================================================================

logger.info("Cleaning data and filtering relevant records...")

# Clean NAN Rows
logger.info("Removing rows with missing values in key fields...")
upenn_df = upenn_df[upenn_df['ptau_ab_ratio_csf'].notna()]
nfl_df = nfl_df[nfl_df['PLASMA_NFL'].notna()]

# Filter out the relevant RIDs before the loop to avoid repeated filtering
logger.info("Finding common patient IDs across datasets...")
common_rids = (
    set(mayo_df['RID'])
    & set(dem_df['RID'])
    & set (cdr_df['RID'])
    & set (dx_df['RID'])
    & set (cvrf_df['RID'])
    & set(cogn_df['RID'])
    & set(upenn_df['RID'])
    & set(apoe_df['RID'])
    & set(hist_df['RID'])
    & set(wmh_df['RID'])
    & set(nfl_df['RID'])
    & set(med_df['RID'])
)
logger.info(f"Found {len(common_rids)} common patient IDs across all datasets")

_before = len(common_rids)
try:
    vec_rids = set(vec['RID'].astype(int).unique())
except Exception:
    vec_rids = set(vec['RID'].unique())

# Save the list of RIDs and scandates that are missing from vec
missing_from_vec = []
for rid in common_rids:
    if rid not in vec_rids:
        # Get all scandates for this RID from mayo_df
        rid_scandates = mayo_df[mayo_df['RID'] == rid]['SCANDATE'].tolist()
        for scandate in rid_scandates:
            missing_from_vec.append({'RID': rid, 'SCANDATE': scandate})

# Save to CSV file
if missing_from_vec:
    missing_df = pd.DataFrame(missing_from_vec)
    missing_df.to_csv('processed/missing_from_vec.csv', index=False)
    logger.info(f"Saved {len(missing_from_vec)} RID-SCANDATE pairs missing from vectors to processed/missing_from_vec.csv")

common_rids = common_rids.intersection(vec_rids)
logger.info(f"Restricting to RIDs with vectors: {_before} -> {len(common_rids)}")


# Filter the DataFrames to include only common RIDs
logger.info("Filtering datasets to include only common patient IDs...")
mayo_filtered = mayo_df[mayo_df['RID'].isin(common_rids)].copy()
upenn_filtered = upenn_df[upenn_df['RID'].isin(common_rids)].copy()
dem_filtered = dem_df[dem_df['RID'].isin(common_rids)].copy()
cdr_filtered = cdr_df[cdr_df['RID'].isin(common_rids)].copy()
dx_filtered = dx_df[dx_df['RID'].isin(common_rids)].copy()
vec_filtered = vec[vec['RID'].isin(common_rids)].copy()
wmh_filtered = wmh_df[wmh_df['RID'].isin(common_rids)].copy()
apoe_filtered = apoe_df[apoe_df['RID'].isin(common_rids)].copy()
hist_filtered = hist_df[hist_df['RID'].isin(common_rids)].copy()
nfl_filtered = nfl_df[nfl_df['RID'].isin(common_rids)].copy()
cogn_filtered = cogn_df[cogn_df['RID'].isin(common_rids)].copy()
cvrf_filtered = cvrf_df[cvrf_df['RID'].isin(common_rids)].copy()

# Print dataset sizes after filtering
logger.info(f"After filtering: Mayo: {len(mayo_filtered)}, UPENN: {len(upenn_filtered)}, NFL: {len(nfl_filtered)}, WMH: {len(wmh_filtered)}, HIST: {len(hist_filtered)}, DEM: {len(dem_filtered)}, CVRF: {len(cvrf_filtered)}, ADSP:{len(cogn_filtered)}, DX: {len(dx_filtered)}, CDR: {len(cdr_filtered)}, APOE: {len(apoe_filtered)}, VEC: {len(vec_filtered)}")

# Sort once by 'SCANDATE' before merging
mayo_sorted = mayo_filtered.sort_values(['SCANDATE'])
upenn_sorted = upenn_filtered.sort_values(['SCANDATE'])
med_sorted = med_df.sort_values(['SCANDATE'])
wmh_sorted = wmh_filtered.sort_values(['SCANDATE']).copy()
nfl_sorted = nfl_filtered.sort_values(['SCANDATE'])
dx_sorted = dx_filtered.sort_values(['SCANDATE'])
cogn_sorted = cogn_filtered.sort_values(['SCANDATE'])

# Merge Into Dem Data
logger.info("Adding demographic data to merged dataset...")
dem_filtered_unique = dem_filtered.drop_duplicates(subset='RID')
final_merged_df = pd.merge(mayo_sorted, dem_filtered_unique, on='RID', how='left') 

# Merge Into CDR Data
logger.info("Adding CDRSB data to merged dataset...")
cdr_filtered_unique = cdr_filtered.drop_duplicates(subset='RID')
final_merged_df = pd.merge(final_merged_df,cdr_filtered_unique, on='RID', how='left')

# Merge Into CVRF Data
logger.info("Adding CVRF data to merged dataset...")
cvrf_filtered = cvrf_filtered.drop_duplicates(subset='RID')
final_merged_df = pd.merge(final_merged_df, cvrf_filtered, on='RID',how='left')
# Clean up columns
final_merged_df = final_merged_df.drop(columns=['UNIQUEID','TYPE','STATUS','PTNOTRT','FINDCOMMENTS','ATLASREGIONS','RASLOCATIONS','SCAN_COMMENT','PHC_ASCVD_10y_FRS_Simple_Ageover30'])
#final_merged_df = final_merged_df.drop(columns=['UNIQUEID','TYPE','STATUS','PTNOTRT','FINDCOMMENTS','ATLASREGIONS','RASLOCATIONS','SCAN_COMMENT'])

# Merge medication history
logger.info("Adding medication data to merged dataset...")

# Load subclass mapping to convert encoding numbers to subclass names
subclass_mapping_df = pd.read_csv('./processed/subclass_mapping.csv')
subclass_mapping = dict(zip(subclass_mapping_df['Encoding'], subclass_mapping_df['Subclass']))

med_list_df = med_df.groupby(['RID', 'SCANDATE']).agg({
    'General_Class_Encoded': lambda x: list(set(x)),
    'Subclass_Encoded': lambda x: list(set(x)), 
    'Sub_Subclass_Encoded': lambda x: list(set(x))
}).reset_index()
med_list_df.rename(columns={'Subclass_Encoded': 'PTMEDS'}, inplace=True)
med_list_df['SCANDATE'] = pd.to_datetime(med_list_df['SCANDATE'])
final_merged_df['SCANDATE'] = pd.to_datetime(final_merged_df['SCANDATE'])
final_merged_df['PTMEDS'] = final_merged_df.apply(
    lambda row: get_meds_within_window(row['RID'], row['SCANDATE'], med_list_df), 
    axis=1
)
unique_meds = sorted(set(med for meds in final_merged_df['PTMEDS'] for med in meds))
for med in unique_meds:
    # Convert encoding number to subclass name
    subclass_name = subclass_mapping.get(med, f'Unknown_{med}')
    # Clean the subclass name for use as column name (replace spaces and special chars)
    clean_subclass_name = subclass_name.replace(' ', '_').replace('&', 'and').replace("'", '').replace('-', '_')
    final_merged_df[f'MED_{clean_subclass_name}'] = final_merged_df['PTMEDS'].apply(lambda x: 1 if med in x else 0)
final_merged_df = final_merged_df.drop(columns=['PTMEDS'])

# Calculate Patient Age
logger.info("Adding patient age to merged dataset...")
final_merged_df['PTDOB'] = pd.to_datetime(final_merged_df['PTDOB'])
final_merged_df['SCANDATE'] = pd.to_datetime(final_merged_df['SCANDATE'])
final_merged_df['PTAGE'] = final_merged_df.apply(
    lambda row: calculate_age(row['PTDOB'], row['SCANDATE']), axis=1
)
final_merged_df = final_merged_df.drop(columns=['PTDOB'])

# Merge APOE Data
logger.info("Adding APOE Genotype to merged dataset...")
final_merged_df = pd.merge(final_merged_df, apoe_filtered, on='RID', how='left')
final_merged_df = final_merged_df.drop(columns=['GENOTYPE'])

# Merge PT HIST
logger.info("Adding patient medical history to merged dataset...")
final_merged_df = pd.merge(final_merged_df,hist_filtered,on='RID')

# Merge DX Data
logger.info("Adding DX Data to Merged Dataset...")
final_merged_df = pd.merge_asof(
    final_merged_df, dx_sorted, 
    on='SCANDATE', by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')  
)


# Merge COGN Data
logger.info("Adding COGN Data to Merged Dataset...")
final_merged_df = pd.merge_asof(
    final_merged_df, cogn_sorted, 
    on='SCANDATE', by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')  
)
# Merge CSF Biomarker Data
logger.info("Adding CSF Biomarker Data to Merged Dataset...")

final_merged_df = pd.merge_asof(
    final_merged_df, upenn_sorted, 
    on='SCANDATE', by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')  
)
final_merged_df = final_merged_df.dropna(subset=['ptau_ab_ratio_csf'])
final_merged_df = final_merged_df.drop(columns=['ABETA40_csf'])

# Merge NFL Data
final_merged_df = pd.merge_asof(
    final_merged_df, nfl_sorted, 
    on='SCANDATE', by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')  # Allow for a delta of 6 months
)
# 
final_merged_df = final_merged_df.drop_duplicates(subset=['RID', 'ptau_ab_ratio_csf','PLASMA_NFL'])
final_merged_df = final_merged_df[final_merged_df['PLASMA_NFL'].notna()]
final_merged_df = final_merged_df.sort_values(['RID', 'SCANDATE'])

# Merge Vector Embedding + WMH + GRAY
logger.info("Adding Imaging Data to Merged Dataset...")
# Extract vector columns
vector_cols = [col for col in vec.columns if col.startswith('Vec_')]
embeddings = vec[vector_cols].values

# Convert embeddings to scalar using PCA method
scalar_features, converter = medical_embedding_to_scalar(
    embeddings, 
    normalize_first=True,
    method='lle'
)

# Add scalar feature to the dataframe
vec['embedding_scalar'] = scalar_features
# Extract only necessary columns for merge
vec_minimal = vec[['RID', 'SCANDATE', 'embedding_scalar']].copy()
# Ensure consistent dtypes for merge keys
try:
    vec_minimal['RID'] = vec_minimal['RID'].astype('int64')
    vec_minimal['SCANDATE'] = pd.to_datetime(vec_minimal['SCANDATE'])
    final_merged_df['RID'] = final_merged_df['RID'].astype('int64')
    final_merged_df['SCANDATE'] = pd.to_datetime(final_merged_df['SCANDATE'])
    wmh_filtered.loc[:, 'RID'] = wmh_filtered['RID'].astype('int64')
    wmh_filtered.loc[:, 'SCANDATE'] = pd.to_datetime(wmh_filtered['SCANDATE'])
except Exception as _dtype_e:
    logger.warning(f"Type coercion issue prior to worsening-m3 merges: {_dtype_e}")
# Step 1: Merge Vector with WMH 
final_merged_df = pd.merge_asof(
    final_merged_df.sort_values('SCANDATE'), 
    vec_minimal.sort_values('SCANDATE'),
    on='SCANDATE', 
    by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')
)

# Step 2: Merge with the combined dataset
final_merged_df = pd.merge_asof(
    final_merged_df.sort_values('SCANDATE'), 
    wmh_filtered.sort_values('SCANDATE'),
    on='SCANDATE', 
    by='RID', 
    direction='nearest', 
    tolerance=pd.Timedelta('360D')  # Allow for a delta of 12 months
)
final_merged_df = final_merged_df.dropna()

logger.info("Removing duplicated rows by ['RID','SCANDATE']...")
# Deduplicate the data by keeping unique rows per RID and SCANDATE
_before_dedup = len(final_merged_df)
final_merged_df = final_merged_df.drop_duplicates(subset=['RID','SCANDATE'])
_after_dedup = len(final_merged_df)
logger.info(f"Deduplication removed {_before_dedup - _after_dedup} rows (remaining {_after_dedup})")
# Sort preview
final_merged_df = final_merged_df.sort_values(['RID','SCANDATE'])

# Run pruning 
def prune(df_in, thresh: float = 0.9):
    """Drop columns with ~zero variance and one of any pair with |ρ|>thresh."""
    # Only consider numeric columns for std/corr calculations
    numeric_cols = df_in.select_dtypes(include=['number', 'float', 'int']).columns

    # Keep only columns with std above threshold (ignoring Timedelta or object types)
    std_mask = df_in[numeric_cols].std() > 1e-6
    removed_low_var_cols = list(std_mask.index[~std_mask])
    if removed_low_var_cols:
        print(f"[prune] Removed {len(removed_low_var_cols)} near-zero-variance columns: {removed_low_var_cols}")

    df_numeric = df_in[numeric_cols].loc[:, std_mask]
    # Prepare output: all columns, but we will drop high correlation from numeric only
    non_numeric_cols = [col for col in df_in.columns if col not in df_numeric.columns]

    if df_numeric.shape[1] < 2:
        # Return original with only low-variance numeric columns removed
        result = df_in.drop(columns=removed_low_var_cols)
        return result

    # Correlation calculation
    corr = df_numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > thresh)]
    if drop:
        print(f"[prune] Removed {len(drop)} highly correlated columns: {drop}")

    # Drop from numeric
    pruned_numeric = df_numeric.drop(columns=drop)
    # Return all non-numeric columns plus remaining numeric cols, preserving original order
    keep_cols = non_numeric_cols + pruned_numeric.columns.tolist()
    # Remove duplicates in keep_cols while preserving order
    seen = set()
    final_cols = [x for x in keep_cols if not (x in seen or seen.add(x))]
    result = df_in.loc[:, final_cols]
    return result
# final_merged_df = prune(final_merged_df)
# ============================================================================
# SAVE OUTPUT DATASETS
# ============================================================================

logger.info("Saving output datasets...")

# 1. Main merged dataset
output_path = os.path.join(base_path, "merge.csv")
denorm_output_path = os.path.join(base_path, "denorm_merge.csv")

try:
    
    logger.info("Normalizing continuous variables for merge.csv (StandardScaler on float columns)...")
    merged_to_save = final_merged_df.copy()
    merged_to_save.to_csv(denorm_output_path, index=False)

    exclude_cols = {"RID", "SCANDATE"}
    float_cols = ['ptau_ab_ratio_csf', 'PLASMA_NFL', 'embedding_scalar', 'NORM_WMH', 'NORM_GRAY','PHC_MEM','PHC_EXF','PHC_LAN','PHC_BMI','CDRSB']
    if float_cols:
        scaler = StandardScaler()
        merged_to_save[float_cols] = scaler.fit_transform(merged_to_save[float_cols])
        logger.info(f"Normalized {len(float_cols)} float columns")
    else:
        logger.info("No float columns found to normalize")
    merged_to_save.to_csv(output_path, index=False)
    final_merged_df = merged_to_save.copy()
  
    logger.info(f"✓ Main merged dataset saved to {output_path}")
    logger.info(f"  - Shape: {final_merged_df.shape}")
    logger.info(f"  - Unique patients: {final_merged_df['RID'].nunique()}")
    logger.info(f"  - Average datapoints per patient: {final_merged_df.groupby('RID').size().mean():.2f}")
except Exception as e:
    logger.error(f"Error saving main merged dataset: {e}")

# 2. Cross-sectional dataset (random single timepoint per RID)
# try:
#     # Randomly select one row per RID to avoid always choosing baseline or last
#     crossSection = (
#         final_merged_df
#         .groupby('RID', group_keys=False)
#         .apply(lambda x: x.sample(n=min(1, len(x))))
#         .reset_index(drop=True)
#     )
#     crossSection.to_csv(os.path.join(base_path, "cross_section.csv"), index=False)
#     logger.info(f"✓ Cross-sectional dataset saved")
#     logger.info(f"  - Shape: {crossSection.shape}")
# except Exception as e:
#     logger.error(f"Error saving cross-sectional dataset: {e}")

# 3. Longitudinal dataset (multiple timepoints per RID)
logger.info("Creating and saving longitudinal dataset...")
try:
    # Identify patients with multiple timepoints
    longitudinal_df = final_merged_df[final_merged_df['RID'].isin(
        final_merged_df.groupby('RID').size()[lambda x: x > 1].index
    )].sort_values(['RID', 'SCANDATE'])
    
    # Add switch status (RIDs with both MCH_pos=0 and MCH_pos=1)
    nofinding_counts = final_merged_df.groupby('RID')['MCH_pos'].value_counts().unstack(fill_value=0)
    valid_rids = nofinding_counts[(nofinding_counts[0] > 0) & (nofinding_counts[1] > 0)].index
    longitudinal_df['SWITCH_STATUS'] = longitudinal_df['RID'].apply(lambda x: 1 if x in valid_rids else 0)
    
    # Save longitudinal dataset
    longitudinal_df.to_csv(os.path.join(base_path, "longitudinal_progression.csv"), index=False)
    logger.info(f"✓ Longitudinal dataset saved")
    logger.info(f"  - Shape: {longitudinal_df.shape}")
    logger.info(f"  - Patients with status switch: {len(valid_rids)}")
except Exception as e:
    logger.error(f"Error creating longitudinal dataset: {e}")


logger.info("Creating m1 classification dataset...")
try:
    combined_total_df = final_merged_df.copy()
    m1_df = combined_total_df.drop(columns=['ptau_ab_ratio_csf', 'PLASMA_NFL', 'embedding_scalar', 'NORM_WMH', 'NORM_GRAY'])
    m1_df = m1_df.sort_values(['RID', 'SCANDATE'])
    m1_df.to_csv(os.path.join(base_path, "classification_m1.csv"), index=False)
    logger.info(f"✓ M1 classification dataset saved")
    logger.info(f"  - Shape: {m1_df.shape}")
except Exception as e:
    logger.error(f"Error creating m1 classiication dataset: {e}")

# 5. M2 classification dataset 
logger.info("Creating m2 classification dataset...")
try:
    m2_df = combined_total_df.drop(columns=[ 'embedding_scalar', 'NORM_WMH', 'NORM_GRAY'])
    m2_df = m2_df.sort_values(['RID', 'SCANDATE'])
    m2_df.to_csv(os.path.join(base_path, "classification_m2.csv"), index=False)
    logger.info(f"✓ M2 classification dataset saved")
    logger.info(f"  - Shape: {m2_df.shape}")
except Exception as e:
    logger.error(f"Error creating m2 classification dataset: {e}")

# 5. M3 classification dataset 
logger.info("Creating m3 classification dataset...")
try:
    m3_df = combined_total_df.copy()
    m3_df = m3_df.sort_values(['RID', 'SCANDATE'])
    m3_df.to_csv(os.path.join(base_path, "classification_m3.csv"), index=False)
    logger.info(f"✓ M3 classification dataset saved")
    logger.info(f"  - Shape: {m3_df.shape}")
except Exception as e:
    logger.error(f"Error creating m3 classification dataset: {e}")




# 7. Longitudinal Progression Datasets: M3, M2, M1 versions
logger.info("Creating m1 worsening dataset...")
try:
    worsening_df = longitudinal_df.copy()    
    worsening_df['Duration'] = worsening_df.groupby('RID')['SCANDATE'].transform(
        lambda x: (x.max() - x.min()).days
    )
    
    worsening_m1_df = worsening_df.drop(columns=['ptau_ab_ratio_csf', 'PLASMA_NFL', 'embedding_scalar', 'NORM_WMH', 'NORM_GRAY'])

    # Save complete worsening dataset
    worsening_m1_df = worsening_m1_df.sort_values(['RID', 'SCANDATE'])
    worsening_m1_df.to_csv(os.path.join(base_path, "worsening_m1.csv"), index=False)
    logger.info(f"✓ M1 worsening progression dataset saved")
    logger.info(f"  - Shape: {worsening_m1_df.shape}")
    logger.info(f"  - Unique patients: {worsening_m1_df['RID'].nunique()}")

except Exception as e:
    logger.error(f"Error creating m1 worsening dataset: {e}")

logger.info("Creating m2 worsening dataset...")
try:
    
    worsening_m2_df = worsening_df.drop(columns=['embedding_scalar', 'NORM_WMH', 'NORM_GRAY'])
    worsening_m2_df = worsening_m2_df.sort_values(['RID', 'SCANDATE'])
    worsening_m2_df.to_csv(os.path.join(base_path, "worsening_m2.csv"), index=False)
    logger.info(f"✓ M2 worsening progression dataset saved")
    logger.info(f"  - Shape: {worsening_m2_df.shape}")
    logger.info(f"  - Unique patients: {worsening_m2_df['RID'].nunique()}")
except Exception as e:
    logger.error(f"Error creating m2 worsening dataset: {e}")

logger.info("Creating m3 worsening dataset...")
try:
    worsening_df = worsening_df.sort_values(['RID', 'SCANDATE'])
    # Save the m3 worsening dataset
    worsening_df.to_csv(os.path.join(base_path, "worsening_m3.csv"), index=False)
    logger.info(f"✓ M3 worsening dataset saved")
    logger.info(f"  - Shape: {worsening_df.shape}")
    logger.info(f"  - Unique patients: {worsening_df['RID'].nunique()}")
    
except Exception as e:
    logger.error(f"Error creating m3 worsening dataset: {e}")

logger.info("All datasets successfully created and saved.")

# Print descriptive statistics for all classification datasets
logger.info("===========================")
logger.info("CLASSIFICATION DATASET STATISTICS")
logger.info("===========================")

# M1 classification dataset
logger.info("SMALL CLASSIFICATION DATASET STATISTICS:")
try:
    m1_df = pd.read_csv(os.path.join(base_path, "classification_m1.csv"))
    logger.info(f"Shape: {m1_df.shape}")
    logger.info(f"Unique patients: {m1_df['RID'].nunique()}")
    logger.info(f"Class distribution: {m1_df['MCH_pos'].value_counts().to_dict()}")
    logger.debug("Numeric Features Statistics:\n%s", m1_df.describe())
        
except Exception as e:
    logger.error(f"Error reading m1 classification dataset: {e}")

# M2 classification dataset
logger.info("MEDIUM CLASSIFICATION DATASET STATISTICS:")
try:
    m2_df = pd.read_csv(os.path.join(base_path, "classification_m2.csv"))
    logger.info(f"Shape: {m2_df.shape}")
    logger.info(f"Unique patients: {m2_df['RID'].nunique()}")
    logger.info(f"Class distribution: {m2_df['MCH_pos'].value_counts().to_dict()}")
    logger.debug("Numeric Features Statistics:\n%s", m2_df.describe())
        
except Exception as e:
    logger.error(f"Error reading m2 classification dataset: {e}")

# M3 classification dataset
logger.info("LARGE CLASSIFICATION DATASET STATISTICS:")
try:
    m3_df = pd.read_csv(os.path.join(base_path, "classification_m3.csv"))
    logger.info(f"Shape: {m3_df.shape}")
    logger.info(f"Unique patients: {m3_df['RID'].nunique()}")
    logger.info(f"Class distribution: {m3_df['MCH_pos'].value_counts().to_dict()}")
    
    # Get basic statistics without the vector columns to keep it readable
    non_vec_cols = [col for col in m3_df.columns if not col.startswith('Vec_')]
    logger.debug("Numeric Features Statistics (excluding vector columns):\n%s", m3_df[non_vec_cols].describe())
    
except Exception as e:
    logger.error(f"Error reading m3 classification dataset: {e}")

# ===========================
logger.info("===========================")
logger.info("WORSENING DATASET STATISTICS")
logger.info("===========================")

for size in ["m1", "m2", "m3"]:
    file_name = f"worsening_{size}.csv"
    logger.info(f"{size.upper()} WORSENING DATASET STATISTICS:")
    try:
        w_df = pd.read_csv(os.path.join(base_path, file_name))
        logger.info(f"Shape: {w_df.shape}")
        logger.info(f"Unique patients: {w_df['RID'].nunique()}")
        logger.info(f"Total scans: {len(w_df)}")
        scans_per_pt = w_df.groupby('RID').size()
        logger.info(f"Mean scans per patient: {scans_per_pt.mean():.2f} (median {scans_per_pt.median():.0f})")
        
       
            
    except Exception as e:
        logger.error(f"Error reading {size} worsening dataset: {e}")

# ----------------------------------------------------------------------------
# Final runtime
# ----------------------------------------------------------------------------
elapsed_seconds = perf_counter() - script_start_time
logger.info(f"Total runtime: {elapsed_seconds:.2f} seconds")

