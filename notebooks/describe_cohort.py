
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Optional, Dict, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

def _filter_to_baseline(
    df: pd.DataFrame,
    id_col: str = 'RID',
    viscode_col: str = 'VISCODE',
    date_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reduce the dataframe to a single baseline row per participant.

    Selection order:
    1) If `viscode_col` exists and has entries equal to 'bl' (case-insensitive),
       keep those rows and collapse to one row per `id_col` (earliest date if
       date columns are available; otherwise first occurrence).
    2) Else, use the earliest available date from the first matching column in
       `date_cols` (defaults to common names like 'SCANDATE', 'EXAMDATE', etc.).
    3) Fallback to the first occurrence per `id_col`.

    If `id_col` is missing, the original dataframe is returned unchanged.
    """

    if id_col not in df.columns:
        return df

    # Default set of candidate date columns
    if date_cols is None:
        date_cols = [
            'SCANDATE', 'EXAMDATE', 'EXAM_DATE', 'SCAN_DATE',
            'VISITDATE', 'VISIT_DATE', 'EXAMDATE1'
        ]

    # 1) Prefer VISCODE == 'bl' when available
    if viscode_col in df.columns:
        viscode_series = df[viscode_col].astype(str).str.lower()
        bl_mask = viscode_series.eq('bl')
        if bl_mask.any():
            df_bl = df.loc[bl_mask].copy()

            # If multiple baseline rows exist per participant, try to pick the earliest by date
            existing_date_cols = [c for c in date_cols if c in df_bl.columns]
            if existing_date_cols:
                # Use the first usable date column
                for dc in existing_date_cols:
                    tmp = df_bl.copy()
                    tmp['_baseline_dt'] = pd.to_datetime(tmp[dc], errors='coerce')
                    tmp = tmp.sort_values(['_baseline_dt', id_col])
                    # Drop rows with NaT dates to avoid selecting non-dated rows first
                    tmp_nonan = tmp.dropna(subset=['_baseline_dt'])
                    if not tmp_nonan.empty:
                        out = (
                            tmp_nonan
                            .sort_values([id_col, '_baseline_dt'])
                            .drop_duplicates(subset=[id_col], keep='first')
                            .drop(columns=['_baseline_dt'])
                        )
                        return out
            # Fallback: first baseline occurrence per participant
            return df_bl.drop_duplicates(subset=[id_col], keep='first')

    # 2) Else, pick earliest by available date column
    existing_date_cols = [c for c in date_cols if c in df.columns]
    for dc in existing_date_cols:
        tmp = df.copy()
        tmp['_baseline_dt'] = pd.to_datetime(tmp[dc], errors='coerce')
        tmp_nonan = tmp.dropna(subset=['_baseline_dt'])
        if not tmp_nonan.empty:
            out = (
                tmp_nonan
                .sort_values([id_col, '_baseline_dt'])
                .drop_duplicates(subset=[id_col], keep='first')
                .drop(columns=['_baseline_dt'])
            )
            return out

    # 3) Final fallback: first occurrence per participant
    return df.drop_duplicates(subset=[id_col], keep='first')

class CohortDescriptor:
    """
    A class to generate comprehensive cohort characteristics tables for medical research.
    
    Features:
    - Automatic detection of continuous vs categorical variables
    - Proper handling of missing data
    - Group comparisons with statistical tests
    - Multiple output formats (console, HTML, LaTeX)
    - Publication-ready formatting
    """
    
    def __init__(self, df: pd.DataFrame, group_col: Optional[str] = None, 
                 exclude_cols: Optional[List[str]] = None,
                 full_df: Optional[pd.DataFrame] = None,
                 id_col: str = 'RID',
                 date_col: str = 'SCANDATE'):
        """
        Initialize the CohortDescriptor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset containing cohort data (typically baseline only)
        group_col : str, optional
            Column name for grouping (e.g., 'treatment_group', 'case_control')
        exclude_cols : list, optional
            List of column names to exclude from the analysis
        full_df : pd.DataFrame, optional
            Full dataset with all visits (for longitudinal analysis). If None, uses df.
        id_col : str
            Participant identifier column (default 'RID')
        date_col : str
            Date column for longitudinal analysis (default 'SCANDATE')
        """
        self.df = df.copy()
        self.group_col = group_col
        self.exclude_cols = exclude_cols or []
        self.full_df = full_df.copy() if full_df is not None else df.copy()
        self.id_col = id_col
        self.date_col = date_col
        
        # Keep a copy of the original df with all columns for internal use (e.g., longitudinal stats)
        self.df_with_all_cols = df.copy()
        
        # Remove excluded columns from baseline df (but keep them in full_df for longitudinal analysis)
        # Also keep id_col and date_col even if excluded, as they're needed for longitudinal analysis
        cols_to_remove = [col for col in self.exclude_cols 
                         if col in self.df.columns 
                         and col != self.id_col  # Keep id_col for internal use
                         and col != self.date_col]  # Keep date_col for internal use
        if cols_to_remove:
            self.df = self.df.drop(columns=cols_to_remove)
        
        # Note: We don't remove excluded columns from full_df because we need them for longitudinal analysis
        # (e.g., SCANDATE, RID are needed even if excluded from baseline table)
        
        # Prepare data
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and categorize the data."""
        # Identify variable types
        self.continuous_vars = []
        self.categorical_vars = []
        
        for col in self.df.columns:
            if col == self.group_col:
                continue
                
            # Check if column is numeric and has enough unique values
            if (pd.api.types.is_numeric_dtype(self.df[col]) and 
                self.df[col].nunique() > 10):
                self.continuous_vars.append(col)
            else:
                self.categorical_vars.append(col)
    
    def _format_continuous_stats(self, series: pd.Series) -> str:
        """Format continuous variable statistics as mean ± SD (n)."""
        n_valid = series.count()
        if n_valid == 0:
            return "No data"
        
        mean_val = series.mean()
        std_val = series.std()
        
        # Determine appropriate decimal places based on magnitude
        if mean_val < 1:
            decimals = 3
        elif mean_val < 10:
            decimals = 2
        else:
            decimals = 1
            
        return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f} (n={n_valid})"
    
    def _format_categorical_stats(self, series: pd.Series, category: str = None) -> str:
        """Format categorical variable statistics as n (%)."""
        total_n = len(series)
        valid_n = series.count()
        
        if category is None:
            # For binary variables, show the positive case
            if series.dtype == bool:
                count = series.sum()
            else:
                # Show most frequent category
                count = series.value_counts().iloc[0] if valid_n > 0 else 0
                category = series.value_counts().index[0] if valid_n > 0 else "N/A"
        else:
            count = (series == category).sum()
        
        percentage = (count / total_n * 100) if total_n > 0 else 0
        return f"{count} ({percentage:.1f}%)"
    
    def _perform_statistical_test(self, var_name: str, is_continuous: bool) -> Tuple[float, str]:
        """Perform appropriate statistical test for group comparison."""
        if self.group_col is None:
            return np.nan, "N/A"
        
        groups = self.df[self.group_col].unique()
        if len(groups) < 2:
            return np.nan, "N/A"
        
        group_data = [self.df[self.df[self.group_col] == group][var_name].dropna() 
                     for group in groups]
        
        # Remove empty groups
        group_data = [data for data in group_data if len(data) > 0]
        
        if len(group_data) < 2:
            return np.nan, "N/A"
        
        try:
            if is_continuous:
                # Use t-test for 2 groups, ANOVA for >2 groups
                if len(group_data) == 2:
                    statistic, p_value = stats.ttest_ind(group_data[0], group_data[1])
                    test_name = "t-test"
                else:
                    statistic, p_value = stats.f_oneway(*group_data)
                    test_name = "ANOVA"
            else:
                # Chi-square test for categorical variables
                # Create contingency table
                contingency = pd.crosstab(self.df[var_name], self.df[self.group_col])
                statistic, p_value, _, _ = stats.chi2_contingency(contingency)
                test_name = "X²"
                
            return p_value, test_name
        except:
            return np.nan, "N/A"
    
    def generate_table(self, include_stats: bool = True, 
                      show_missing: bool = True) -> pd.DataFrame:
        """
        Generate the cohort characteristics table.
        
        Parameters:
        -----------
        include_stats : bool
            Whether to include statistical tests for group comparisons
        show_missing : bool
            Whether to show missing data information
            
        Returns:
        --------
        pd.DataFrame
            The formatted cohort characteristics table
        """
        
        # Initialize results
        results = []
        
        # Determine groups - use df_with_all_cols to ensure consistency with longitudinal stats
        if self.group_col:
            if self.group_col in self.df_with_all_cols.columns:
                groups = sorted(self.df_with_all_cols[self.group_col].unique())
                group_sizes = self.df_with_all_cols[self.group_col].value_counts().sort_index()
            elif self.group_col in self.df.columns:
                groups = sorted(self.df[self.group_col].unique())
                group_sizes = self.df[self.group_col].value_counts().sort_index()
            else:
                groups = ['Overall']
                group_sizes = pd.Series([len(self.df)])
        else:
            groups = ['Overall']
            group_sizes = pd.Series([len(self.df)])
        
        # Process continuous variables
        if self.continuous_vars:
            results.append({
                'Variable': 'CONTINUOUS VARIABLES',
                'Type': '',
                **{f'{group}' if self.group_col else 'Overall': '' for group in groups},
                'P-value': '' if include_stats and self.group_col else None,
                'Test': '' if include_stats and self.group_col else None
            })
            
            for var in self.continuous_vars:
                row = {
                    'Variable': self._format_variable_name(var),
                    'Type': 'Mean ± SD (n)'
                }
                
                if self.group_col:
                    for group in groups:
                        group_data = self.df[self.df[self.group_col] == group][var]
                        row[f'{group}'] = self._format_continuous_stats(group_data)
                else:
                    row['Overall'] = self._format_continuous_stats(self.df[var])
                
                # Add statistical test
                if include_stats and self.group_col:
                    p_val, test_name = self._perform_statistical_test(var, True)
                    row['P-value'] = self._format_p_value(p_val)
                    row['Test'] = test_name
                
                results.append(row)
        
        # Process categorical variables
        if self.categorical_vars:
            results.append({
                'Variable': 'CATEGORICAL VARIABLES',
                'Type': '',
                **{f'{group}' if self.group_col else 'Overall': '' for group in groups},
                'P-value': '' if include_stats and self.group_col else None,
                'Test': '' if include_stats and self.group_col else None
            })
            
            for var in self.categorical_vars:
                unique_vals = self.df[var].value_counts()
                
                # Main variable row
                row = {
                    'Variable': self._format_variable_name(var),
                    'Type': 'n (%)'
                }
                
                if self.group_col:
                    for group in groups:
                        group_data = self.df[self.df[self.group_col] == group][var]
                        row[f'{group}'] = f"n={len(group_data)}"
                else:
                    row['Overall'] = f"n={len(self.df[var])}"
                
                # Add statistical test
                if include_stats and self.group_col:
                    p_val, test_name = self._perform_statistical_test(var, False)
                    row['P-value'] = self._format_p_value(p_val)
                    row['Test'] = test_name
                else:
                    row['P-value'] = None
                    row['Test'] = None
                
                results.append(row)
                
                # Add subcategories
                for category in unique_vals.index[:5]:  # Show top 5 categories
                    subrow = {
                        'Variable': f"  {category}",
                        'Type': ''
                    }
                    
                    if self.group_col:
                        for group in groups:
                            group_data = self.df[self.df[self.group_col] == group][var]
                            subrow[f'{group}'] = self._format_categorical_stats(group_data, category)
                    else:
                        subrow['Overall'] = self._format_categorical_stats(self.df[var], category)
                    
                    subrow['P-value'] = None
                    subrow['Test'] = None
                    results.append(subrow)
        
        # Add longitudinal MCH statistics
        longitudinal_stats = self._compute_longitudinal_mch_stats()
        if longitudinal_stats:
            results.append({
                'Variable': 'LONGITUDINAL MCH',
                'Type': '',
                **{f'{group}' if self.group_col else 'Overall': '' for group in groups},
                'P-value': '' if include_stats and self.group_col else None,
                'Test': '' if include_stats and self.group_col else None
            })
            
            # 1. Patients with longitudinal MCH assessments, n(%)
            row1 = {
                'Variable': 'Patients with longitudinal MCH assessments, n(%)',
                'Type': 'n (%)'
            }
            if self.group_col:
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        row1[f'{group}'] = f"{stats['n_with_longitudinal']} ({stats['pct_with_longitudinal']:.1f}%)"
                    else:
                        row1[f'{group}'] = "N/A"
            else:
                if 'Overall' in longitudinal_stats:
                    stats = longitudinal_stats['Overall']
                    row1['Overall'] = f"{stats['n_with_longitudinal']} ({stats['pct_with_longitudinal']:.1f}%)"
                else:
                    row1['Overall'] = "N/A"
            
            # Statistical test for longitudinal assessments (chi-square or Fisher's exact)
            if include_stats and self.group_col and len(groups) >= 2:
                # Create contingency table - include all groups
                contingency_data = []
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        n_with = int(stats['n_with_longitudinal'])
                        n_without = int(stats['n_total'] - stats['n_with_longitudinal'])
                        contingency_data.append([n_with, n_without])
                    else:
                        # If group not found, use zeros
                        contingency_data.append([0, 0])
                
                # Ensure we have at least 2 groups with data
                if len(contingency_data) >= 2 and any(sum(row) > 0 for row in contingency_data):
                    try:
                        contingency = pd.DataFrame(contingency_data, index=groups, columns=['With', 'Without'])
                        
                        # For 2x2 tables, check if we should use Fisher's exact test
                        if len(groups) == 2:
                            # Check expected frequencies for chi-square assumption
                            chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency)
                            
                            # Use Fisher's exact if any expected frequency < 5 (conservative approach)
                            if (expected < 5).any().any():
                                from scipy.stats import fisher_exact
                                oddsratio, p_val = fisher_exact(contingency.values)
                                row1['P-value'] = self._format_p_value(p_val)
                                row1['Test'] = "Fisher"
                            else:
                                row1['P-value'] = self._format_p_value(p_val_chi2)
                                row1['Test'] = "X²"
                        else:
                            # For more than 2 groups, use chi-square
                            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency)
                            row1['P-value'] = self._format_p_value(p_val)
                            row1['Test'] = "X²"
                    except Exception as e:
                        # Fallback to Fisher's exact for 2x2 if chi-square fails
                        if len(groups) == 2:
                            try:
                                from scipy.stats import fisher_exact
                                contingency_2x2 = pd.DataFrame(contingency_data, columns=['With', 'Without'])
                                oddsratio, p_val = fisher_exact(contingency_2x2.values)
                                row1['P-value'] = self._format_p_value(p_val)
                                row1['Test'] = "Fisher"
                            except Exception as e2:
                                row1['P-value'] = "N/A"
                                row1['Test'] = "N/A"
                        else:
                            row1['P-value'] = "N/A"
                            row1['Test'] = "N/A"
                else:
                    row1['P-value'] = "N/A"
                    row1['Test'] = "N/A"
            else:
                row1['P-value'] = None
                row1['Test'] = None
            
            results.append(row1)
            
            # 2. Number of longitudinal MCH assessments, mean ± sd
            row2 = {
                'Variable': 'Number of longitudinal MCH assessments, mean ± sd',
                'Type': 'Mean ± SD'
            }
            if self.group_col:
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        row2[f'{group}'] = f"{stats['mean_assessments']:.2f} ± {stats['sd_assessments']:.2f}"
                    else:
                        row2[f'{group}'] = "N/A"
            else:
                if 'Overall' in longitudinal_stats:
                    stats = longitudinal_stats['Overall']
                    row2['Overall'] = f"{stats['mean_assessments']:.2f} ± {stats['sd_assessments']:.2f}"
                else:
                    row2['Overall'] = "N/A"
            
            # Statistical test for number of assessments (t-test or ANOVA)
            if include_stats and self.group_col and len(groups) >= 2:
                group_data_list = []
                for group in groups:
                    if group in longitudinal_stats:
                        stats_dict = longitudinal_stats[group]
                        assessments = stats_dict.get('longitudinal_assessments', [])
                        
                        # Ensure we have a list
                        if not isinstance(assessments, list):
                            if hasattr(assessments, 'tolist'):
                                assessments = assessments.tolist()
                            elif assessments is not None:
                                assessments = list(assessments)
                            else:
                                assessments = []
                        
                        # Filter out any None or NaN values and ensure numeric
                        clean_data = []
                        for x in assessments:
                            if x is not None:
                                try:
                                    x_val = float(x)
                                    if not (np.isnan(x_val) or np.isinf(x_val)):
                                        clean_data.append(x_val)
                                except (ValueError, TypeError):
                                    continue
                        
                        # Only add if we have valid data
                        if len(clean_data) > 0:
                            group_data_list.append(clean_data)
                
                if len(group_data_list) >= 2:
                    try:
                        if len(group_data_list) == 2:
                            # Use Welch's t-test (unequal variances) for more robust comparison
                            statistic, p_val = stats.ttest_ind(group_data_list[0], group_data_list[1], equal_var=False)
                            row2['P-value'] = self._format_p_value(p_val)
                            row2['Test'] = "t-test"
                        else:
                            statistic, p_val = stats.f_oneway(*group_data_list)
                            row2['P-value'] = self._format_p_value(p_val)
                            row2['Test'] = "ANOVA"
                    except Exception as e:
                        row2['P-value'] = "N/A"
                        row2['Test'] = "N/A"
                else:
                    row2['P-value'] = "N/A"
                    row2['Test'] = "N/A"
            else:
                row2['P-value'] = None
                row2['Test'] = None
            
            results.append(row2)
            
            # 3. Duration of longitudinal follow-up from baseline (days), mean ± sd
            row3 = {
                'Variable': 'Duration of longitudinal follow-up from baseline (days), mean ± sd',
                'Type': 'Mean ± SD'
            }
            if self.group_col:
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        row3[f'{group}'] = f"{stats['mean_followup_days']:.1f} ± {stats['sd_followup_days']:.1f}"
                    else:
                        row3[f'{group}'] = "N/A"
            else:
                if 'Overall' in longitudinal_stats:
                    stats = longitudinal_stats['Overall']
                    row3['Overall'] = f"{stats['mean_followup_days']:.1f} ± {stats['sd_followup_days']:.1f}"
                else:
                    row3['Overall'] = "N/A"
            
            # Statistical test for follow-up duration (t-test or ANOVA)
            if include_stats and self.group_col and len(groups) >= 2:
                group_data_list = []
                for group in groups:
                    if group in longitudinal_stats:
                        stats_dict = longitudinal_stats[group]
                        durations = stats_dict.get('followup_durations', [])
                        
                        # Ensure we have a list
                        if not isinstance(durations, list):
                            if hasattr(durations, 'tolist'):
                                durations = durations.tolist()
                            elif durations is not None:
                                durations = list(durations)
                            else:
                                durations = []
                        
                        # Filter out any None or NaN values and ensure numeric
                        clean_data = []
                        for x in durations:
                            if x is not None:
                                try:
                                    x_val = float(x)
                                    if not (np.isnan(x_val) or np.isinf(x_val)):
                                        clean_data.append(x_val)
                                except (ValueError, TypeError):
                                    continue
                        
                        # Only add if we have valid data
                        if len(clean_data) > 0:
                            group_data_list.append(clean_data)
                
                if len(group_data_list) >= 2:
                    try:
                        if len(group_data_list) == 2:
                            # Use Welch's t-test (unequal variances) for more robust comparison
                            statistic, p_val = stats.ttest_ind(group_data_list[0], group_data_list[1], equal_var=False)
                            row3['P-value'] = self._format_p_value(p_val)
                            row3['Test'] = "t-test"
                        else:
                            statistic, p_val = stats.f_oneway(*group_data_list)
                            row3['P-value'] = self._format_p_value(p_val)
                            row3['Test'] = "ANOVA"
                    except Exception as e:
                        row3['P-value'] = "N/A"
                        row3['Test'] = "N/A"
                else:
                    row3['P-value'] = "N/A"
                    row3['Test'] = "N/A"
            else:
                row3['P-value'] = None
                row3['Test'] = None
            
            results.append(row3)
            
            # 4. Incidence of MCH after baseline, n(%)
            row4 = {
                'Variable': 'Incidence of MCH after baseline, n(%)',
                'Type': 'n (%)'
            }
            if self.group_col:
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        row4[f'{group}'] = f"{stats['n_mch_after_baseline']} ({stats['pct_mch_after_baseline']:.1f}%)"
                    else:
                        row4[f'{group}'] = "N/A"
            else:
                if 'Overall' in longitudinal_stats:
                    stats = longitudinal_stats['Overall']
                    row4['Overall'] = f"{stats['n_mch_after_baseline']} ({stats['pct_mch_after_baseline']:.1f}%)"
                else:
                    row4['Overall'] = "N/A"
            
            # Statistical test for MCH incidence (chi-square or Fisher's exact)
            if include_stats and self.group_col and len(groups) >= 2:
                # Create contingency table: MCH after baseline vs not (only among those with longitudinal data)
                contingency_data = []
                for group in groups:
                    if group in longitudinal_stats:
                        stats = longitudinal_stats[group]
                        n_with_longitudinal = int(stats['n_with_longitudinal'])
                        n_mch_after = int(stats['n_mch_after_baseline'])
                        n_no_mch = n_with_longitudinal - n_mch_after
                        contingency_data.append([n_mch_after, n_no_mch])
                    else:
                        contingency_data.append([0, 0])
                
                if len(contingency_data) >= 2 and any(sum(row) > 0 for row in contingency_data):
                    try:
                        contingency = pd.DataFrame(contingency_data, index=groups, columns=['MCH_after', 'No_MCH_after'])
                        
                        # For 2x2 tables, check if we should use Fisher's exact test
                        if len(groups) == 2:
                            # Check expected frequencies for chi-square assumption
                            chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency)
                            
                            # Use Fisher's exact if any expected frequency < 5 (conservative approach)
                            if (expected < 5).any().any():
                                from scipy.stats import fisher_exact
                                oddsratio, p_val = fisher_exact(contingency.values)
                                row4['P-value'] = self._format_p_value(p_val)
                                row4['Test'] = "Fisher"
                            else:
                                row4['P-value'] = self._format_p_value(p_val_chi2)
                                row4['Test'] = "X²"
                        else:
                            # For more than 2 groups, use chi-square
                            chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency)
                            row4['P-value'] = self._format_p_value(p_val)
                            row4['Test'] = "X²"
                    except Exception as e:
                        # Fallback to Fisher's exact for 2x2 if chi-square fails
                        if len(groups) == 2:
                            try:
                                from scipy.stats import fisher_exact
                                contingency_2x2 = pd.DataFrame(contingency_data, columns=['MCH_after', 'No_MCH_after'])
                                oddsratio, p_val = fisher_exact(contingency_2x2.values)
                                row4['P-value'] = self._format_p_value(p_val)
                                row4['Test'] = "Fisher"
                            except Exception as e2:
                                row4['P-value'] = "N/A"
                                row4['Test'] = "N/A"
                        else:
                            row4['P-value'] = "N/A"
                            row4['Test'] = "N/A"
                else:
                    row4['P-value'] = "N/A"
                    row4['Test'] = "N/A"
            else:
                row4['P-value'] = None
                row4['Test'] = None
            
            results.append(row4)
        
        # Create DataFrame
        table_df = pd.DataFrame(results)
        
        # Clean up columns
        if not include_stats or not self.group_col:
            table_df = table_df.drop(columns=['P-value', 'Test'], errors='ignore')
        
        return table_df
    
    def _format_variable_name(self, var_name: str) -> str:
        """Format variable names for display."""
        # Replace underscores with spaces and title case
        formatted = var_name.replace('_', ' ').title()
        
        # Handle common medical abbreviations (ASCII-safe for Windows console)
        replacements = {
            'Csf': 'CSF',
            'Mri': 'MRI',
            'Dti': 'DTI',
            'Ptau': 'pTau',
            'Abeta': 'Abeta',  # Changed from 'Aβ' to 'Abeta' for ASCII compatibility
            'Med': 'Medication',
            'Mh': 'Medical History',
            'Dx': 'Diagnosis'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _format_p_value(self, p_val: float) -> str:
        """Format p-values for display."""
        if pd.isna(p_val):
            return "N/A"
        elif p_val < 0.001:
            return "<0.001"
        elif p_val < 0.01:
            return f"{p_val:.3f}"
        else:
            return f"{p_val:.2f}"
    
    def _compute_longitudinal_mch_stats(self) -> Optional[Dict]:
        """
        Compute longitudinal MCH statistics.
        
        Returns:
        --------
        dict with keys:
        - 'n_with_longitudinal': number of patients with longitudinal assessments
        - 'pct_with_longitudinal': percentage with longitudinal assessments
        - 'mean_assessments': mean number of longitudinal assessments
        - 'sd_assessments': SD of number of longitudinal assessments
        - 'mean_followup_days': mean follow-up duration in days
        - 'sd_followup_days': SD of follow-up duration in days
        - 'n_mch_after_baseline': number with MCH after baseline
        - 'pct_mch_after_baseline': percentage with MCH after baseline
        """
        # Check if required columns exist
        if self.id_col not in self.full_df.columns:
            print(f"Warning: {self.id_col} not found in full dataset. Skipping longitudinal MCH stats.")
            return None
        
        if self.date_col not in self.full_df.columns:
            print(f"Warning: {self.date_col} not found in full dataset. Skipping longitudinal MCH stats.")
            return None
        
        # Check for MCH_pos column
        mch_col = None
        for col in ['MCH_pos', 'MCH_pos_flag', 'MCH_POS']:
            if col in self.full_df.columns:
                mch_col = col
                break
        
        if mch_col is None:
            print("Warning: MCH_pos column not found in full dataset. Skipping longitudinal MCH stats.")
            return None
        
        # Get baseline RIDs from self.df_with_all_cols (which has all columns including id_col)
        # This ensures we can always get RIDs even if they were "excluded" from display
        if self.id_col in self.df_with_all_cols.columns:
            baseline_rids = set(self.df_with_all_cols[self.id_col].unique())
        elif self.id_col in self.df.columns:
            baseline_rids = set(self.df[self.id_col].unique())
        elif self.id_col in self.full_df.columns:
            baseline_rids = set(self.full_df[self.id_col].unique())
        else:
            print(f"Warning: {self.id_col} not found in any dataset. Skipping longitudinal MCH stats.")
            return None
        
        if len(baseline_rids) == 0:
            print("Warning: No baseline RIDs found. Skipping longitudinal MCH stats.")
            return None
        
        # Filter full dataset to baseline participants
        full_subset = self.full_df[self.full_df[self.id_col].isin(baseline_rids)].copy()
        
        if len(full_subset) == 0:
            return None
        
        # Convert date column to datetime
        full_subset[self.date_col] = pd.to_datetime(full_subset[self.date_col], errors='coerce')
        full_subset = full_subset.dropna(subset=[self.date_col])
        
        # Sort by RID and date
        full_subset = full_subset.sort_values([self.id_col, self.date_col])
        
        # Compute statistics per group
        stats_by_group = {}
        
        # Get group assignments - try df_with_all_cols first, then df, then full_df
        if self.group_col:
            if self.group_col in self.df_with_all_cols.columns and self.id_col in self.df_with_all_cols.columns:
                groups = sorted(self.df_with_all_cols[self.group_col].unique())
                baseline_groups = self.df_with_all_cols[[self.id_col, self.group_col]].set_index(self.id_col)[self.group_col].to_dict()
            elif self.group_col in self.df.columns and self.id_col in self.df.columns:
                groups = sorted(self.df[self.group_col].unique())
                baseline_groups = self.df[[self.id_col, self.group_col]].set_index(self.id_col)[self.group_col].to_dict()
            else:
                groups = ['Overall']
                baseline_groups = {rid: 'Overall' for rid in baseline_rids}
        else:
            groups = ['Overall']
            baseline_groups = {rid: 'Overall' for rid in baseline_rids}
        
        for group in groups:
            # Get RIDs for this group
            if group == 'Overall':
                group_rids = baseline_rids
            else:
                # Try to get group RIDs from df_with_all_cols first, then df
                if self.group_col in self.df_with_all_cols.columns and self.id_col in self.df_with_all_cols.columns:
                    group_rids = set(self.df_with_all_cols[self.df_with_all_cols[self.group_col] == group][self.id_col].unique())
                elif self.group_col in self.df.columns and self.id_col in self.df.columns:
                    group_rids = set(self.df[self.df[self.group_col] == group][self.id_col].unique())
                else:
                    # Fallback: use baseline_groups dict
                    group_rids = {rid for rid, g in baseline_groups.items() if g == group}
            
            if len(group_rids) == 0:
                continue
            
            # Filter to this group's data
            group_data = full_subset[full_subset[self.id_col].isin(group_rids)].copy()
            
            if len(group_data) == 0:
                continue
            
            # Compute per-participant statistics
            participant_stats = []
            
            for rid in group_rids:
                rid_data = group_data[group_data[self.id_col] == rid].copy()
                
                if len(rid_data) == 0:
                    continue
                
                # Sort by date
                rid_data = rid_data.sort_values(self.date_col)
                
                # Number of visits
                n_visits = len(rid_data)
                n_longitudinal = max(0, n_visits - 1)  # Number of follow-up assessments
                
                # Follow-up duration (days from baseline to last visit)
                if n_visits > 1:
                    baseline_date = rid_data.iloc[0][self.date_col]
                    last_date = rid_data.iloc[-1][self.date_col]
                    followup_days = (last_date - baseline_date).days
                else:
                    followup_days = 0
                
                # MCH status at baseline and after baseline
                baseline_mch = pd.to_numeric(rid_data.iloc[0][mch_col], errors='coerce')
                baseline_mch = 0 if pd.isna(baseline_mch) or baseline_mch == 0 else 1
                
                # Check if MCH appears after baseline
                mch_after_baseline = 0
                if n_visits > 1:
                    followup_mch = pd.to_numeric(rid_data.iloc[1:][mch_col], errors='coerce').fillna(0)
                    mch_after_baseline = 1 if (followup_mch > 0).any() else 0
                
                participant_stats.append({
                    'rid': rid,
                    'n_visits': n_visits,
                    'n_longitudinal': n_longitudinal,
                    'followup_days': followup_days,
                    'baseline_mch': baseline_mch,
                    'mch_after_baseline': mch_after_baseline,
                    'has_longitudinal': n_visits > 1
                })
            
            if len(participant_stats) == 0:
                continue
            
            stats_df = pd.DataFrame(participant_stats)
            
            # Compute aggregate statistics
            n_total = len(stats_df)
            n_with_longitudinal = stats_df['has_longitudinal'].sum()
            pct_with_longitudinal = (n_with_longitudinal / n_total * 100) if n_total > 0 else 0
            
            # Mean and SD of longitudinal assessments (only for those with follow-up)
            longitudinal_assessments = stats_df[stats_df['has_longitudinal']]['n_longitudinal']
            mean_assessments = longitudinal_assessments.mean() if len(longitudinal_assessments) > 0 else 0
            sd_assessments = longitudinal_assessments.std() if len(longitudinal_assessments) > 0 else 0
            
            # Mean and SD of follow-up duration (only for those with follow-up)
            followup_durations = stats_df[stats_df['has_longitudinal']]['followup_days']
            mean_followup_days = followup_durations.mean() if len(followup_durations) > 0 else 0
            sd_followup_days = followup_durations.std() if len(followup_durations) > 0 else 0
            
            # Incidence of MCH after baseline (among those with longitudinal data)
            n_mch_after = stats_df[stats_df['has_longitudinal']]['mch_after_baseline'].sum()
            pct_mch_after = (n_mch_after / n_with_longitudinal * 100) if n_with_longitudinal > 0 else 0
            
            # Convert Series to lists, handling empty cases
            # Ensure we convert to Python list and handle NaN values
            # Drop NaN values from Series before converting to list
            if len(longitudinal_assessments) > 0:
                assessments_clean = longitudinal_assessments.dropna()
                assessments_list = [float(x) for x in assessments_clean.tolist()]
            else:
                assessments_list = []
            
            if len(followup_durations) > 0:
                durations_clean = followup_durations.dropna()
                durations_list = [float(x) for x in durations_clean.tolist()]
            else:
                durations_list = []
            
            if n_with_longitudinal > 0:
                mch_after_series = stats_df[stats_df['has_longitudinal']]['mch_after_baseline']
                mch_after_clean = mch_after_series.dropna()
                mch_after_list = [int(x) for x in mch_after_clean.tolist()]
            else:
                mch_after_list = []
            
            stats_by_group[group] = {
                'n_total': n_total,
                'n_with_longitudinal': n_with_longitudinal,
                'pct_with_longitudinal': pct_with_longitudinal,
                'mean_assessments': mean_assessments,
                'sd_assessments': sd_assessments,
                'mean_followup_days': mean_followup_days,
                'sd_followup_days': sd_followup_days,
                'n_mch_after_baseline': n_mch_after,
                'pct_mch_after_baseline': pct_mch_after,
                'longitudinal_assessments': assessments_list,
                'followup_durations': durations_list,
                'mch_after_baseline': mch_after_list
            }
        
        return stats_by_group if stats_by_group else None
    
    def display_table(self, include_stats: bool = True, 
                     show_missing: bool = True,
                     style: str = 'publication') -> None:
        """
        Display the cohort characteristics table with formatting.
        
        Parameters:
        -----------
        include_stats : bool
            Whether to include statistical tests
        show_missing : bool
            Whether to show missing data information
        style : str
            Display style ('publication', 'simple', 'fancy')
        """
        
        table_df = self.generate_table(include_stats, show_missing)
        
        print("\n" + "="*80)
        print("COHORT CHARACTERISTICS TABLE")
        print("="*80)
        
        if self.group_col:
            group_sizes = self.df[self.group_col].value_counts().sort_index()
            print(f"\nGroup sizes:")
            for group, size in group_sizes.items():
                print(f"  {group}: n={size}")
        else:
            print(f"\nTotal sample size: n={len(self.df)}")
        
        print(f"\nVariables analyzed:")
        print(f"  Continuous: {len(self.continuous_vars)}")
        print(f"  Categorical: {len(self.categorical_vars)}")
        print(f"  Total: {len(self.continuous_vars) + len(self.categorical_vars)}")
        
        if self.exclude_cols:
            print(f"\nExcluded variables: {', '.join(self.exclude_cols)}")
        
        print("\n" + "-"*80)
        
        # Display table based on style
        if style == 'publication':
            self._display_publication_style(table_df)
        elif style == 'fancy':
            self._display_fancy_style(table_df)
        else:
            print(table_df.to_string(index=False))
    
    def _display_publication_style(self, table_df: pd.DataFrame) -> None:
        """Display table in publication style."""
        # Create formatted display
        for _, row in table_df.iterrows():
            if row['Variable'] in ['CONTINUOUS VARIABLES', 'CATEGORICAL VARIABLES', 'LONGITUDINAL MCH']:
                print(f"\n{row['Variable']}")
                print("-" * len(row['Variable']))
            elif row['Variable'].startswith('  '):
                # Subcategory
                line = f"    {row['Variable'][2:]:<35}"
                for col in table_df.columns[2:]:
                    if col not in ['P-value', 'Test'] and pd.notna(row[col]):
                        line += f" {str(row[col]):<15}"
                print(line)
            else:
                # Main variable
                line = f"{row['Variable']:<40}"
                if pd.notna(row['Type']):
                    line += f" ({row['Type']})"
                for col in table_df.columns[2:]:
                    if col not in ['P-value', 'Test'] and pd.notna(row[col]):
                        line += f"\n    {str(row[col]):<15}"
                if 'P-value' in table_df.columns and pd.notna(row['P-value']):
                    line += f" (p={row['P-value']}, {row['Test']})"
                print(line)
    
    def _display_fancy_style(self, table_df: pd.DataFrame) -> None:
        """Display table with fancy formatting using rich styling."""
        try:
            import rich
            from rich.table import Table
            from rich.console import Console
            
            console = Console()
            table = Table(show_header=True, header_style="bold magenta")
            
            # Add columns
            for col in table_df.columns:
                table.add_column(col, style="cyan")
            
            # Add rows
            for _, row in table_df.iterrows():
                table.add_row(*[str(val) if pd.notna(val) else "" for val in row])
            
            console.print(table)
            
        except ImportError:
            print("Rich library not available. Using simple style.")
            print(table_df.to_string(index=False))
    
    def save_table(self, filename: str, format: str = 'html', 
                  include_stats: bool = True) -> None:
        """
        Save the table to file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        format : str
            Output format ('html', 'latex', 'csv', 'excel')
        include_stats : bool
            Whether to include statistical tests
        """
        
        table_df = self.generate_table(include_stats)
        
        if format.lower() == 'html':
            html_content = self._generate_html_table(table_df)
            with open(filename, 'w') as f:
                f.write(html_content)
        elif format.lower() == 'latex':
            latex_content = self._generate_latex_table(table_df)
            with open(filename, 'w') as f:
                f.write(latex_content)
        elif format.lower() == 'csv':
            table_df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            table_df.to_excel(filename, index=False)
        
        print(f"Table saved as {filename}")
    
    def _generate_html_table(self, table_df: pd.DataFrame) -> str:
        """Generate publication-ready HTML table."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: bold; }
                .section-header { background-color: #e6f3ff; font-weight: bold; }
                .subcategory { padding-left: 20px; font-style: italic; }
            </style>
        </head>
        <body>
            <h2>Cohort Characteristics</h2>
        """
        
        html += table_df.to_html(index=False, escape=False, 
                                classes='table table-striped table-bordered')
        html += "</body></html>"
        
        return html
    
    def _generate_latex_table(self, table_df: pd.DataFrame) -> str:
        """Generate publication-ready LaTeX table."""
        # Create a copy and clean Unicode characters for LaTeX compatibility
        latex_df = table_df.copy()
        
        # Replace Unicode characters that cause issues in LaTeX
        for col in latex_df.columns:
            if latex_df[col].dtype == 'object':
                latex_df[col] = latex_df[col].astype(str).str.replace('X²', '$X^2$', regex=False)
                latex_df[col] = latex_df[col].str.replace('χ²', '$\\chi^2$', regex=False)
                latex_df[col] = latex_df[col].str.replace('β', '$\\beta$', regex=False)
                latex_df[col] = latex_df[col].str.replace('α', '$\\alpha$', regex=False)
                latex_df[col] = latex_df[col].str.replace('γ', '$\\gamma$', regex=False)
                latex_df[col] = latex_df[col].str.replace('δ', '$\\delta$', regex=False)
                latex_df[col] = latex_df[col].str.replace('ε', '$\\epsilon$', regex=False)
                latex_df[col] = latex_df[col].str.replace('μ', '$\\mu$', regex=False)
                latex_df[col] = latex_df[col].str.replace('σ', '$\\sigma$', regex=False)
                # Handle other common Unicode issues
                latex_df[col] = latex_df[col].str.replace('±', '$\\pm$', regex=False)
                latex_df[col] = latex_df[col].str.replace('≤', '$\\leq$', regex=False)
                latex_df[col] = latex_df[col].str.replace('≥', '$\\geq$', regex=False)
                latex_df[col] = latex_df[col].str.replace('≠', '$\\neq$', regex=False)
        
        # Add proper indentation for subcategories
        if 'Variable' in latex_df.columns:
            latex_df['Variable'] = latex_df['Variable'].apply(self._format_latex_indentation)
        
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Cohort Characteristics}
\\label{tab:cohort_characteristics}
        """
        
        latex += latex_df.to_latex(index=False, escape=False)
        latex += "\\end{table}"
        
        return latex
    
    def _format_latex_indentation(self, var_name: str) -> str:
        """Format variable names with proper LaTeX indentation."""
        if pd.isna(var_name) or var_name == 'nan':
            return ''
        
        var_name = str(var_name).strip()
        
        # Section headers (all caps with "VARIABLES")
        if var_name.isupper() and 'VARIABLES' in var_name:
            return f"\\textbf{{{var_name}}}"
        
        # Subcategories: True/False, single digits, simple values
        elif (var_name in ['False', 'True', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or
              (var_name.replace('.', '').isdigit() and len(var_name) <= 3)):
            return f"\\quad\\quad {var_name}"
        
        # Regular variables (no indentation)
        else:
            return var_name


def create_cohort_table(csv_file: str, group_col: str = None, 
                       exclude_cols: List[str] = None,
                       save_output: bool = False,
                       output_format: str = 'html',
                       baseline_only: bool = True,
                       id_col: str = 'RID',
                       viscode_col: str = 'VISCODE',
                       date_cols: Optional[List[str]] = None) -> CohortDescriptor:
    """
    Convenience function to create a cohort characteristics table from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    group_col : str, optional
        Column name for grouping analysis
    exclude_cols : list, optional
        List of columns to exclude
    save_output : bool
        Whether to save the output to file
    output_format : str
        Format for saved output ('html', 'latex', 'csv', 'excel')
    baseline_only : bool
        If True (default), restrict to baseline rows per participant before analysis
    id_col : str
        Participant identifier column (default 'RID')
    viscode_col : str
        Visit code column that may include 'bl' for baseline (default 'VISCODE')
    date_cols : list[str], optional
        Candidate date columns to determine earliest visit when VISCODE is not available
    
    Returns:
    --------
    CohortDescriptor
        The cohort descriptor object
    
    Example:
    --------
    # Basic usage
    descriptor = create_cohort_table('data.csv')
    descriptor.display_table()
    
    # With grouping
    descriptor = create_cohort_table('data.csv', group_col='treatment_group')
    descriptor.display_table(include_stats=True)
    
    # Save to file
    descriptor = create_cohort_table('data.csv', save_output=True)
    """
    
    # Load data
    try:
        df_full = pd.read_csv(csv_file)
        print(f"Loaded dataset: {csv_file}")
        print(f"Shape: {df_full.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Restrict to baseline per participant if requested
    if baseline_only:
        original_rows = len(df_full)
        df_baseline = _filter_to_baseline(df_full, id_col=id_col, viscode_col=viscode_col, date_cols=date_cols)
        try:
            unique_participants = df_baseline[id_col].nunique() if id_col in df_baseline.columns else 'N/A'
        except Exception:
            unique_participants = 'N/A'
        print(f"Baseline-only filtering applied: {original_rows} -> {len(df_baseline)} rows; participants: {unique_participants}")
    else:
        df_baseline = df_full.copy()

    # Determine date column for longitudinal analysis
    if date_cols and len(date_cols) > 0:
        date_col = date_cols[0]
    else:
        # Try to find a date column in the dataset
        default_date_cols = ['SCANDATE', 'EXAMDATE', 'EXAM_DATE', 'SCAN_DATE', 'VISITDATE', 'VISIT_DATE']
        date_col = 'SCANDATE'  # Default
        for dc in default_date_cols:
            if dc in df_full.columns:
                date_col = dc
                break
    
    # Create descriptor with full dataset for longitudinal analysis
    descriptor = CohortDescriptor(
        df_baseline, 
        group_col=group_col, 
        exclude_cols=exclude_cols,
        full_df=df_full,  # Pass full dataset for longitudinal MCH stats
        id_col=id_col,
        date_col=date_col
    )
    
    # Display table
    descriptor.display_table(include_stats=bool(group_col))
    
    # Save if requested
    if save_output:
        base_name = csv_file.replace('.csv', '')
        output_file = f"{base_name}_cohort_table.{output_format}"
        descriptor.save_table(output_file, format=output_format)
    
    return descriptor


# Example usage and testing
if __name__ == "__main__":
    print("Cohort Characteristics Table Generator")
    print("=====================================")

    print("Example usage:")
    print("1. Basic table: create_cohort_table('your_data.csv')")
    print("2. With groups: create_cohort_table('your_data.csv', group_col='treatment')")
    print("3. Exclude cols: create_cohort_table('your_data.csv', exclude_cols=['id', 'date'])")
    print("4. Save output: create_cohort_table('your_data.csv', save_output=True)")
    print()
    print("For the ARIA-Guard project, try:")
    print("descriptor = create_cohort_table('../../processed/worsening_medium.csv')")
    print("descriptor.display_table()")
    print()
    
    # Test with actual ARIA-Guard data
    test_file = 'processed/denorm_merge.csv'
    print(f"Testing with: {test_file}")
    
    try:
        descriptor = create_cohort_table(test_file, 
                                    exclude_cols=['SCANDATE','RID'], group_col='MCH_pos',save_output=True)
        descriptor.display_table(include_stats=True)     # <-- force stats

        print(f"Number of unique RIDs: {pd.read_csv(test_file)['RID'].nunique()}")
        
        if descriptor is not None:
            print("✅ Successfully loaded data!")
            # descriptor.display_table()
        else:
            print("❌ Failed to create descriptor - file not found")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\nTry using the absolute path to your CSV file:")
        print("descriptor = create_cohort_table(r'C:\\full\\path\\to\\your\\file.csv')")
        print("descriptor.display_table()")
