#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Characteristics Comparison for MCH Positive vs Negative Participants

This script analyzes baseline characteristics of individuals comparing those who are 
MCH positive at their first observation versus MCH negative participants.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the merged dataset and prepare baseline data."""
    # Load the merged dataset
    df = pd.read_csv("processed/merge.csv")
    
    # Get first observation for each participant
    baseline_df = df.loc[df.groupby('RID')['SCANDATE'].idxmin()].copy()
    
    print(f"Total participants: {baseline_df['RID'].nunique()}")
    print(f"MCH positive at baseline: {baseline_df['MCH_pos'].sum()}")
    print(f"MCH negative at baseline: {(baseline_df['MCH_pos'] == 0).sum()}")
    
    return baseline_df

def perform_baseline_analysis(df):
    """Perform comprehensive baseline characteristics analysis."""
    
    # Define variable categories for reporting
    variable_categories = {
        'Demographics': ['PTAGE', 'PTGENDER', 'PTEDUCAT', 'RACE_ETHNICITY'],
        'Genetics': ['e4_GENOTYPE'],
        'Cognitive Scores': ['CDRSB', 'MMSE', 'FAQ', 'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP', 'DIAGNOSIS'],
        'Biomarkers': ['ptau_ab_ratio_csf', 'PLASMA_NFL'],
        'Imaging': ['NORM_WMH', 'NORM_GRAY'],
        'Clinical Measures': ['PHC_BMI', 'PHC_Smoker'],
        'Medical History': ['CARD', 'RENA', 'PSYCH', 'NEURL', 'HEAD', 'RESP', 'HEPAT', 
                           'DERM', 'MUSCL', 'ENDO', 'GAST', 'HEMA', 'ALLE', 'ALCH'],
        'Medications': ['MED_Anti_Thrombotic', 'MED_AD_and_Dementia', 'MED_Lipid_Lowering',
                       'MED_Blood_Pressure', 'MED_Thyroid', 'MED_Vitamins/Minerals',
                       'MED_NSAIDs', 'MED_Herbal_Supplements', 'MED_Analgesics',
                       'MED_Antidepressants', 'MED_Diabetes_Oral', 'MED_Other_GI',
                       'MED_General_Urological', 'MED_Antibiotics', 'MED_Osteoporosis',
                       'MED_Inhalers', 'MED_Sleep_Aids', 'MED_Hormone_Replacement',
                       'MED_GERD/PPI', 'MED_Anticonvulsants', 'MED_Antihistamines',
                       'MED_H2_Blockers', 'MED_Nasal_Sprays', 'MED_Diabetes_Injectable',
                       'MED_Corticosteroids', 'MED_Benzodiazepines', 'MED_Eye_Supplements',
                       'MED_Other_Respiratory', 'MED_Antiviral', 'MED_Anti_Arrythmia']
    }
    
    # Separate MCH positive and negative groups
    mch_pos = df[df['MCH_pos'] == 1]
    mch_neg = df[df['MCH_pos'] == 0]
    
    results = {}
    
    for category, variables in variable_categories.items():
        category_results = {}
        
        for var in variables:
            if var not in df.columns:
                continue
                
            try:
                if df[var].dtype in ['object', 'category'] or var in ['PTGENDER', 'RACE_ETHNICITY']:
                    # Categorical variables
                    category_results[var] = analyze_categorical_variable(mch_pos, mch_neg, var)
                else:
                    # Continuous variables
                    category_results[var] = analyze_continuous_variable(mch_pos, mch_neg, var)
                    
            except Exception as e:
                print(f"Error analyzing {var}: {e}")
                continue
                
        if category_results:
            results[category] = category_results
    
    return results

def analyze_continuous_variable(mch_pos, mch_neg, var):
    """Analyze continuous variables between MCH groups."""
    
    # Get values, dropping NaN
    pos_vals = mch_pos[var].dropna()
    neg_vals = mch_neg[var].dropna()
    
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return None
    
    # Descriptive statistics
    pos_mean = pos_vals.mean()
    pos_std = pos_vals.std()
    pos_n = len(pos_vals)
    
    neg_mean = neg_vals.mean()
    neg_std = neg_vals.std()
    neg_n = len(neg_vals)
    
    # Statistical test
    # Check for normality approximation using Shapiro-Wilk on smaller sample
    if min(pos_n, neg_n) > 30:
        # Use t-test for larger samples
        statistic, p_value = ttest_ind(pos_vals, neg_vals)
        test_name = "Independent t-test"
    else:
        # Use Mann-Whitney U for smaller samples
        statistic, p_value = mannwhitneyu(pos_vals, neg_vals, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((pos_n - 1) * pos_std**2 + (neg_n - 1) * neg_std**2) / (pos_n + neg_n - 2))
    cohens_d = (pos_mean - neg_mean) / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interp = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    return {
        'test_name': test_name,
        'pos_n': pos_n,
        'neg_n': neg_n,
        'pos_mean': pos_mean,
        'pos_std': pos_std,
        'neg_mean': neg_mean,
        'neg_std': neg_std,
        'mean_diff': pos_mean - neg_mean,
        'statistic': statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_interpretation': effect_interp,
        'significant': p_value < 0.05
    }

def analyze_categorical_variable(mch_pos, mch_neg, var):
    """Analyze categorical variables between MCH groups."""
    
    # Create contingency table
    pos_counts = mch_pos[var].value_counts()
    neg_counts = mch_neg[var].value_counts()
    
    # Combine to get all categories
    all_categories = set(pos_counts.index) | set(neg_counts.index)
    
    # Build contingency table
    contingency_data = []
    for cat in all_categories:
        contingency_data.append([pos_counts.get(cat, 0), neg_counts.get(cat, 0)])
    
    contingency_table = pd.DataFrame(contingency_data, 
                                   index=list(all_categories), 
                                   columns=['MCH_pos', 'MCH_neg'])
    
    # Chi-square test
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return None
        
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Cramer's V for effect size
    n = contingency_table.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    
    # Interpret Cramer's V
    if cramers_v < 0.1:
        effect_interp = "negligible"
    elif cramers_v < 0.3:
        effect_interp = "small"
    elif cramers_v < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    # Calculate percentages
    pos_total = len(mch_pos[var].dropna())
    neg_total = len(mch_neg[var].dropna())
    
    pos_percentages = {cat: (count / pos_total * 100) for cat, count in pos_counts.items()}
    neg_percentages = {cat: (count / neg_total * 100) for cat, count in neg_counts.items()}
    
    return {
        'test_name': "Chi-square test",
        'pos_n': pos_total,
        'neg_n': neg_total,
        'contingency_table': contingency_table.to_dict(),
        'pos_percentages': pos_percentages,
        'neg_percentages': neg_percentages,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'effect_interpretation': effect_interp,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05
    }

def format_results_for_publication(results):
    """Format results in a publication-ready format."""
    
    output_lines = []
    output_lines.append("BASELINE CHARACTERISTICS COMPARISON")
    output_lines.append("=" * 60)
    output_lines.append(f"Comparing MCH-positive vs MCH-negative participants at first observation")
    output_lines.append("")
    
    for category, variables in results.items():
        output_lines.append(f"\n{category.upper()}:")
        output_lines.append("-" * (len(category) + 1))
        
        for var, result in variables.items():
            if result is None:
                continue
                
            if 'pos_mean' in result:  # Continuous variable
                pos_mean = result['pos_mean']
                pos_std = result['pos_std']
                neg_mean = result['neg_mean']
                neg_std = result['neg_std']
                p_val = result['p_value']
                
                # Format p-value
                if p_val < 0.001:
                    p_str = "p < 0.001"
                else:
                    p_str = f"p = {p_val:.4f}"
                
                # Significance indicator
                sig = "*" if result['significant'] else ""
                
                output_lines.append(f"  {var}: MCH-pos {pos_mean:.1f} ± {pos_std:.1f} vs MCH-neg {neg_mean:.1f} ± {neg_std:.1f}, {p_str} {sig}")
                
            else:  # Categorical variable
                pos_pct = result['pos_percentages']
                neg_pct = result['neg_percentages']
                p_val = result['p_value']
                
                # Find the category with highest difference
                max_diff = 0
                max_cat = None
                for cat in set(pos_pct.keys()) | set(neg_pct.keys()):
                    pos_val = pos_pct.get(cat, 0)
                    neg_val = neg_pct.get(cat, 0)
                    diff = abs(pos_val - neg_val)
                    if diff > max_diff:
                        max_diff = diff
                        max_cat = cat
                
                if max_cat:
                    # Format p-value
                    if p_val < 0.001:
                        p_str = "p < 0.001"
                    else:
                        p_str = f"p = {p_val:.4f}"
                    
                    # Significance indicator
                    sig = "*" if result['significant'] else ""
                    
                    pos_val = pos_pct.get(max_cat, 0)
                    neg_val = neg_pct.get(max_cat, 0)
                    output_lines.append(f"  {var} ({max_cat}): MCH-pos {pos_val:.1f}% vs MCH-neg {neg_val:.1f}%, {p_str} {sig}")
    
    output_lines.append("\n" + "=" * 60)
    output_lines.append("* indicates statistically significant difference (p < 0.05)")
    
    return "\n".join(output_lines)

def create_summary_table(results):
    """Create a summary table of significant findings."""
    
    significant_findings = []
    
    for category, variables in results.items():
        for var, result in variables.items():
            if result is None or not result['significant']:
                continue
                
            if 'pos_mean' in result:  # Continuous
                significant_findings.append({
                    'Category': category,
                    'Variable': var,
                    'Type': 'Continuous',
                    'MCH_pos_mean': f"{result['pos_mean']:.3f}",
                    'MCH_neg_mean': f"{result['neg_mean']:.3f}",
                    'Difference': f"{result['mean_diff']:.3f}",
                    'P_value': result['p_value'],
                    'Effect_size': f"{result['cohens_d']:.3f}"
                })
            else:  # Categorical
                significant_findings.append({
                    'Category': category,
                    'Variable': var,
                    'Type': 'Categorical',
                    'Test': result['test_name'],
                    'P_value': result['p_value'],
                    'Effect_size': f"{result['cramers_v']:.3f}"
                })
    
    if significant_findings:
        summary_df = pd.DataFrame(significant_findings)
        return summary_df
    else:
        return None

def main():
    """Main analysis function."""
    print("Loading and preparing baseline data...")
    baseline_df = load_and_prepare_data()
    
    print("\nPerforming baseline characteristics analysis...")
    results = perform_baseline_analysis(baseline_df)
    
    print("\nGenerating publication-ready results...")
    formatted_results = format_results_for_publication(results)
    
    # Save to file
    with open("EDA/baseline_characteristics_comparison.txt", "w") as f:
        f.write(formatted_results)
    
    print("Results saved to: EDA/baseline_characteristics_comparison.txt")
    print("\n" + formatted_results)
    
    # Create summary table of significant findings
    summary_df = create_summary_table(results)
    if summary_df is not None:
        summary_df.to_csv("EDA/baseline_characteristics_summary.csv", index=False)
        print(f"\nSummary of {len(summary_df)} significant findings saved to: EDA/baseline_characteristics_summary.csv")
        print("\nSignificant findings summary:")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo significant findings found at p < 0.05 level")

if __name__ == "__main__":
    main()
