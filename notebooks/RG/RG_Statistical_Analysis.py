"""
Statistical Analysis for Regression (RG) Model Variants
Determines if performance differences between m3, medium, and m1 models are statistically significant.

This script performs:
1. Bootstrap resampling to estimate confidence intervals for R², MSE, MAE
2. Paired t-tests for regression metric comparisons
3. Wilcoxon signed-rank test for non-parametric comparison
4. Permutation tests for significance
5. Analysis of residual distributions
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, normaltest
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class RGStatisticalAnalysis:
    """Statistical analysis for regression model comparisons."""
    
    def __init__(self, base_path="../../"):
        """Initialize with paths to models and data."""
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.processed_path = self.base_path / "processed"
        self.viz_path = self.base_path / "viz" / "regression_results"
        
        # Create visualization directory if it doesn't exist
        self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Model variants
        self.variants = ['m3', 'medium', 'm1']
        self.models = {}
        self.data = {}
        self.predictions = {}
        self.results = {}
        
    def load_models_and_data(self):
        """Load trained models and test data for each variant."""
        print("Loading models and data...")
        
        for variant in self.variants:
            # Load model
            model_file = self.models_path / f"rg_{variant}_model.joblib"
            if model_file.exists():
                self.models[variant] = joblib.load(model_file)
                print(f"✓ Loaded {variant} model")
            else:
                print(f"✗ Model file not found: {model_file}")
                
            # Load data - using worsening data for regression
            data_file = self.processed_path / f"RG_{variant}_test.csv"
            if data_file.exists():
                self.data[variant] = pd.read_csv(data_file)
                print(f"✓ Loaded {variant} data ({len(self.data[variant])} samples)")
            else:
                print(f"✗ Data file not found: {data_file}")
                
    def prepare_test_sets(self):
        """Prepare consistent test sets for comparison."""
        print("\nPreparing test sets...")
        
        # Use the same random seed for reproducibility
        np.random.seed(42)
        
        for variant in self.variants:
            if variant not in self.data:
                continue
                
            df = self.data[variant]
            
            # Assuming 'cdrsb_change' or similar is the target for regression
            target_col = 'Duration'
            
            if target_col is None:
                print(f"⚠ Warning: No target column found for {variant}")
                continue

            feature_cols = [col for col in df.columns if col != 'Duration']
            
            if not feature_cols:
                print(f"⚠ Warning: No valid feature columns found for {variant}")
                continue
            
            # Check if we have a model to get expected features
            if variant in self.models:
                model = self.models[variant]
                
                # Try to get the number of features the model expects
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    if len(feature_cols) != expected_features:
                        print(f"  Feature mismatch: Data has {len(feature_cols)} features, "
                              f"model expects {expected_features}")
                        # Try to match features by taking first N features
                        if len(feature_cols) > expected_features:
                            feature_cols = feature_cols[:expected_features]
                            print(f"  Using first {expected_features} features")
                        else:
                            print(f"  ⚠ Warning: Not enough features for {variant}")
                            continue
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            
            # Use last 20% as test set (consistent with training scripts)
            test_size = int( len(X))
            X_test = X[-test_size:]
            y_test = y[-test_size:]
            
            # Generate predictions
            if variant in self.models:
                model = self.models[variant]
                y_pred = model.predict(X_test)
                
                self.predictions[variant] = {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'X_test': X_test,
                    'residuals': y_test - y_pred
                }
                
                # Calculate metrics
                self.results[variant] = {
                    'r2': r2_score(y_test, y_pred),
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100  # Mean Absolute Percentage Error
                }
                
                print(f"✓ {variant}: R²={self.results[variant]['r2']:.3f}, "
                      f"RMSE={self.results[variant]['rmse']:.3f}, "
                      f"MAE={self.results[variant]['mae']:.3f}")
    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap confidence intervals for regression metrics."""
        print(f"\nCalculating bootstrap confidence intervals (n={n_bootstrap})...")
        
        bootstrap_results = {}
        
        for variant in self.variants:
            if variant not in self.predictions:
                continue
                
            y_true = self.predictions[variant]['y_true']
            y_pred = self.predictions[variant]['y_pred']
            
            r2_scores = []
            mse_scores = []
            rmse_scores = []
            mae_scores = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = resample(range(len(y_true)), replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                
                # Calculate metrics
                r2_scores.append(r2_score(y_true_boot, y_pred_boot))
                mse_scores.append(mean_squared_error(y_true_boot, y_pred_boot))
                rmse_scores.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
                mae_scores.append(mean_absolute_error(y_true_boot, y_pred_boot))
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            bootstrap_results[variant] = {
                'r2_ci': (np.percentile(r2_scores, lower_percentile),
                         np.percentile(r2_scores, upper_percentile)),
                'r2_mean': np.mean(r2_scores),
                'r2_std': np.std(r2_scores),
                'mse_ci': (np.percentile(mse_scores, lower_percentile),
                          np.percentile(mse_scores, upper_percentile)),
                'mse_mean': np.mean(mse_scores),
                'rmse_ci': (np.percentile(rmse_scores, lower_percentile),
                           np.percentile(rmse_scores, upper_percentile)),
                'rmse_mean': np.mean(rmse_scores),
                'mae_ci': (np.percentile(mae_scores, lower_percentile),
                          np.percentile(mae_scores, upper_percentile)),
                'mae_mean': np.mean(mae_scores)
            }
            
            print(f"\n{variant.upper()} Model:")
            print(f"  R²: {bootstrap_results[variant]['r2_mean']:.3f} "
                  f"({bootstrap_results[variant]['r2_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['r2_ci'][1]:.3f})")
            print(f"  RMSE: {bootstrap_results[variant]['rmse_mean']:.3f} "
                  f"({bootstrap_results[variant]['rmse_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['rmse_ci'][1]:.3f})")
            print(f"  MAE: {bootstrap_results[variant]['mae_mean']:.3f} "
                  f"({bootstrap_results[variant]['mae_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['mae_ci'][1]:.3f})")
            
        return bootstrap_results
    
    def paired_tests(self):
        """Perform paired t-tests and Wilcoxon signed-rank tests."""
        print("\nPerforming paired statistical tests...")
        
        test_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'medium'), ('m3', 'm1'), ('medium', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
                
            y_true = self.predictions[model1]['y_true']
            pred1 = self.predictions[model1]['y_pred']
            pred2 = self.predictions[model2]['y_pred']
            
            # Calculate squared errors for each model
            squared_errors1 = (y_true - pred1) ** 2
            squared_errors2 = (y_true - pred2) ** 2
            
            # Paired t-test on squared errors
            t_stat, t_pvalue = ttest_rel(squared_errors1, squared_errors2)
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            w_stat, w_pvalue = wilcoxon(squared_errors1, squared_errors2)
            
            # Calculate absolute errors
            abs_errors1 = np.abs(y_true - pred1)
            abs_errors2 = np.abs(y_true - pred2)
            
            # Paired t-test on absolute errors
            t_stat_mae, t_pvalue_mae = ttest_rel(abs_errors1, abs_errors2)
            
            test_results[f"{model1}_vs_{model2}"] = {
                't_stat_mse': t_stat,
                't_pvalue_mse': t_pvalue,
                'wilcoxon_stat': w_stat,
                'wilcoxon_pvalue': w_pvalue,
                't_stat_mae': t_stat_mae,
                't_pvalue_mae': t_pvalue_mae,
                'significant_mse': t_pvalue < 0.05,
                'significant_mae': t_pvalue_mae < 0.05
            }
            
            print(f"\n{model1.upper()} vs {model2.upper()}:")
            print(f"  Paired t-test (MSE):")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {t_pvalue:.4f}")
            print(f"    Significant: {'Yes' if t_pvalue < 0.05 else 'No'}")
            print(f"  Wilcoxon test (MSE):")
            print(f"    statistic: {w_stat:.3f}")
            print(f"    p-value: {w_pvalue:.4f}")
            print(f"    Significant: {'Yes' if w_pvalue < 0.05 else 'No'}")
            
        return test_results
    
    def permutation_test(self, n_permutations=1000):
        """Perform permutation test for model comparison."""
        print(f"\nPerforming permutation test (n={n_permutations})...")
        
        permutation_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'medium'), ('m3', 'm1'), ('medium', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
                
            y_true = self.predictions[model1]['y_true']
            pred1 = self.predictions[model1]['y_pred']
            pred2 = self.predictions[model2]['y_pred']
            
            # Observed difference in R²
            r2_1 = r2_score(y_true, pred1)
            r2_2 = r2_score(y_true, pred2)
            observed_diff_r2 = r2_1 - r2_2
            
            # Observed difference in RMSE
            rmse_1 = np.sqrt(mean_squared_error(y_true, pred1))
            rmse_2 = np.sqrt(mean_squared_error(y_true, pred2))
            observed_diff_rmse = rmse_1 - rmse_2
            
            # Permutation test
            permuted_diffs_r2 = []
            permuted_diffs_rmse = []
            
            for _ in range(n_permutations):
                # Randomly swap predictions between models
                swap_mask = np.random.random(len(pred1)) > 0.5
                pred1_perm = np.where(swap_mask, pred2, pred1)
                pred2_perm = np.where(swap_mask, pred1, pred2)
                
                r2_1_perm = r2_score(y_true, pred1_perm)
                r2_2_perm = r2_score(y_true, pred2_perm)
                permuted_diffs_r2.append(r2_1_perm - r2_2_perm)
                
                rmse_1_perm = np.sqrt(mean_squared_error(y_true, pred1_perm))
                rmse_2_perm = np.sqrt(mean_squared_error(y_true, pred2_perm))
                permuted_diffs_rmse.append(rmse_1_perm - rmse_2_perm)
            
            # Calculate p-values
            permuted_diffs_r2 = np.array(permuted_diffs_r2)
            permuted_diffs_rmse = np.array(permuted_diffs_rmse)
            
            p_value_r2 = np.mean(np.abs(permuted_diffs_r2) >= np.abs(observed_diff_r2))
            p_value_rmse = np.mean(np.abs(permuted_diffs_rmse) >= np.abs(observed_diff_rmse))
            
            permutation_results[f"{model1}_vs_{model2}"] = {
                'observed_diff_r2': observed_diff_r2,
                'p_value_r2': p_value_r2,
                'significant_r2': p_value_r2 < 0.05,
                'observed_diff_rmse': observed_diff_rmse,
                'p_value_rmse': p_value_rmse,
                'significant_rmse': p_value_rmse < 0.05
            }
            
            print(f"\n{model1.upper()} vs {model2.upper()}:")
            print(f"  R² difference: {observed_diff_r2:.4f}, p-value: {p_value_r2:.4f}")
            print(f"  RMSE difference: {observed_diff_rmse:.4f}, p-value: {p_value_rmse:.4f}")
            
        return permutation_results
    
    def analyze_residuals(self):
        """Analyze residual distributions for each model."""
        print("\nAnalyzing residual distributions...")
        
        residual_analysis = {}
        
        for variant in self.variants:
            if variant not in self.predictions:
                continue
                
            residuals = self.predictions[variant]['residuals']
            
            # Normality test
            stat, p_value = normaltest(residuals)
            
            # Calculate residual statistics
            residual_analysis[variant] = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals),
                'normality_stat': stat,
                'normality_pvalue': p_value,
                'is_normal': p_value > 0.05
            }
            
            print(f"\n{variant.upper()} Model Residuals:")
            print(f"  Mean: {residual_analysis[variant]['mean']:.4f}")
            print(f"  Std Dev: {residual_analysis[variant]['std']:.4f}")
            print(f"  Skewness: {residual_analysis[variant]['skewness']:.4f}")
            print(f"  Kurtosis: {residual_analysis[variant]['kurtosis']:.4f}")
            print(f"  Normal distribution: {'Yes' if residual_analysis[variant]['is_normal'] else 'No'} (p={p_value:.4f})")
            
        return residual_analysis
    
    def visualize_results(self, bootstrap_results, residual_analysis):
        """Create visualizations for statistical analysis results."""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. R² comparison with confidence intervals
        ax = axes[0, 0]
        variants_list = list(bootstrap_results.keys())
        r2_scores = [bootstrap_results[v]['r2_mean'] for v in variants_list]
        ci_lower = [bootstrap_results[v]['r2_ci'][0] for v in variants_list]
        ci_upper = [bootstrap_results[v]['r2_ci'][1] for v in variants_list]
        errors = [[r2_scores[i] - ci_lower[i] for i in range(len(variants_list))],
                 [ci_upper[i] - r2_scores[i] for i in range(len(variants_list))]]
        
        x_pos = np.arange(len(variants_list))
        ax.bar(x_pos, r2_scores, yerr=errors, capsize=10,
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)
        
        # 2. RMSE comparison
        ax = axes[0, 1]
        rmse_scores = [bootstrap_results[v]['rmse_mean'] for v in variants_list]
        rmse_ci_lower = [bootstrap_results[v]['rmse_ci'][0] for v in variants_list]
        rmse_ci_upper = [bootstrap_results[v]['rmse_ci'][1] for v in variants_list]
        rmse_errors = [[rmse_scores[i] - rmse_ci_lower[i] for i in range(len(variants_list))],
                      [rmse_ci_upper[i] - rmse_scores[i] for i in range(len(variants_list))]]
        
        ax.bar(x_pos, rmse_scores, yerr=rmse_errors, capsize=10,
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)
        
        # 3. MAE comparison
        ax = axes[0, 2]
        mae_scores = [bootstrap_results[v]['mae_mean'] for v in variants_list]
        mae_ci_lower = [bootstrap_results[v]['mae_ci'][0] for v in variants_list]
        mae_ci_upper = [bootstrap_results[v]['mae_ci'][1] for v in variants_list]
        mae_errors = [[mae_scores[i] - mae_ci_lower[i] for i in range(len(variants_list))],
                     [mae_ci_upper[i] - mae_scores[i] for i in range(len(variants_list))]]
        
        ax.bar(x_pos, mae_scores, yerr=mae_errors, capsize=10,
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('MAE')
        ax.set_title('MAE with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)
        
        # 4. Residual distributions
        ax = axes[1, 0]
        for variant in self.variants:
            if variant in self.predictions:
                ax.hist(self.predictions[variant]['residuals'], alpha=0.5, 
                       label=variant.upper(), bins=30)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Q-Q plot for residuals
        ax = axes[1, 1]
        for i, variant in enumerate(self.variants):
            if variant in self.predictions:
                stats.probplot(self.predictions[variant]['residuals'], 
                             dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)
        
        # 6. Predicted vs Actual scatter plot
        ax = axes[1, 2]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        for i, variant in enumerate(self.variants):
            if variant in self.predictions:
                y_true = self.predictions[variant]['y_true']
                y_pred = self.predictions[variant]['y_pred']
                ax.scatter(y_true, y_pred, alpha=0.5, label=variant.upper(),
                          color=colors[i], s=20)
        
        # Add diagonal line
        min_val = min([self.predictions[v]['y_true'].min() for v in self.variants if v in self.predictions])
        max_val = max([self.predictions[v]['y_true'].max() for v in self.variants if v in self.predictions])
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Regression Model Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = self.viz_path / 'statistical_analysis_rg.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
    
    def generate_report(self, bootstrap_results, paired_results, permutation_results, residual_analysis):
        """Generate a comprehensive statistical analysis report."""
        print("\nGenerating statistical analysis report...")
        
        report = []
        report.append("=" * 80)
        report.append("REGRESSION MODEL STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        for variant in self.variants:
            if variant in self.results:
                report.append(f"\n{variant.upper()} Model:")
                report.append(f"  R² Score: {self.results[variant]['r2']:.4f}")
                report.append(f"  RMSE: {self.results[variant]['rmse']:.4f}")
                report.append(f"  MAE: {self.results[variant]['mae']:.4f}")
                report.append(f"  MAPE: {self.results[variant]['mape']:.2f}%")
        
        # Bootstrap confidence intervals
        report.append("\n2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        report.append("-" * 40)
        for variant, results in bootstrap_results.items():
            report.append(f"\n{variant.upper()} Model:")
            report.append(f"  R²: {results['r2_mean']:.3f} "
                         f"({results['r2_ci'][0]:.3f}, {results['r2_ci'][1]:.3f})")
            report.append(f"  RMSE: {results['rmse_mean']:.3f} "
                         f"({results['rmse_ci'][0]:.3f}, {results['rmse_ci'][1]:.3f})")
            report.append(f"  MAE: {results['mae_mean']:.3f} "
                         f"({results['mae_ci'][0]:.3f}, {results['mae_ci'][1]:.3f})")
        
        # Statistical tests
        report.append("\n3. STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        
        report.append("\nPaired T-Test Results (MSE):")
        for comparison, results in paired_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    T-statistic: {results['t_stat_mse']:.3f}")
            report.append(f"    P-value: {results['t_pvalue_mse']:.4f}")
            report.append(f"    Significant: {'Yes' if results['significant_mse'] else 'No'}")
        
        report.append("\nWilcoxon Signed-Rank Test Results:")
        for comparison, results in paired_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    Statistic: {results['wilcoxon_stat']:.3f}")
            report.append(f"    P-value: {results['wilcoxon_pvalue']:.4f}")
            report.append(f"    Significant: {'Yes' if results['wilcoxon_pvalue'] < 0.05 else 'No'}")
        
        report.append("\nPermutation Test Results:")
        for comparison, results in permutation_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    R² difference: {results['observed_diff_r2']:.4f} "
                         f"(p={results['p_value_r2']:.4f})")
            report.append(f"    RMSE difference: {results['observed_diff_rmse']:.4f} "
                         f"(p={results['p_value_rmse']:.4f})")
        
        # Residual analysis
        report.append("\n4. RESIDUAL ANALYSIS")
        report.append("-" * 40)
        for variant, analysis in residual_analysis.items():
            report.append(f"\n{variant.upper()} Model:")
            report.append(f"  Mean residual: {analysis['mean']:.4f}")
            report.append(f"  Std deviation: {analysis['std']:.4f}")
            report.append(f"  Skewness: {analysis['skewness']:.4f}")
            report.append(f"  Kurtosis: {analysis['kurtosis']:.4f}")
            report.append(f"  Normally distributed: {'Yes' if analysis['is_normal'] else 'No'} "
                         f"(p={analysis['normality_pvalue']:.4f})")
        
        # Conclusions
        report.append("\n5. CONCLUSIONS")
        report.append("-" * 40)
        
        # Check for overlapping confidence intervals
        ci_overlap = []
        if 'm3' in bootstrap_results and 'medium' in bootstrap_results:
            if (bootstrap_results['m3']['r2_ci'][0] <= bootstrap_results['medium']['r2_ci'][1] and
                bootstrap_results['medium']['r2_ci'][0] <= bootstrap_results['m3']['r2_ci'][1]):
                ci_overlap.append("M3 and M2 models have overlapping R² confidence intervals")
        
        if 'm3' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['m3']['r2_ci'][0] <= bootstrap_results['m1']['r2_ci'][1] and
                bootstrap_results['m1']['r2_ci'][0] <= bootstrap_results['m3']['r2_ci'][1]):
                ci_overlap.append("M3 and M1 models have overlapping R² confidence intervals")
        
        if 'medium' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['medium']['r2_ci'][0] <= bootstrap_results['m1']['r2_ci'][1] and
                bootstrap_results['m1']['r2_ci'][0] <= bootstrap_results['medium']['r2_ci'][1]):
                ci_overlap.append("M2 and M1 models have overlapping R² confidence intervals")
        
        if ci_overlap:
            report.append("\nConfidence Interval Analysis:")
            for overlap in ci_overlap:
                report.append(f"  • {overlap}")
        
        # Summary of significant differences
        sig_diffs = []
        for comparison, results in paired_results.items():
            if results['significant_mse']:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Paired t-test, MSE)")
            if results['wilcoxon_pvalue'] < 0.05:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Wilcoxon test)")
        
        for comparison, results in permutation_results.items():
            if results['significant_r2']:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Permutation test, R²)")
            if results['significant_rmse']:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Permutation test, RMSE)")
        
        if sig_diffs:
            report.append("\nStatistically Significant Differences Found:")
            for diff in sig_diffs:
                report.append(f"  • {diff}")
        else:
            report.append("\nNo statistically significant differences found between model variants.")
        
        # Best performing model
        if self.results:
            best_r2 = max(self.results.items(), key=lambda x: x[1]['r2'])
            best_rmse = min(self.results.items(), key=lambda x: x[1]['rmse'])
            report.append(f"\nBest performing model by R²: {best_r2[0].upper()} (R²={best_r2[1]['r2']:.4f})")
            report.append(f"Best performing model by RMSE: {best_rmse[0].upper()} (RMSE={best_rmse[1]['rmse']:.4f})")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.viz_path / 'statistical_analysis_rg_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("Starting Regression Model Statistical Analysis")
        print("=" * 60)
        
        # Load models and data
        self.load_models_and_data()
        
        # Prepare test sets
        self.prepare_test_sets()
        
        if len(self.predictions) < 2:
            print("\n⚠ Warning: Need at least 2 models for comparison")
            return
        
        # Run statistical tests
        bootstrap_results = self.bootstrap_confidence_intervals(n_bootstrap=1000)
        paired_results = self.paired_tests()
        permutation_results = self.permutation_test(n_permutations=1000)
        residual_analysis = self.analyze_residuals()
        
        # Visualize results
        self.visualize_results(bootstrap_results, residual_analysis)
        
        # Generate report
        self.generate_report(bootstrap_results, paired_results, 
                            permutation_results, residual_analysis)
        
        print("\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    analyzer = RGStatisticalAnalysis()
    analyzer.run_analysis()
