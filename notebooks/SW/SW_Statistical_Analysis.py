"""
Statistical Analysis for Switch (SW) Model Variants
Determines if accuracy differences between m3, m2, and m1 models are statistically significant.

This script performs:
1. Bootstrap resampling to estimate 95% confidence intervals for accuracy
2. McNemar's test for paired binary classification comparisons across variants
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class SWStatisticalAnalysis:
    """Statistical analysis for switch model comparisons."""
    
    def __init__(self, base_path=None):
        """Initialize with paths to models and data.
        If base_path is None, derive project root from this file location.
        """
        self.base_path = Path(base_path) if base_path is not None else Path(__file__).resolve().parents[2]
        self.models_path = self.base_path / "models"
        self.processed_path = self.base_path / "processed"
        self.viz_path = self.base_path / "viz" / "switch_results"
        
        # Create visualization directory if it doesn't exist
        self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Model variants
        self.variants = ['m3', 'm2', 'm1']
        self.models = {}
        self.data = {}
        self.predictions = {}
        self.results = {}
        
    def load_models_and_data(self):
        """Load trained models and test data for each variant."""
        print("Loading models and data...")
        
        
        # Load variant-specific models and data
        for variant in self.variants:
            # Try multiple expected filenames for robustness
            candidate_model_files = [
                self.models_path / f"sw_{variant}_model.joblib",
                self.models_path / f"sw_{variant}_model.pkl",
                self.models_path / f"sw_{variant}_model.sav",
                self.models_path / f"sw{variant}_model.pkl",
                self.models_path / f"sw{variant}_model.joblib",
            ]
            model_loaded = False
            for model_file in candidate_model_files:
                if model_file.exists():
                    try:
                        self.models[variant] = joblib.load(model_file)
                        print(f"✓ Loaded {variant} model from {model_file.name}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"✗ Failed to load {variant} model from {model_file.name}: {e}")
            if not model_loaded:
                print(f"✗ No model file found for {variant} in {self.models_path}")
            
            # Load classification data for each variant
            data_file = self.processed_path / f"SW_{variant}_test.csv"

            if data_file.exists():
                self.data[variant] = pd.read_csv(data_file)
                print(f"✓ Loaded {variant} data ({len(self.data[variant])} samples)")
           
                
    def prepare_test_sets(self):
        """Prepare test sets using saved SW_{variant}_test.csv for each variant."""
        print("\nPreparing test sets from saved SW_*_test.csv files...")

        for variant in self.variants:
            if variant not in self.data:
                continue

            df = self.data[variant].copy()

            # Validate target column and drop rows with missing target
            if 'SWITCH_STATUS' not in df.columns:
                print(f"✗ 'SWITCH_STATUS' not found in test data for {variant}; skipping.")
                continue
            df = df.dropna(subset=['SWITCH_STATUS'])

            # Select feature columns: exclude identifiers and other non-features if present
            non_feature_cols = {'SWITCH_STATUS', 'RID', 'SCANDATE', 'MCH_pos', 'MCH_count'}
            feature_cols = [col for col in df.columns if col not in non_feature_cols]

            # Preserve identifiers for sample alignment across variants
            key_cols = [c for c in ['RID', 'SCANDATE'] if c in df.columns]
            if key_cols:
                keys_df = df[key_cols].copy()
            else:
                keys_df = pd.DataFrame({'row_idx': np.arange(len(df))})

            X_test = df[feature_cols].values
            y_test = df['SWITCH_STATUS'].values

            if variant in self.models:
                model = self.models[variant]

                # Predict probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if y_pred_proba.ndim > 1:
                        y_pred_proba = y_pred_proba[:, 1]
                else:
                    # Fallback: treat model.predict output as score
                    y_pred_proba = model.predict(X_test)

                # Default threshold 0.5 (can be adjusted if variant-specific threshold is known)
                y_pred = (y_pred_proba >= 0.5).astype(int)

                self.predictions[variant] = {
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'X_test': X_test,
                    'keys': keys_df
                }

                # Calculate metrics
                # Calculate metrics (include AUC, guard for single-class edge case)
                try:
                    auc_value = roc_auc_score(y_test, y_pred_proba)
                except Exception:
                    auc_value = np.nan

                self.results[variant] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'roc_auc': auc_value
                }

                if np.isnan(auc_value):
                    print(f"✓ {variant}: Accuracy={self.results[variant]['accuracy']:.3f}, AUC=NA (single-class or invalid)")
                else:
                    print(f"✓ {variant}: Accuracy={self.results[variant]['accuracy']:.3f}, AUC={auc_value:.3f}")
    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap 95% confidence intervals for accuracy and AUC."""
        print(f"\nCalculating bootstrap confidence intervals (n={n_bootstrap})...")
        
        bootstrap_results = {}
        
        for variant in self.variants:
            if variant not in self.predictions:
                continue
                
            y_true = self.predictions[variant]['y_true']
            y_pred = self.predictions[variant]['y_pred']
            y_pred_proba = self.predictions[variant]['y_pred_proba']
            
            accuracies = []
            aucs = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = resample(range(len(y_true)), replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                y_pred_proba_boot = y_pred_proba[indices]
                
                # Calculate metrics
                accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
                # Only compute AUC if both classes are present in the bootstrap sample
                if len(np.unique(y_true_boot)) > 1:
                    try:
                        aucs.append(roc_auc_score(y_true_boot, y_pred_proba_boot))
                    except Exception:
                        pass
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            bootstrap_results[variant] = {
                'accuracy_ci': (np.percentile(accuracies, lower_percentile),
                               np.percentile(accuracies, upper_percentile)),
                'accuracy_mean': np.mean(accuracies),
                'auc_ci': (np.percentile(aucs, lower_percentile),
                           np.percentile(aucs, upper_percentile)) if len(aucs) > 0 else (np.nan, np.nan),
                'auc_mean': np.mean(aucs) if len(aucs) > 0 else np.nan
            }
            
            print(f"\n{variant.upper()} Model:")
            print(f"  Accuracy: {bootstrap_results[variant]['accuracy_mean']:.3f} "
                  f"({bootstrap_results[variant]['accuracy_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['accuracy_ci'][1]:.3f})")
            if not np.isnan(bootstrap_results[variant]['auc_mean']):
                print(f"  AUC: {bootstrap_results[variant]['auc_mean']:.3f} "
                      f"({bootstrap_results[variant]['auc_ci'][0]:.3f}, "
                      f"{bootstrap_results[variant]['auc_ci'][1]:.3f})")
            else:
                print("  AUC: NA (insufficient class variation in bootstrap samples)")
            
        return bootstrap_results
    
    def pairwise_mcnemar_tests(self):
        """Run pairwise McNemar's tests using paired predictions across variants.

        Alignment uses common keys among ['RID', 'SCANDATE'] if present; otherwise, row indices.
        """
        print("\nRunning McNemar's tests (paired comparisons)...")

        mcnemar_results = {}
        pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]

        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue

            keys1 = self.predictions[model1].get('keys')
            keys2 = self.predictions[model2].get('keys')
            y_true1 = self.predictions[model1]['y_true']
            y_pred1 = self.predictions[model1]['y_pred']
            y_true2 = self.predictions[model2]['y_true']
            y_pred2 = self.predictions[model2]['y_pred']

            df1 = keys1.copy() if isinstance(keys1, pd.DataFrame) else pd.DataFrame({'row_idx': np.arange(len(y_true1))})
            df1['y_true'] = y_true1
            df1['y_pred'] = y_pred1

            df2 = keys2.copy() if isinstance(keys2, pd.DataFrame) else pd.DataFrame({'row_idx': np.arange(len(y_true2))})
            df2['y_true'] = y_true2
            df2['y_pred'] = y_pred2

            common_keys = [c for c in df1.columns if c in df2.columns and c not in ['y_true', 'y_pred']]
            merged = pd.merge(df1, df2, on=common_keys, suffixes=('1', '2'))

            if merged.empty:
                print(f"  ⚠ Skipping {model1} vs {model2}: no overlapping samples found.")
                continue

            merged = merged[merged['y_true1'] == merged['y_true2']].copy()
            if merged.empty:
                print(f"  ⚠ Skipping {model1} vs {model2}: no rows with matching ground truth.")
                continue

            correct1 = merged['y_pred1'] == merged['y_true1']
            correct2 = merged['y_pred2'] == merged['y_true2']

            a = int(np.sum(correct1 & correct2))
            b = int(np.sum(correct1 & (~correct2)))
            c = int(np.sum((~correct1) & correct2))
            d = int(np.sum((~correct1) & (~correct2)))

            table = [[a, b], [c, d]]

            if (b + c) == 0:
                statistic = 0.0
                p_value = 1.0
            else:
                res = mcnemar(table, exact=False, correction=True)
                statistic = float(res.statistic)
                p_value = float(res.pvalue)

            better = model1 if b > c else (model2 if c > b else 'tie')

            mcnemar_results[f"{model1}_vs_{model2}"] = {
                'n': int(a + b + c + d),
                'a_both_correct': a,
                'b_model1_only': b,
                'c_model2_only': c,
                'd_both_wrong': d,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'better': better
            }

            print(f"\n{model1.upper()} vs {model2.upper()}:")
            print(f"  Contingency: a={a}, b={b}, c={c}, d={d} (n={a+b+c+d})")
            print(f"  McNemar statistic: {statistic:.3f}, p-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

        return mcnemar_results
    
    def calibration_analysis(self):
        """Deprecated: Calibration analysis removed in favor of McNemar's paired tests."""
        print("\nCalibration analysis skipped (not used in this SW analysis).")
        return {}
    
    def visualize_results(self, bootstrap_results, mcnemar_results):
        """Create visualizations for accuracy/AUC CIs and McNemar results, with ROC curves."""
        print("\nCreating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Accuracy comparison with confidence intervals
        ax = axes[0, 0]
        variants_list = [v for v in self.variants if v in bootstrap_results]
        accuracies = [bootstrap_results[v]['accuracy_mean'] for v in variants_list]
        ci_lower = [bootstrap_results[v]['accuracy_ci'][0] for v in variants_list]
        ci_upper = [bootstrap_results[v]['accuracy_ci'][1] for v in variants_list]
        errors = [[accuracies[i] - ci_lower[i] for i in range(len(variants_list))],
                 [ci_upper[i] - accuracies[i] for i in range(len(variants_list))]]

        x_pos = np.arange(len(variants_list))
        ax.bar(x_pos, accuracies, yerr=errors, capsize=10,
               color=['#2E86AB', '#A23B72', '#F18F01'][:len(variants_list)], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Accuracy')
        ax.set_title('Switch Model Accuracy with 95% CI')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)

        # 2. McNemar results table
        # 2. AUC comparison with confidence intervals
        ax = axes[0, 1]
        auc_means = [bootstrap_results[v]['auc_mean'] for v in variants_list]
        auc_ci_lower = [bootstrap_results[v]['auc_ci'][0] for v in variants_list]
        auc_ci_upper = [bootstrap_results[v]['auc_ci'][1] for v in variants_list]

        # Handle NaNs: replace with zeros for plotting; mask labels accordingly
        auc_plot = [0 if np.isnan(x) else x for x in auc_means]
        auc_errors = [
            [0 if np.isnan(auc_means[i]) else auc_means[i] - auc_ci_lower[i] for i in range(len(variants_list))],
            [0 if np.isnan(auc_means[i]) else auc_ci_upper[i] - auc_means[i] for i in range(len(variants_list))]
        ]

        ax.bar(x_pos, auc_plot, yerr=auc_errors, capsize=10,
               color=['#7E57C2', '#26A69A', '#EF5350'][:len(variants_list)], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('ROC-AUC')
        ax.set_title('Switch Model ROC-AUC with 95% CI')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)

        # Add NA labels for AUC where applicable
        for i, m in enumerate(auc_means):
            if np.isnan(m):
                ax.text(i, 0.02, 'NA', ha='center', va='bottom', fontsize=9)

        # 3. McNemar results table
        ax = axes[1, 0]
        ax.axis('off')
        header = ['Pair', 'b (1✔,2✘)', 'c (1✘,2✔)', 'Statistic', 'p-value', 'Significant']
        rows = []
        for pair_key in [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]:
            key = f"{pair_key[0]}_vs_{pair_key[1]}"
            if key in mcnemar_results:
                res = mcnemar_results[key]
                rows.append([
                    f"{pair_key[0].upper()} vs {pair_key[1].upper()}",
                    res['b_model1_only'],
                    res['c_model2_only'],
                    f"{res['statistic']:.3f}",
                    f"{res['p_value']:.4f}",
                    'Yes' if res['significant'] else 'No'
                ])

        the_table = ax.table(cellText=rows, colLabels=header, loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.4)
        ax.set_title("McNemar's Test (paired) Results")

        # 4. ROC curves
        ax = axes[1, 1]
        for variant in self.variants:
            if variant in self.predictions:
                y_true = self.predictions[variant]['y_true']
                y_pred_proba = self.predictions[variant]['y_pred_proba']
                if y_pred_proba.ndim > 1:
                    y_pred_proba_plot = y_pred_proba
                else:
                    y_pred_proba_plot = y_pred_proba
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba_plot)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{variant.upper()} (AUC = {roc_auc:.3f})", linewidth=2)
                except Exception:
                    continue
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Switch Model Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        output_path = self.viz_path / 'statistical_analysis_sw.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")

        plt.show()
    
    def generate_report(self, bootstrap_results, mcnemar_results):
        """Generate a comprehensive statistical analysis report (accuracy and McNemar)."""
        print("\nGenerating statistical analysis report...")
        
        report = []
        report.append("=" * 80)
        report.append("SWITCH MODEL STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        for variant in self.variants:
            if variant in self.results:
                report.append(f"\n{variant.upper()} Model:")
                report.append(f"  Accuracy: {self.results[variant]['accuracy']:.4f}")
        
        # Bootstrap confidence intervals
        report.append("\n2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        report.append("-" * 40)
        for variant, results in bootstrap_results.items():
            report.append(f"\n{variant.upper()} Model:")
            report.append(f"  Accuracy: {results['accuracy_mean']:.3f} "
                         f"({results['accuracy_ci'][0]:.3f}, {results['accuracy_ci'][1]:.3f})")
            if not np.isnan(results.get('auc_mean', np.nan)):
                report.append(f"  ROC-AUC: {results['auc_mean']:.3f} "
                             f"({results['auc_ci'][0]:.3f}, {results['auc_ci'][1]:.3f})")
            else:
                report.append("  ROC-AUC: NA (insufficient class variation)")

        # McNemar test results
        report.append("\n3. MCNEMAR'S TEST FOR PAIRED COMPARISON")
        report.append("-" * 40)
        for comparison, results in mcnemar_results.items():
            report.append(f"\n{comparison.replace('_', ' ').upper()}:")
            report.append(f"  Contingency: a={results['a_both_correct']}, b={results['b_model1_only']}, c={results['c_model2_only']}, d={results['d_both_wrong']}")
            report.append(f"  McNemar statistic: {results['statistic']:.3f}")
            report.append(f"  P-value: {results['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if results['significant'] else 'No'}")
        
        # Conclusions
        report.append("\n4. CONCLUSIONS")
        report.append("-" * 40)
        
        # Summary of significant differences
        sig_diffs = []
        for comparison, results in mcnemar_results.items():
            if results['significant']:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (McNemar)")
        
        if sig_diffs:
            report.append("\nStatistically Significant Differences Found:")
            for diff in sig_diffs:
                report.append(f"  • {diff}")
        else:
            report.append("\nNo statistically significant differences found between model variants by McNemar's test.")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.viz_path / 'statistical_analysis_sw_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("Starting Switch Model Statistical Analysis")
        print("=" * 60)
        
        # Load models and data
        self.load_models_and_data()
        
        # Prepare test sets
        self.prepare_test_sets()
        
        if len(self.predictions) < 2:
            print("\n⚠ Warning: Need at least 2 models for comparison")
            print("Note: Switch models may use the same base model across variants.")
            print("Consider running variant-specific training if needed.")
            return
        
        # Run statistical tests
        bootstrap_results = self.bootstrap_confidence_intervals(n_bootstrap=1000)
        mcnemar_results = self.pairwise_mcnemar_tests()
        
        # Visualize results
        self.visualize_results(bootstrap_results, mcnemar_results)
        
        # Generate report
        self.generate_report(bootstrap_results, mcnemar_results)
        
        print("\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    analyzer = SWStatisticalAnalysis()
    analyzer.run_analysis()
