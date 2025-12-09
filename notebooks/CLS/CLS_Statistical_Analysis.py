"""
Statistical Analysis for Classification (CLS) Model Variants
Determines if accuracy differences between m3, m2, and m1 models are statistically significant.

This script performs:
1. Bootstrap resampling to estimate confidence intervals
2. McNemar's test for paired binary classification comparisons
3. Permutation tests for significance
4. Friedman test for multiple related samples
5. Post-hoc Nemenyi test for pairwise comparisons
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class CLSStatisticalAnalysis:
    """Statistical analysis for classification model comparisons."""
    
    def __init__(self, base_path="../../"):
        """Initialize with paths to models and data."""
        self.base_path = Path(base_path) if base_path is not None else Path(__file__).resolve().parents[2]
        self.models_path = self.base_path / "models"
        self.processed_path = self.base_path / "processed"
        self.viz_path = self.base_path / "viz" / "classification_results"
        
        # Model variants
        self.variants = ['m3', 'm2', 'm1']
        self.models = {}
        self.data = {}
        self.predictions = {}
        self.results = {}
        
    def load_models_and_data(self):
        """Load trained models and test data for each variant."""
        print("Loading models and data...")
        
        for variant in self.variants:
            # Load model
            model_file = self.models_path / f"cls_{variant}_model.joblib"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[variant] =  joblib.load(f)
                print(f"✓ Loaded {variant} model")
            else:
                print(f"✗ Model file not found: {model_file}")
                
            # Load data
            data_file = self.processed_path / f"CLS_{variant}_test.csv"
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
            
            # Assuming 'stable' column is the target
            if 'MCH_pos' in df.columns:
                # Get features and target
                feature_cols = [col for col in df.columns if col not in 
                               ['MCH_pos']]
                
                X = df[feature_cols].values
                y = df['MCH_pos'].values
                
                # Use the entire test dataset (these are already test sets)
                X_test = X
                y_test = y
                
                # Generate predictions
                if variant in self.models:
                    model = self.models[variant]
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    self.predictions[variant] = {
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba,
                        'X_test': X_test
                    }
                    
                    # Calculate metrics
                    self.results[variant] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]),
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred)
                    }
                    
                    print(f"✓ {variant}: Accuracy={self.results[variant]['accuracy']:.3f}, "
                          f"AUC={self.results[variant]['roc_auc']:.3f}")
                    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap confidence intervals for model metrics."""
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
            f1s = []
            precisions = []
            recalls = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = resample(range(len(y_true)), replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                y_pred_proba_boot = y_pred_proba[indices]
                
                # Calculate metrics
                accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
                if len(np.unique(y_true_boot)) > 1:  # Check if both classes present
                    aucs.append(roc_auc_score(y_true_boot, y_pred_proba_boot[:, 1]))
                    f1s.append(f1_score(y_true_boot, y_pred_boot))
                    precisions.append(precision_score(y_true_boot, y_pred_boot))
                    recalls.append(recall_score(y_true_boot, y_pred_boot))
            
            # Calculate confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            bootstrap_results[variant] = {
                'accuracy_ci': (np.percentile(accuracies, lower_percentile),
                               np.percentile(accuracies, upper_percentile)),
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'auc_ci': (np.percentile(aucs, lower_percentile),
                          np.percentile(aucs, upper_percentile)) if aucs else (0, 0),
                'auc_mean': np.mean(aucs) if aucs else 0,
                'f1_ci': (np.percentile(f1s, lower_percentile),
                         np.percentile(f1s, upper_percentile)) if f1s else (0, 0),
                'f1_mean': np.mean(f1s) if f1s else 0,
                'precision_ci': (np.percentile(precisions, lower_percentile),
                                np.percentile(precisions, upper_percentile)) if precisions else (0, 0),
                'precision_mean': np.mean(precisions) if precisions else 0,
                'recall_ci': (np.percentile(recalls, lower_percentile),
                              np.percentile(recalls, upper_percentile)) if recalls else (0, 0),
                'recall_mean': np.mean(recalls) if recalls else 0
            }
            
            print(f"\n{variant.upper()} Model:")
            print(f"  Accuracy: {bootstrap_results[variant]['accuracy_mean']:.3f} "
                  f"({bootstrap_results[variant]['accuracy_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['accuracy_ci'][1]:.3f})")
            print(f"  AUC: {bootstrap_results[variant]['auc_mean']:.3f} "
                  f"({bootstrap_results[variant]['auc_ci'][0]:.3f}, "
                  f"{bootstrap_results[variant]['auc_ci'][1]:.3f})")
            
        return bootstrap_results
    
    def mcnemar_test(self):
        """Perform McNemar's test for pairwise model comparisons."""
        print("\nPerforming McNemar's test for pairwise comparisons...")
        
        mcnemar_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
                
            y_true = self.predictions[model1]['y_true']
            pred1 = self.predictions[model1]['y_pred']
            pred2 = self.predictions[model2]['y_pred']
            
            # Create contingency table
            # Both correct, model1 correct/model2 wrong, model1 wrong/model2 correct, both wrong
            correct1_correct2 = np.sum((pred1 == y_true) & (pred2 == y_true))
            correct1_wrong2 = np.sum((pred1 == y_true) & (pred2 != y_true))
            wrong1_correct2 = np.sum((pred1 != y_true) & (pred2 == y_true))
            wrong1_wrong2 = np.sum((pred1 != y_true) & (pred2 != y_true))
            
            # McNemar's test uses only discordant pairs
            contingency_table = [[correct1_correct2, correct1_wrong2],
                                [wrong1_correct2, wrong1_wrong2]]
            
            # Perform test
            result = mcnemar(contingency_table, exact=True)
            
            mcnemar_results[f"{model1}_vs_{model2}"] = {
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'contingency_table': contingency_table,
                'significant': result.pvalue < 0.05
            }
            
            print(f"\n{model1.upper()} vs {model2.upper()}:")
            print(f"  McNemar statistic: {result.statistic:.3f}")
            print(f"  P-value: {result.pvalue:.4f}")
            print(f"  Significant difference: {'Yes' if result.pvalue < 0.05 else 'No'}")
            
        return mcnemar_results
    
    def permutation_test(self, n_permutations=1000):
        """Perform permutation test for model comparison."""
        print(f"\nPerforming permutation test (n={n_permutations})...")
        
        permutation_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
                
            y_true = self.predictions[model1]['y_true']
            pred1 = self.predictions[model1]['y_pred']
            pred2 = self.predictions[model2]['y_pred']
            
            # Observed difference in accuracy
            acc1 = accuracy_score(y_true, pred1)
            acc2 = accuracy_score(y_true, pred2)
            observed_diff = acc1 - acc2
            
            # Permutation test
            permuted_diffs = []
            
            for _ in range(n_permutations):
                # Randomly swap predictions between models
                swap_mask = np.random.random(len(pred1)) > 0.5
                pred1_perm = np.where(swap_mask, pred2, pred1)
                pred2_perm = np.where(swap_mask, pred1, pred2)
                
                acc1_perm = accuracy_score(y_true, pred1_perm)
                acc2_perm = accuracy_score(y_true, pred2_perm)
                permuted_diffs.append(acc1_perm - acc2_perm)
            
            # Calculate p-value
            permuted_diffs = np.array(permuted_diffs)
            p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
            
            permutation_results[f"{model1}_vs_{model2}"] = {
                'observed_diff': observed_diff,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            print(f"\n{model1.upper()} vs {model2.upper()}:")
            print(f"  Observed difference: {observed_diff:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
        return permutation_results
    
    def friedman_test(self):
        """Perform Friedman test for comparing all three models."""
        print("\nPerforming Friedman test for all models...")
        
        # Prepare data for Friedman test
        # We need matched samples, so we'll use cross-validation scores
        # For simplicity, we'll use bootstrap samples as "blocks"
        
        n_blocks = 100
        scores = {variant: [] for variant in self.variants}
        
        for _ in range(n_blocks):
            # Create a bootstrap sample
            indices = resample(range(len(self.predictions['m3']['y_true'])), replace=True)
            
            for variant in self.variants:
                if variant in self.predictions:
                    y_true = self.predictions[variant]['y_true'][indices]
                    y_pred = self.predictions[variant]['y_pred'][indices]
                    scores[variant].append(accuracy_score(y_true, y_pred))
        
        # Perform Friedman test
        scores_array = [scores[variant] for variant in self.variants if variant in scores]
        
        if len(scores_array) == 3:
            statistic, p_value = friedmanchisquare(*scores_array)
            
            print(f"Friedman statistic: {statistic:.3f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Significant difference among all models: {'Yes' if p_value < 0.05 else 'No'}")
            
            # If significant, perform post-hoc Wilcoxon signed-rank tests
            if p_value < 0.05:
                print("\nPost-hoc Wilcoxon signed-rank tests:")
                pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]
                
                for model1, model2 in pairs:
                    if model1 in scores and model2 in scores:
                        stat, p = wilcoxon(scores[model1], scores[model2])
                        # Apply Bonferroni correction
                        corrected_p = p * len(pairs)
                        print(f"  {model1.upper()} vs {model2.upper()}: "
                              f"p={p:.4f} (corrected={corrected_p:.4f}), "
                              f"significant={'Yes' if corrected_p < 0.05 else 'No'}")
            
            return {'statistic': statistic, 'p_value': p_value}
        
        return None
    
    def visualize_results(self, bootstrap_results):
        """Create visualizations for statistical analysis results."""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Accuracy comparison with confidence intervals
        ax = axes[0, 0]
        variants_list = list(bootstrap_results.keys())
        accuracies = [bootstrap_results[v]['accuracy_mean'] for v in variants_list]
        ci_lower = [bootstrap_results[v]['accuracy_ci'][0] for v in variants_list]
        ci_upper = [bootstrap_results[v]['accuracy_ci'][1] for v in variants_list]
        errors = [[accuracies[i] - ci_lower[i] for i in range(len(variants_list))],
                 [ci_upper[i] - accuracies[i] for i in range(len(variants_list))]]
        
        x_pos = np.arange(len(variants_list))
        ax.bar(x_pos, accuracies, yerr=errors, capsize=10, 
               color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (v, ci_l, ci_u) in enumerate(zip(accuracies, ci_lower, ci_upper)):
            ax.text(i, v + 0.01, f'{v:.3f}\n({ci_l:.3f}-{ci_u:.3f})', 
                   ha='center', va='bottom', fontsize=9)
        
        # 2. Precision-Recall-F1 comparison (top-right)
        ax = axes[0, 1]
        metrics_data = []
        for v in variants_list:
            if v in bootstrap_results:
                metrics_data.append([
                    bootstrap_results[v]['precision_mean'],
                    bootstrap_results[v]['recall_mean'],
                    bootstrap_results[v]['f1_mean']
                ])
        if metrics_data:
            metrics_array = np.array(metrics_data).T
            width = 0.25
            ax.bar(x_pos - width, metrics_array[0], width, label='Precision', alpha=0.7)
            ax.bar(x_pos,         metrics_array[1], width, label='Recall',    alpha=0.7)
            ax.bar(x_pos + width, metrics_array[2], width, label='F1',        alpha=0.7)
            ax.set_xlabel('Model Variant')
            ax.set_ylabel('Score')
            ax.set_title('Precision, Recall, and F1 Scores')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([v.upper() for v in variants_list])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Pairwise differences
        ax = axes[1, 0]
        if len(variants_list) == 3:
            differences = {
                'M3-M2': accuracies[0] - accuracies[1],
                'M3-M1': accuracies[0] - accuracies[2],
                'M2-M1': accuracies[1] - accuracies[2]
            }
            
            bars = ax.bar(range(len(differences)), list(differences.values()),
                          color=['#4CAF50' if d > 0 else '#F44336' for d in differences.values()],
                          alpha=0.7)
            ax.set_xlabel('Model Comparison')
            ax.set_ylabel('Accuracy Difference')
            ax.set_title('Pairwise Accuracy Differences')
            ax.set_xticks(range(len(differences)))
            ax.set_xticklabels(list(differences.keys()), rotation=45)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, differences.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 4. Summary statistics table
        # ax = axes[1, 1]
        # ax.axis('tight')
        # ax.axis('off')
        
        # # Create summary table
        # table_data = []
        # table_data.append(['Metric'] + [v.upper() for v in variants_list])
        # table_data.append(['Accuracy'] + [f'{self.results[v]["accuracy"]:.3f}' for v in variants_list if v in self.results])
        # table_data.append(['ROC-AUC'] + [f'{self.results[v]["roc_auc"]:.3f}' for v in variants_list if v in self.results])
        # table_data.append(['F1 Score'] + [f'{self.results[v]["f1"]:.3f}' for v in variants_list if v in self.results])
        # table_data.append(['Precision'] + [f'{self.results[v]["precision"]:.3f}' for v in variants_list if v in self.results])
        # table_data.append(['Recall'] + [f'{self.results[v]["recall"]:.3f}' for v in variants_list if v in self.results])
        
        # table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        # table.auto_set_font_size(False)
        # table.set_fontsize(10)
        # table.scale(1.2, 1.5)
        
        # # Style the header row
        # for i in range(len(table_data[0])):
        #     table[(0, i)].set_facecolor('#E0E0E0')
        #     table[(0, i)].set_text_props(weight='bold')
        
        # 5. ROC curves for all variants (stacked on same axes)
        ax = axes[1, 1]
        for variant in self.variants:
            if variant in self.predictions:
                y_true = self.predictions[variant]['y_true']
                y_pred_proba = self.predictions[variant]['y_pred_proba'][:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{variant.upper()} (AUC = {roc_auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Classification Model Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = self.viz_path / 'statistical_analysis_cls.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
    
    def generate_report(self, bootstrap_results, mcnemar_results, permutation_results, friedman_results):
        """Generate a comprehensive statistical analysis report."""
        print("\nGenerating statistical analysis report...")
        
        report = []
        report.append("=" * 80)
        report.append("CLASSIFICATION MODEL STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        for variant in self.variants:
            if variant in self.results:
                report.append(f"\n{variant.upper()} Model:")
                for metric, value in self.results[variant].items():
                    report.append(f"  {metric.capitalize()}: {value:.4f}")
        
        # Bootstrap confidence intervals
        report.append("\n2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        report.append("-" * 40)
        for variant, results in bootstrap_results.items():
            report.append(f"\n{variant.upper()} Model:")
            report.append(f"  Accuracy: {results['accuracy_mean']:.3f} "
                         f"({results['accuracy_ci'][0]:.3f}, {results['accuracy_ci'][1]:.3f})")
            report.append(f"  ROC-AUC: {results['auc_mean']:.3f} "
                         f"({results['auc_ci'][0]:.3f}, {results['auc_ci'][1]:.3f})")
        
        # Statistical tests
        report.append("\n3. STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        
        report.append("\nMcNemar's Test Results:")
        for comparison, results in mcnemar_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    P-value: {results['p_value']:.4f}")
            report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
        
        report.append("\nPermutation Test Results:")
        for comparison, results in permutation_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    Observed difference: {results['observed_diff']:.4f}")
            report.append(f"    P-value: {results['p_value']:.4f}")
            report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
        
        if friedman_results:
            report.append("\nFriedman Test Result:")
            report.append(f"  Statistic: {friedman_results['statistic']:.3f}")
            report.append(f"  P-value: {friedman_results['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if friedman_results['p_value'] < 0.05 else 'No'}")
        
        # Conclusions
        report.append("\n4. CONCLUSIONS")
        report.append("-" * 40)
        
        # Check for overlapping confidence intervals
        ci_overlap = []
        if 'm3' in bootstrap_results and 'm2' in bootstrap_results:
            if (bootstrap_results['m3']['accuracy_ci'][0] <= bootstrap_results['m2']['accuracy_ci'][1] and
                bootstrap_results['m2']['accuracy_ci'][0] <= bootstrap_results['m3']['accuracy_ci'][1]):
                ci_overlap.append("M3 and M2 models have overlapping confidence intervals")
        
        if 'm3' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['m3']['accuracy_ci'][0] <= bootstrap_results['m1']['accuracy_ci'][1] and
                bootstrap_results['m1']['accuracy_ci'][0] <= bootstrap_results['m3']['accuracy_ci'][1]):
                ci_overlap.append("M3 and M1 models have overlapping confidence intervals")
        
        if 'm2' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['m2']['accuracy_ci'][0] <= bootstrap_results['m1']['accuracy_ci'][1] and
                bootstrap_results['m1']['accuracy_ci'][0] <= bootstrap_results['m2']['accuracy_ci'][1]):
                ci_overlap.append("M2 and M1 models have overlapping confidence intervals")
        
        if ci_overlap:
            report.append("\nConfidence Interval Analysis:")
            for overlap in ci_overlap:
                report.append(f"  • {overlap}")
        
        # Summary of significant differences
        sig_diffs = []
        for test_name, test_results in [("McNemar's test", mcnemar_results), 
                                        ("Permutation test", permutation_results)]:
            for comparison, results in test_results.items():
                if results['significant']:
                    sig_diffs.append(f"{comparison.replace('_', ' ').upper()} ({test_name})")
        
        if sig_diffs:
            report.append("\nStatistically Significant Differences Found:")
            for diff in sig_diffs:
                report.append(f"  • {diff}")
        else:
            report.append("\nNo statistically significant differences found between model variants.")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.viz_path / 'statistical_analysis_cls_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("Starting Classification Model Statistical Analysis")
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
        mcnemar_results = self.mcnemar_test()
        permutation_results = self.permutation_test(n_permutations=1000)
        friedman_results = self.friedman_test() if len(self.predictions) == 3 else None
        
        # Visualize results
        self.visualize_results(bootstrap_results)
        
        # Generate report
        self.generate_report(bootstrap_results, mcnemar_results, 
                            permutation_results, friedman_results)
        
        print("\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    analyzer = CLSStatisticalAnalysis()
    analyzer.run_analysis()
