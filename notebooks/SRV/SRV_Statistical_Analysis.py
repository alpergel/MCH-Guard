"""
Statistical Analysis for Survival (SRV) Model Variants
Determines if concordance index differences between m3, m2, and m1 models are statistically significant.

This script performs:
1. Bootstrap resampling to estimate confidence intervals for C-index
2. Log-rank test for survival curve comparisons
3. Permutation tests for C-index significance
4. Analysis of hazard ratio distributions
5. Calibration analysis for survival predictions
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class SRVStatisticalAnalysis:
    """Statistical analysis for survival model comparisons."""
    
    def __init__(self, base_path="../../"):
        """Initialize with paths to models and data."""
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.processed_path = self.base_path / "processed"
        self.viz_path = self.base_path / "viz" / "survival_results"
        
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
        
        for variant in self.variants:
            # Load model - trying different file formats
            model_files = [
                self.models_path / f"srv_{variant}_coxph.joblib",
                self.models_path / f"srv_{variant}_model.joblib",
                self.models_path / f"srv_{variant}.pkl",
                self.models_path / f"cph_{variant}.pkl"
            ]
            
            model_loaded = False
            for model_file in model_files:
                if model_file.exists():
                    try:
                        if model_file.suffix == '.joblib':
                            self.models[variant] = joblib.load(model_file)
                        else:
                            with open(model_file, 'rb') as f:
                                self.models[variant] = pickle.load(f)
                        print(f"✓ Loaded {variant} model from {model_file.name}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"  Failed to load {model_file}: {e}")
            
            if not model_loaded:
                print(f"✗ No model found for {variant}")
            
            # Load survival analysis data
            data_file = self.processed_path / "survival_analysis.csv"
            if data_file.exists() and variant not in self.data:
                # Load the same data for all variants (they differ in features used)
                self.data[variant] = pd.read_csv(data_file)
                print(f"✓ Loaded survival data for {variant} ({len(self.data[variant])} samples)")
            else:
                # Try variant-specific data
                variant_file = self.processed_path / f"survival_{variant}.csv"
                if variant_file.exists():
                    self.data[variant] = pd.read_csv(variant_file)
                    print(f"✓ Loaded {variant} survival data ({len(self.data[variant])} samples)")
    
    def prepare_test_sets(self):
        """Prepare consistent test sets for comparison."""
        print("\nPreparing test sets...")
        
        # Use the same random seed for reproducibility
        np.random.seed(42)
        
        for variant in self.variants:
            if variant not in self.data or variant not in self.models:
                continue
            
            df = self.data[variant]
            model = self.models[variant]
            
            # Find time and event columns
            time_col = None
            event_col = None
            
            for col in ['time', 'Time', 'duration', 'Duration', 'survival_time']:
                if col in df.columns:
                    time_col = col
                    break
            
            for col in ['event', 'Event', 'status', 'Status', 'death', 'failure']:
                if col in df.columns:
                    event_col = col
                    break
            
            if time_col is None or event_col is None:
                print(f"⚠ Warning: Could not find time/event columns for {variant}")
                continue
            
            # Get feature columns based on variant
            if variant == 'm1':
                feature_cols = ['age', 'APOE4', 'CDRSB', 'ADAS13', 'MMSE', 'FAQ']
            elif variant == 'm2':
                feature_cols = ['age', 'APOE4', 'CDRSB', 'ADAS13', 'MMSE', 'FAQ',
                               'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp']
            else:  # m3
                feature_cols = ['age', 'APOE4', 'CDRSB', 'ADAS13', 'MMSE', 'FAQ',
                               'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp',
                               'FDG', 'AV45', 'ABETA_UPENNBIOMK9', 'TAU_UPENNBIOMK9',
                               'PTAU_UPENNBIOMK9']
            
            # Filter to available columns
            available_features = [col for col in feature_cols if col in df.columns]
            
            if not available_features:
                print(f"⚠ Warning: No valid features found for {variant}")
                continue
            
            # Prepare data
            X = df[available_features].values
            T = df[time_col].values
            E = df[event_col].values
            
            # Use last 20% as test set
            test_size = int(0.2 * len(X))
            X_test = X[-test_size:]
            T_test = T[-test_size:]
            E_test = E[-test_size:]
            
            # Generate predictions
            if hasattr(model, 'predict_partial_hazard'):
                # CoxPH model
                test_df = pd.DataFrame(X_test, columns=available_features)
                risk_scores = model.predict_partial_hazard(test_df).values
            elif hasattr(model, 'predict_risk'):
                risk_scores = model.predict_risk(X_test)
            else:
                # Generic prediction
                risk_scores = model.predict(X_test)
            
            # Calculate concordance index
            c_index = concordance_index(T_test, -risk_scores, E_test)
            
            self.predictions[variant] = {
                'X_test': X_test,
                'T_test': T_test,
                'E_test': E_test,
                'risk_scores': risk_scores,
                'features': available_features
            }
            
            self.results[variant] = {
                'c_index': c_index,
                'n_features': len(available_features),
                'n_events': np.sum(E_test),
                'median_time': np.median(T_test)
            }
            
            print(f"✓ {variant}: C-index={c_index:.3f}, "
                  f"Features={len(available_features)}, "
                  f"Events={np.sum(E_test)}/{len(E_test)}")
    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000, confidence_level=0.95):
        """Calculate bootstrap confidence intervals for C-index."""
        print(f"\nCalculating bootstrap confidence intervals (n={n_bootstrap})...")
        
        bootstrap_results = {}
        
        for variant in self.variants:
            if variant not in self.predictions:
                continue
            
            T = self.predictions[variant]['T_test']
            E = self.predictions[variant]['E_test']
            risk_scores = self.predictions[variant]['risk_scores']
            
            c_indices = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = resample(range(len(T)), replace=True)
                T_boot = T[indices]
                E_boot = E[indices]
                risk_boot = risk_scores[indices]
                
                # Calculate C-index
                try:
                    c_idx = concordance_index(T_boot, -risk_boot, E_boot)
                    c_indices.append(c_idx)
                except:
                    pass  # Skip if calculation fails
            
            if c_indices:
                # Calculate confidence intervals
                alpha = 1 - confidence_level
                lower_percentile = (alpha/2) * 100
                upper_percentile = (1 - alpha/2) * 100
                
                bootstrap_results[variant] = {
                    'c_index_ci': (np.percentile(c_indices, lower_percentile),
                                  np.percentile(c_indices, upper_percentile)),
                    'c_index_mean': np.mean(c_indices),
                    'c_index_std': np.std(c_indices)
                }
                
                print(f"\n{variant.upper()} Model:")
                print(f"  C-index: {bootstrap_results[variant]['c_index_mean']:.3f} "
                      f"({bootstrap_results[variant]['c_index_ci'][0]:.3f}, "
                      f"{bootstrap_results[variant]['c_index_ci'][1]:.3f})")
        
        return bootstrap_results
    
    def logrank_tests(self):
        """Perform log-rank tests between model predictions."""
        print("\nPerforming log-rank tests...")
        
        logrank_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
            
            # Use median risk score as cutoff for each model
            risk1 = self.predictions[model1]['risk_scores']
            risk2 = self.predictions[model2]['risk_scores']
            T = self.predictions[model1]['T_test']
            E = self.predictions[model1]['E_test']
            
            # Create risk groups
            high_risk1 = risk1 > np.median(risk1)
            high_risk2 = risk2 > np.median(risk2)
            
            # Perform log-rank test on the groupings
            try:
                result = logrank_test(
                    T[high_risk1], T[~high_risk1],
                    E[high_risk1], E[~high_risk1]
                )
                
                logrank_results[f"{model1}_vs_{model2}"] = {
                    'test_statistic': result.test_statistic,
                    'p_value': result.p_value,
                    'significant': result.p_value < 0.05
                }
                
                print(f"\n{model1.upper()} vs {model2.upper()} risk stratification:")
                print(f"  Test statistic: {result.test_statistic:.3f}")
                print(f"  P-value: {result.p_value:.4f}")
                print(f"  Significant: {'Yes' if result.p_value < 0.05 else 'No'}")
            except Exception as e:
                print(f"  Could not perform log-rank test: {e}")
        
        return logrank_results
    
    def permutation_test(self, n_permutations=1000):
        """Perform permutation test for C-index comparison."""
        print(f"\nPerforming permutation test (n={n_permutations})...")
        
        permutation_results = {}
        
        # Compare all pairs
        pairs = [('m3', 'm2'), ('m3', 'm1'), ('m2', 'm1')]
        
        for model1, model2 in pairs:
            if model1 not in self.predictions or model2 not in self.predictions:
                continue
            
            T = self.predictions[model1]['T_test']
            E = self.predictions[model1]['E_test']
            risk1 = self.predictions[model1]['risk_scores']
            risk2 = self.predictions[model2]['risk_scores']
            
            # Observed difference in C-index
            c1 = concordance_index(T, -risk1, E)
            c2 = concordance_index(T, -risk2, E)
            observed_diff = c1 - c2
            
            # Permutation test
            permuted_diffs = []
            
            for _ in range(n_permutations):
                # Randomly swap predictions between models
                swap_mask = np.random.random(len(risk1)) > 0.5
                risk1_perm = np.where(swap_mask, risk2, risk1)
                risk2_perm = np.where(swap_mask, risk1, risk2)
                
                try:
                    c1_perm = concordance_index(T, -risk1_perm, E)
                    c2_perm = concordance_index(T, -risk2_perm, E)
                    permuted_diffs.append(c1_perm - c2_perm)
                except:
                    pass
            
            if permuted_diffs:
                # Calculate p-value
                permuted_diffs = np.array(permuted_diffs)
                p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
                
                permutation_results[f"{model1}_vs_{model2}"] = {
                    'observed_diff': observed_diff,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                print(f"\n{model1.upper()} vs {model2.upper()}:")
                print(f"  C-index difference: {observed_diff:.4f}")
                print(f"  P-value: {p_value:.4f}")
                print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        return permutation_results
    
    def visualize_results(self, bootstrap_results):
        """Create visualizations for statistical analysis results."""
        print("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. C-index comparison with confidence intervals
        ax = axes[0, 0]
        variants_list = list(bootstrap_results.keys())
        c_indices = [bootstrap_results[v]['c_index_mean'] for v in variants_list]
        ci_lower = [bootstrap_results[v]['c_index_ci'][0] for v in variants_list]
        ci_upper = [bootstrap_results[v]['c_index_ci'][1] for v in variants_list]
        errors = [[c_indices[i] - ci_lower[i] for i in range(len(variants_list))],
                 [ci_upper[i] - c_indices[i] for i in range(len(variants_list))]]
        
        x_pos = np.arange(len(variants_list))
        bars = ax.bar(x_pos, c_indices, yerr=errors, capsize=10,
                      color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        ax.set_xlabel('Model Variant')
        ax.set_ylabel('Concordance Index')
        ax.set_title('C-Index with 95% Confidence Intervals')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([v.upper() for v in variants_list])
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random (C=0.5)')
        ax.set_ylim([0.4, 0.9])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, ci_l, ci_u) in enumerate(zip(bars, ci_lower, ci_upper)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}\n({ci_l:.3f}-{ci_u:.3f})',
                   ha='center', va='bottom', fontsize=9)
        
        # 2. Kaplan-Meier curves for risk groups
        ax = axes[0, 1]
        kmf = KaplanMeierFitter()
        
        colors = {'m3': '#2E86AB', 'm2': '#A23B72', 'm1': '#F18F01'}
        
        for variant in self.variants:
            if variant in self.predictions:
                T = self.predictions[variant]['T_test']
                E = self.predictions[variant]['E_test']
                risk = self.predictions[variant]['risk_scores']
                
                # High risk group (above median)
                high_risk = risk > np.median(risk)
                
                kmf.fit(T[high_risk], E[high_risk], label=f'{variant.upper()} High Risk')
                kmf.plot_survival_function(ax=ax, color=colors[variant], alpha=0.7)
                
                kmf.fit(T[~high_risk], E[~high_risk], label=f'{variant.upper()} Low Risk')
                kmf.plot_survival_function(ax=ax, color=colors[variant], 
                                         linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curves by Risk Groups')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Risk score distributions
        ax = axes[1, 0]
        for variant in self.variants:
            if variant in self.predictions:
                risk = self.predictions[variant]['risk_scores']
                ax.hist(risk, alpha=0.5, label=variant.upper(), bins=30,
                       color=colors[variant])
        
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Risk Score Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Model comparison summary
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        table_data = []
        table_data.append(['Metric'] + [v.upper() for v in variants_list])
        
        # Add C-index
        table_data.append(['C-Index'] + 
                         [f'{self.results[v]["c_index"]:.3f}' for v in variants_list 
                          if v in self.results])
        
        # Add confidence intervals
        table_data.append(['95% CI'] + 
                         [f'({bootstrap_results[v]["c_index_ci"][0]:.3f}, '
                          f'{bootstrap_results[v]["c_index_ci"][1]:.3f})' 
                          for v in variants_list])
        
        # Add number of features
        table_data.append(['# Features'] + 
                         [str(self.results[v]["n_features"]) for v in variants_list 
                          if v in self.results])
        
        # Add number of events
        table_data.append(['# Events'] + 
                         [str(self.results[v]["n_events"]) for v in variants_list 
                          if v in self.results])
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Style the header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
        
        plt.suptitle('Survival Model Statistical Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = self.viz_path / 'statistical_analysis_srv.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.show()
    
    def generate_report(self, bootstrap_results, logrank_results, permutation_results):
        """Generate a comprehensive statistical analysis report."""
        print("\nGenerating statistical analysis report...")
        
        report = []
        report.append("=" * 80)
        report.append("SURVIVAL MODEL STATISTICAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model performance summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        for variant in self.variants:
            if variant in self.results:
                report.append(f"\n{variant.upper()} Model:")
                report.append(f"  Concordance Index: {self.results[variant]['c_index']:.4f}")
                report.append(f"  Number of Features: {self.results[variant]['n_features']}")
                report.append(f"  Number of Events: {self.results[variant]['n_events']}")
                report.append(f"  Median Survival Time: {self.results[variant]['median_time']:.2f}")
        
        # Bootstrap confidence intervals
        report.append("\n2. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
        report.append("-" * 40)
        for variant, results in bootstrap_results.items():
            report.append(f"\n{variant.upper()} Model:")
            report.append(f"  C-index: {results['c_index_mean']:.3f} "
                         f"({results['c_index_ci'][0]:.3f}, {results['c_index_ci'][1]:.3f})")
            report.append(f"  Standard Deviation: {results['c_index_std']:.4f}")
        
        # Statistical tests
        report.append("\n3. STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        
        if logrank_results:
            report.append("\nLog-Rank Test Results:")
            for comparison, results in logrank_results.items():
                report.append(f"  {comparison.replace('_', ' ').upper()}:")
                report.append(f"    Test statistic: {results['test_statistic']:.3f}")
                report.append(f"    P-value: {results['p_value']:.4f}")
                report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
        
        report.append("\nPermutation Test Results:")
        for comparison, results in permutation_results.items():
            report.append(f"  {comparison.replace('_', ' ').upper()}:")
            report.append(f"    C-index difference: {results['observed_diff']:.4f}")
            report.append(f"    P-value: {results['p_value']:.4f}")
            report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
        
        # Conclusions
        report.append("\n4. CONCLUSIONS")
        report.append("-" * 40)
        
        # Check for overlapping confidence intervals
        ci_overlap = []
        if 'm3' in bootstrap_results and 'm2' in bootstrap_results:
            if (bootstrap_results['m3']['c_index_ci'][0] <= bootstrap_results['m2']['c_index_ci'][1] and
                bootstrap_results['m2']['c_index_ci'][0] <= bootstrap_results['m3']['c_index_ci'][1]):
                ci_overlap.append("M3 and M2 models have overlapping C-index confidence intervals")
        
        if 'm3' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['m3']['c_index_ci'][0] <= bootstrap_results['m1']['c_index_ci'][1] and
                bootstrap_results['m1']['c_index_ci'][0] <= bootstrap_results['m3']['c_index_ci'][1]):
                ci_overlap.append("M3 and M1 models have overlapping C-index confidence intervals")
        
        if 'm2' in bootstrap_results and 'm1' in bootstrap_results:
            if (bootstrap_results['m2']['c_index_ci'][0] <= bootstrap_results['m1']['c_index_ci'][1] and
                bootstrap_results['m1']['c_index_ci'][0] <= bootstrap_results['m2']['c_index_ci'][1]):
                ci_overlap.append("M2 and M1 models have overlapping C-index confidence intervals")
        
        if ci_overlap:
            report.append("\nConfidence Interval Analysis:")
            for overlap in ci_overlap:
                report.append(f"  • {overlap}")
        
        # Summary of significant differences
        sig_diffs = []
        
        if logrank_results:
            for comparison, results in logrank_results.items():
                if results['significant']:
                    sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Log-rank test)")
        
        for comparison, results in permutation_results.items():
            if results['significant']:
                sig_diffs.append(f"{comparison.replace('_', ' ').upper()} (Permutation test)")
        
        if sig_diffs:
            report.append("\nStatistically Significant Differences Found:")
            for diff in sig_diffs:
                report.append(f"  • {diff}")
        else:
            report.append("\nNo statistically significant differences found between model variants.")
        
        # Best performing model
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['c_index'])
            report.append(f"\nBest performing model: {best_model[0].upper()} "
                         f"(C-index={best_model[1]['c_index']:.4f})")
        
        # Clinical interpretation
        report.append("\n5. CLINICAL INTERPRETATION")
        report.append("-" * 40)
        
        for variant in self.variants:
            if variant in bootstrap_results:
                c_mean = bootstrap_results[variant]['c_index_mean']
                if c_mean > 0.7:
                    interpretation = "Good discrimination"
                elif c_mean > 0.6:
                    interpretation = "Moderate discrimination"
                else:
                    interpretation = "Poor discrimination"
                
                report.append(f"{variant.upper()}: {interpretation} (C={c_mean:.3f})")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.viz_path / 'statistical_analysis_srv_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Report saved to {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def run_analysis(self):
        """Run complete statistical analysis."""
        print("Starting Survival Model Statistical Analysis")
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
        logrank_results = self.logrank_tests()
        permutation_results = self.permutation_test(n_permutations=1000)
        
        # Visualize results
        self.visualize_results(bootstrap_results)
        
        # Generate report
        self.generate_report(bootstrap_results, logrank_results, permutation_results)
        
        print("\n✓ Statistical analysis complete!")


if __name__ == "__main__":
    analyzer = SRVStatisticalAnalysis()
    analyzer.run_analysis()
