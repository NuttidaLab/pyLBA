#!/usr/bin/env python3
"""
MCMC fitting example that replicates the notebook analysis.
This script demonstrates the complete workflow from data generation to model fitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyLBA import LBAModel, LBAParameters


def main():
    """Main function replicating the notebook analysis."""
    print("=" * 60)
    print("pyLBA MCMC Fitting Example")
    print("(Replicating the notebook analysis)")
    print("=" * 60)
    
    # 1. Set up model and parameters (same as notebook)
    print("\n1. Setting up model and parameters...")
    model = LBAModel()
    
    # Use the same parameters as in the original notebook
    true_params = LBAParameters(
        A=4,
        b=[6, 10, 20],
        v=1,
        s=1,
        tau=0
    )
    
    print(f"   True parameters: {true_params}")
    
    # 2. Generate synthetic data (same as notebook)
    print("\n2. Generating synthetic data...")
    np.random.seed(42)  # For reproducibility
    data = model.generate_data(
        n_trials=500,
        parameters=true_params,
        n_acc=3
    )
    
    print(f"   Generated {len(data)} trials")
    print("\n   Response distribution:")
    response_summary = data.groupby('response').agg({
        'rt': ['count', 'mean', 'std']
    }).round(3)
    print(response_summary)
    
    # 3. Attempt MCMC fitting (same approach as notebook)
    print("\n3. Attempting MCMC fitting...")
    
    try:
        # Import PyMC - this will fail if not installed
        import pymc as pm
        import arviz as az
        
        print("   PyMC available, proceeding with MCMC fitting...")
        
        # Custom priors (similar to notebook but more informative)
        print("\n4. Setting up custom priors...")
        custom_priors = {
            'A': pm.Uniform('A', lower=1, upper=10, shape=3),
            'b': pm.Uniform('b', lower=3, upper=30, shape=3),
            'v': pm.Uniform('v', lower=0.1, upper=5, shape=3),
            's': pm.Uniform('s', lower=0.1, upper=5, shape=3),
            'tau': pm.Uniform('tau', lower=0, upper=2, shape=3)
        }
        
        # Fit model
        print("\n5. Fitting model with MCMC...")
        print("   This may take a few minutes...")
        
        fitted_model = model.fit_mcmc(
            data=data,
            priors=custom_priors,
            draws=200,      # Reduced for faster demo
            tune=200,       # Reduced for faster demo
            chains=2,       # Reduced for faster demo
            cores=2,
            progressbar=True
        )
        
        print("   ✅ MCMC fitting completed!")
        
        # 6. Analyze results
        print("\n6. Analyzing results...")
        summary = az.summary(fitted_model.trace)
        print("\n   Parameter estimates:")
        print(summary[['mean', 'hdi_3%', 'hdi_97%']].round(3))
        
        # Compare with true parameters
        print("\n   Comparison with true parameters:")
        param_names = ['A', 'b', 'v', 's', 'tau']
        means = summary['mean']
        
        for param in param_names:
            if param in means.index:
                estimated = means[param]
                true_val = getattr(true_params, param)
                
                if isinstance(true_val, (list, np.ndarray)):
                    print(f"   {param}: Estimated={estimated:.3f}, True={true_val}")
                else:
                    print(f"   {param}: Estimated={estimated:.3f}, True={true_val}")
        
        # 7. Generate predictions
        print("\n7. Generating predictions...")
        predictions = fitted_model.predict(n_trials=100, seed=123)
        
        print("\n   Prediction summary:")
        pred_summary = predictions.groupby('response')['rt'].mean()
        orig_summary = data.groupby('response')['rt'].mean()
        
        for resp in sorted(predictions['response'].unique()):
            pred_rt = pred_summary.get(resp, 0)
            orig_rt = orig_summary.get(resp, 0)
            print(f"   Response {resp}: Predicted={pred_rt:.3f}, Original={orig_rt:.3f}")
        
        # 8. Create diagnostic plots
        print("\n8. Creating diagnostic plots...")
        create_diagnostic_plots(fitted_model.trace, data, predictions)
        
        # 9. Check convergence
        print("\n9. Checking convergence...")
        rhat_max = summary['r_hat'].max()
        ess_min = summary['ess_bulk'].min()
        
        print(f"   Max R-hat: {rhat_max:.3f} (should be < 1.1)")
        print(f"   Min ESS: {ess_min:.0f} (should be > 400)")
        
        if rhat_max < 1.1 and ess_min > 400:
            print("   ✅ Convergence looks good!")
        else:
            print("   ⚠️  Convergence may be questionable, consider longer chains")
        
    except ImportError:
        print("   ❌ PyMC not available!")
        print("   To run MCMC fitting, install PyMC:")
        print("   pip install pymc")
        print("\n   Demonstrating other package features instead...")
        
        # Show what we can do without PyMC
        demonstrate_without_mcmc(model, data, true_params)
    
    except Exception as e:
        print(f"   ❌ MCMC fitting failed: {e}")
        print("   This might be due to numerical issues or insufficient data.")
        print("   Try adjusting the parameters or increasing the number of trials.")
        
        # Show what we can do without successful fitting
        demonstrate_without_mcmc(model, data, true_params)
    
    print("\n" + "=" * 60)
    print("MCMC fitting example completed!")
    print("=" * 60)


def demonstrate_without_mcmc(model, data, true_params):
    """Demonstrate package features without MCMC fitting."""
    print("\n   Demonstrating package features without MCMC...")
    
    # Test different parameter configurations
    print("\n   Testing different parameter configurations:")
    test_configs = [
        ("High threshold", LBAParameters(A=4, b=15, v=1, s=1, tau=0)),
        ("Low threshold", LBAParameters(A=4, b=5, v=1, s=1, tau=0)),
        ("High drift", LBAParameters(A=4, b=8, v=2, s=1, tau=0)),
        ("Low drift", LBAParameters(A=4, b=8, v=0.5, s=1, tau=0)),
    ]
    
    for name, params in test_configs:
        if params.validate():
            sample = model.generate_data(n_trials=100, parameters=params, n_acc=2, seed=42)
            mean_rt = sample['rt'].mean()
            print(f"     {name}: Mean RT = {mean_rt:.3f}s")
        else:
            print(f"     {name}: Invalid parameters")
    
    # Test parameter validation
    print("\n   Testing parameter validation:")
    invalid_params = LBAParameters(A=4, b=3, v=1, s=1, tau=0)  # b <= A
    print(f"     Invalid params (b <= A): {invalid_params.validate()}")
    
    # Show data statistics
    print("\n   Original data statistics:")
    print(f"     Total trials: {len(data)}")
    print(f"     RT range: {data['rt'].min():.3f} - {data['rt'].max():.3f}")
    print(f"     Response distribution: {data['response'].value_counts().to_dict()}")


def create_diagnostic_plots(trace, original_data, predictions):
    """Create diagnostic plots for the fitted model."""
    try:
        import arviz as az
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Posterior plot
        az.plot_posterior(trace, ax=axes[0, 0])
        axes[0, 0].set_title("Posterior Distributions")
        
        # Trace plot
        az.plot_trace(trace, ax=axes[0, 1])
        axes[0, 1].set_title("Trace Plots")
        
        # Data comparison
        for i, resp in enumerate(sorted(original_data['response'].unique())):
            if i < 2:  # Only plot first 2 responses
                ax = axes[1, i]
                
                # Original data
                orig_subset = original_data[original_data['response'] == resp]
                ax.hist(orig_subset['rt'], bins=20, alpha=0.5, label='Original', density=True)
                
                # Predictions
                pred_subset = predictions[predictions['response'] == resp]
                if len(pred_subset) > 0:
                    ax.hist(pred_subset['rt'], bins=20, alpha=0.5, label='Predicted', density=True)
                
                ax.set_title(f'Response {resp}: Data vs Predictions')
                ax.set_xlabel('RT')
                ax.set_ylabel('Density')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('mcmc_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   Diagnostic plots saved as 'mcmc_diagnostics.png'")
        
    except Exception as e:
        print(f"   Could not create diagnostic plots: {e}")


if __name__ == "__main__":
    main()
