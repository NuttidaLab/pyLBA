"""
Example usage of pyLBA package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyLBA import LBAModel, LBAParameters


def basic_example():
    """Basic example of LBA model usage."""
    print("=== Basic LBA Example ===")
    
    # Create model
    model = LBAModel()
    
    # Define parameters
    params = LBAParameters(
        A=4.0,           # Start point variability
        b=[6, 10, 20],   # Response thresholds (3 accumulators)
        v=1.0,           # Drift rate
        s=1.0,           # Drift rate SD
        tau=0.0          # Non-decision time
    )
    
    print(f"Parameters: {params}")
    print(f"Valid parameters: {params.validate()}")
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data = model.generate_data(n_trials=500, parameters=params, n_acc=3, seed=42)
    
    print(f"Data shape: {data.shape}")
    print(f"Response distribution:\n{data.groupby('response').size()}")
    print(f"Mean RT by response:\n{data.groupby('response')['rt'].mean()}")
    
    return model, data, params


def fitting_example():
    """Example of model fitting."""
    print("\n=== Model Fitting Example ===")
    
    # Get data from basic example
    model, data, true_params = basic_example()
    
    # Fit model using MCMC
    print("\nFitting model using MCMC...")
    fitted_model = model.fit_mcmc(
        data=data,
        draws=200,
        tune=200,
        chains=2,
        cores=2,
        progressbar=True
    )
    
    # Check results
    import arviz as az
    summary = az.summary(fitted_model.trace)
    print(f"\nParameter estimates:\n{summary[['mean', 'hdi_3%', 'hdi_97%']].round(3)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = fitted_model.predict(n_trials=100, seed=123)
    
    print(f"Prediction summary:\n{predictions.groupby('response')['rt'].mean()}")
    
    return fitted_model, predictions


def custom_priors_example():
    """Example with custom priors."""
    print("\n=== Custom Priors Example ===")
    
    import pymc as pm
    
    # Create model and data
    model = LBAModel()
    
    # Generate data with different parameters
    params = LBAParameters(
        A=3.0,
        b=[8, 12, 16],
        v=[0.5, 1.0, 1.5],  # Different drift rates
        s=0.8,
        tau=0.1
    )
    
    data = model.generate_data(n_trials=300, parameters=params, n_acc=3, seed=99)
    
    # Define custom priors
    custom_priors = {
        'A': pm.Uniform('A', lower=1, upper=8, shape=3),
        'b': pm.Uniform('b', lower=5, upper=20, shape=3),
        'v': pm.Normal('v', mu=1, sigma=0.5, shape=3),
        's': pm.HalfNormal('s', sigma=1, shape=3),
        'tau': pm.Uniform('tau', lower=0, upper=0.5, shape=3)
    }
    
    print("Fitting with custom priors...")
    fitted_model = model.fit_mcmc(
        data=data,
        priors=custom_priors,
        draws=150,
        tune=150,
        chains=2
    )
    
    # Results
    import arviz as az
    summary = az.summary(fitted_model.trace)
    print(f"\nResults with custom priors:\n{summary[['mean', 'hdi_3%', 'hdi_97%']].round(3)}")
    
    return fitted_model


def plot_example():
    """Example of plotting results."""
    print("\n=== Plotting Example ===")
    
    # Generate data
    model = LBAModel()
    params = LBAParameters(A=4, b=[6, 10, 20], v=1, s=1, tau=0)
    data = model.generate_data(n_trials=1000, parameters=params, n_acc=3, seed=42)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # RT histograms by response
    for i, response in enumerate(sorted(data['response'].unique())):
        ax = axes[0, i] if i < 2 else axes[1, i-2]
        subset = data[data['response'] == response]
        ax.hist(subset['rt'], bins=30, alpha=0.7, density=True)
        ax.set_title(f'Response {response} RT Distribution')
        ax.set_xlabel('Response Time')
        ax.set_ylabel('Density')
    
    # Response proportions
    if len(data['response'].unique()) == 3:
        response_counts = data['response'].value_counts().sort_index()
        axes[1, 1].bar(response_counts.index, response_counts.values)
        axes[1, 1].set_title('Response Proportions')
        axes[1, 1].set_xlabel('Response')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('lba_example_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Plots saved as 'lba_example_plots.png'")


if __name__ == "__main__":
    # Run examples
    print("Running pyLBA examples...")
    
    # Basic usage
    basic_example()
    
    # Model fitting
    try:
        fitting_example()
    except Exception as e:
        print(f"Fitting example failed: {e}")
    
    # Custom priors
    try:
        custom_priors_example()
    except Exception as e:
        print(f"Custom priors example failed: {e}")
    
    # Plotting
    try:
        plot_example()
    except Exception as e:
        print(f"Plotting example failed: {e}")
    
    print("\nAll examples completed!")
