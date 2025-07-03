#!/usr/bin/env python3
"""
Complete example demonstrating pyLBA package functionality.
This script replicates the analysis from the original notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyLBA import LBAModel, LBAParameters


def main():
    """Main example function."""
    print("=" * 60)
    print("pyLBA Package Complete Example")
    print("=" * 60)
    
    # 1. Create model and parameters (matching original notebook)
    print("\n1. Creating LBA model and parameters...")
    model = LBAModel()
    
    # Use the same parameters as in the original notebook
    original_params = LBAParameters(
        A=4,
        b=[6, 10, 20],
        v=1,
        s=1,
        tau=0
    )
    
    print(f"   Model: {model.name}")
    print(f"   Parameters: {original_params}")
    print(f"   Parameters valid: {original_params.validate()}")
    
    # 2. Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)  # For reproducibility
    data = model.generate_data(
        n_trials=500,
        parameters=original_params,
        n_acc=3
    )
    
    print(f"   Generated {len(data)} trials")
    print(f"   Response distribution:")
    response_counts = data['response'].value_counts().sort_index()
    for resp, count in response_counts.items():
        print(f"     Response {resp}: {count} trials ({count/len(data)*100:.1f}%)")
    
    print(f"   Mean RT by response:")
    mean_rts = data.groupby('response')['rt'].mean()
    for resp, rt in mean_rts.items():
        print(f"     Response {resp}: {rt:.3f} seconds")
    
    # 3. Visualize the data
    print("\n3. Creating visualizations...")
    create_visualizations(data)
    
    # 4. Test parameter validation
    print("\n4. Testing parameter validation...")
    test_parameter_validation()
    
    # 5. Test different parameter configurations
    print("\n5. Testing different parameter configurations...")
    test_different_configs(model)
    
    # 6. Demonstrate model PDF calculation
    print("\n6. Testing PDF calculation...")
    test_pdf_calculation(model, data, original_params)
    
    # 7. Show model extensibility
    print("\n7. Package structure overview...")
    show_package_structure()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


def create_visualizations(data):
    """Create visualizations of the generated data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # RT histograms by response
    responses = sorted(data['response'].unique())
    colors = ['blue', 'orange', 'green']
    
    for i, response in enumerate(responses):
        if i < 3:  # Only plot first 3 responses
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            subset = data[data['response'] == response]
            ax.hist(subset['rt'], bins=30, alpha=0.7, color=colors[i], density=True)
            ax.set_title(f'Response {response} RT Distribution')
            ax.set_xlabel('Response Time (s)')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
    
    # Response proportions
    response_counts = data['response'].value_counts().sort_index()
    axes[1, 1].bar(response_counts.index, response_counts.values, color=colors[:len(response_counts)])
    axes[1, 1].set_title('Response Proportions')
    axes[1, 1].set_xlabel('Response')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lba_analysis_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("   Visualization saved as 'lba_analysis_example.png'")


def test_parameter_validation():
    """Test parameter validation functionality."""
    test_cases = [
        # (params, expected_valid, description)
        (LBAParameters(A=4, b=8, v=1, s=1, tau=0), True, "Valid parameters"),
        (LBAParameters(A=4, b=3, v=1, s=1, tau=0), False, "Invalid: b <= A"),
        (LBAParameters(A=-1, b=8, v=1, s=1, tau=0), False, "Invalid: negative A"),
        (LBAParameters(A=4, b=8, v=1, s=-1, tau=0), False, "Invalid: negative s"),
        (LBAParameters(A=4, b=8, v=1, s=1, tau=-0.1), False, "Invalid: negative tau"),
    ]
    
    for params, expected, description in test_cases:
        result = params.validate()
        status = "✓" if result == expected else "✗"
        print(f"   {status} {description}: {result}")


def test_different_configs(model):
    """Test different parameter configurations."""
    configs = [
        ("Equal thresholds", LBAParameters(A=4, b=8, v=1, s=1, tau=0)),
        ("Different thresholds", LBAParameters(A=4, b=[6, 10, 20], v=1, s=1, tau=0)),
        ("Different drift rates", LBAParameters(A=4, b=8, v=[0.5, 1.0, 1.5], s=1, tau=0)),
        ("With non-decision time", LBAParameters(A=4, b=8, v=1, s=1, tau=0.1)),
    ]
    
    for name, params in configs:
        try:
            # Generate a small sample to test
            sample_data = model.generate_data(n_trials=50, parameters=params, n_acc=3, seed=42)
            mean_rt = sample_data['rt'].mean()
            print(f"   {name}: Mean RT = {mean_rt:.3f}s")
        except Exception as e:
            print(f"   {name}: Error - {e}")


def test_pdf_calculation(model, data, params):
    """Test PDF calculation functionality."""
    # Take a small sample for testing
    sample_data = data.head(10)
    rt = sample_data['rt'].to_numpy()
    response = sample_data['response'].to_numpy()
    
    try:
        # Calculate PDF
        pdf = model.pdf(rt, response, params, n_acc=3)
        
        # This would normally be evaluated in a PyTensor context
        print(f"   PDF calculation successful for {len(rt)} data points")
        print(f"   PDF tensor shape: {pdf.type if hasattr(pdf, 'type') else 'N/A'}")
        
    except Exception as e:
        print(f"   PDF calculation error: {e}")


def show_package_structure():
    """Show the package structure and capabilities."""
    print("   Package modules:")
    print("   ├── pyLBA/")
    print("   │   ├── __init__.py          # Main package imports")
    print("   │   ├── core.py              # Abstract base classes")
    print("   │   ├── models.py            # LBA model implementation")
    print("   │   ├── fitting.py           # MCMC and EM fitting")
    print("   │   └── utils.py             # Utility functions")
    print("   ├── examples/")
    print("   │   ├── basic_usage.py       # Basic usage examples")
    print("   │   └── demo_notebook.py     # Jupyter notebook demo")
    print("   ├── tests/")
    print("   │   ├── test_lba.py          # Unit tests")
    print("   │   └── conftest.py          # Test configuration")
    print("   └── setup.py                 # Package installation")
    
    print("\n   Key features:")
    print("   • Standardized API for accumulator models")
    print("   • Type-safe parameter management with named tuples")
    print("   • MCMC fitting via PyMC integration")
    print("   • Extensible design for adding new models")
    print("   • Comprehensive test coverage")
    print("   • Data generation and validation utilities")


if __name__ == "__main__":
    main()
