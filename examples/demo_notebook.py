"""
Demonstration notebook for pyLBA package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyLBA import LBAModel, LBAParameters


def create_demo_notebook():
    """Create a demonstration notebook."""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# pyLBA Package Demo\n\n",
                    "This notebook demonstrates the basic usage of the pyLBA package for fitting Linear Ballistic Accumulator models.\n\n",
                    "## Installation\n\n",
                    "```bash\n",
                    "pip install -e .\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from pyLBA import LBAModel, LBAParameters\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Basic Model Creation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create an LBA model\n",
                    "model = LBAModel()\n",
                    "print(f\"Model name: {model.name}\")\n",
                    "print(f\"Model fitted: {model.fitted}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Parameter Definition"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define model parameters\n",
                    "params = LBAParameters(\n",
                    "    A=4.0,           # Start point variability\n",
                    "    b=[6, 10, 20],   # Response thresholds (3 accumulators)\n",
                    "    v=1.0,           # Drift rate\n",
                    "    s=1.0,           # Drift rate SD\n",
                    "    tau=0.0          # Non-decision time\n",
                    ")\n",
                    "\n",
                    "print(f\"Parameters: {params}\")\n",
                    "print(f\"Valid: {params.validate()}\")\n",
                    "print(f\"As dict: {params.to_dict()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Data Generation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Generate synthetic data\n",
                    "np.random.seed(42)\n",
                    "data = model.generate_data(n_trials=500, parameters=params, n_acc=3)\n",
                    "\n",
                    "print(f\"Data shape: {data.shape}\")\n",
                    "print(f\"\\nFirst few rows:\")\n",
                    "print(data.head())\n",
                    "\n",
                    "print(f\"\\nResponse distribution:\")\n",
                    "print(data['response'].value_counts().sort_index())\n",
                    "\n",
                    "print(f\"\\nMean RT by response:\")\n",
                    "print(data.groupby('response')['rt'].mean())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Data Visualization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create visualizations\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n",
                    "\n",
                    "# RT histograms by response\n",
                    "for i, response in enumerate(sorted(data['response'].unique())):\n",
                    "    row = i // 2\n",
                    "    col = i % 2\n",
                    "    ax = axes[row, col]\n",
                    "    \n",
                    "    subset = data[data['response'] == response]\n",
                    "    ax.hist(subset['rt'], bins=30, alpha=0.7, density=True, \n",
                    "            label=f'Response {response}')\n",
                    "    ax.set_title(f'Response {response} RT Distribution')\n",
                    "    ax.set_xlabel('Response Time')\n",
                    "    ax.set_ylabel('Density')\n",
                    "    ax.legend()\n",
                    "\n",
                    "# Response proportions\n",
                    "if len(data['response'].unique()) == 3:\n",
                    "    response_counts = data['response'].value_counts().sort_index()\n",
                    "    axes[1, 1].bar(response_counts.index, response_counts.values)\n",
                    "    axes[1, 1].set_title('Response Proportions')\n",
                    "    axes[1, 1].set_xlabel('Response')\n",
                    "    axes[1, 1].set_ylabel('Count')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Model Fitting (Optional - requires PyMC)\n",
                    "\n",
                    "**Note:** This section requires PyMC to be installed and may take some time to run."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Uncomment to run model fitting\n",
                    "# try:\n",
                    "#     print(\"Fitting model using MCMC...\")\n",
                    "#     fitted_model = model.fit_mcmc(\n",
                    "#         data=data,\n",
                    "#         draws=200,\n",
                    "#         tune=200,\n",
                    "#         chains=2,\n",
                    "#         progressbar=True\n",
                    "#     )\n",
                    "#     \n",
                    "#     # Display results\n",
                    "#     import arviz as az\n",
                    "#     summary = az.summary(fitted_model.trace)\n",
                    "#     print(\"\\nParameter estimates:\")\n",
                    "#     print(summary[['mean', 'hdi_3%', 'hdi_97%']].round(3))\n",
                    "#     \n",
                    "#     # Plot posterior\n",
                    "#     az.plot_posterior(fitted_model.trace, figsize=(10, 6))\n",
                    "#     plt.tight_layout()\n",
                    "#     plt.show()\n",
                    "#     \n",
                    "# except Exception as e:\n",
                    "#     print(f\"Fitting failed: {e}\")\n",
                    "#     print(\"Make sure PyMC is installed: pip install pymc\")\n",
                    "\n",
                    "print(\"Model fitting code is commented out. Uncomment to run.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Parameter Validation Examples"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Valid parameters\n",
                    "valid_params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)\n",
                    "print(f\"Valid params: {valid_params.validate()}\")\n",
                    "\n",
                    "# Invalid parameters (b <= A)\n",
                    "invalid_params1 = LBAParameters(A=4, b=3, v=1, s=1, tau=0)\n",
                    "print(f\"Invalid params (b <= A): {invalid_params1.validate()}\")\n",
                    "\n",
                    "# Invalid parameters (negative A)\n",
                    "invalid_params2 = LBAParameters(A=-1, b=8, v=1, s=1, tau=0)\n",
                    "print(f\"Invalid params (negative A): {invalid_params2.validate()}\")\n",
                    "\n",
                    "# Invalid parameters (negative s)\n",
                    "invalid_params3 = LBAParameters(A=4, b=8, v=1, s=-1, tau=0)\n",
                    "print(f\"Invalid params (negative s): {invalid_params3.validate()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Different Parameter Configurations"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare different parameter configurations\n",
                    "configs = [\n",
                    "    LBAParameters(A=4, b=[6, 10, 20], v=1, s=1, tau=0),\n",
                    "    LBAParameters(A=4, b=[8, 12, 16], v=1, s=1, tau=0),\n",
                    "    LBAParameters(A=4, b=[6, 10, 20], v=[0.5, 1.0, 1.5], s=1, tau=0),\n",
                    "]\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
                    "\n",
                    "for i, config in enumerate(configs):\n",
                    "    data_config = model.generate_data(n_trials=300, parameters=config, n_acc=3, seed=42+i)\n",
                    "    \n",
                    "    # Plot mean RT by response\n",
                    "    mean_rt = data_config.groupby('response')['rt'].mean()\n",
                    "    axes[i].bar(mean_rt.index, mean_rt.values)\n",
                    "    axes[i].set_title(f'Config {i+1}: Mean RT by Response')\n",
                    "    axes[i].set_xlabel('Response')\n",
                    "    axes[i].set_ylabel('Mean RT')\n",
                    "    axes[i].set_ylim(0, max(mean_rt.values) * 1.1)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Print parameter details\n",
                    "for i, config in enumerate(configs):\n",
                    "    print(f\"\\nConfig {i+1}: {config}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Summary\n",
                    "\n",
                    "This notebook demonstrated:\n",
                    "\n",
                    "1. **Model Creation**: How to create an LBA model instance\n",
                    "2. **Parameter Definition**: Using named tuples for type-safe parameter handling\n",
                    "3. **Data Generation**: Simulating synthetic data from the model\n",
                    "4. **Visualization**: Plotting RT distributions and response proportions\n",
                    "5. **Model Fitting**: Framework for MCMC-based parameter estimation\n",
                    "6. **Parameter Validation**: Ensuring parameter constraints are met\n",
                    "7. **Configuration Comparison**: Testing different parameter settings\n",
                    "\n",
                    "The pyLBA package provides a clean, extensible framework for working with accumulator models in Python."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook_content


if __name__ == "__main__":
    # This would create a proper notebook file
    print("Demo notebook structure created")
    print("To create actual .ipynb file, use jupyter nbconvert or similar tools")
