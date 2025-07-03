# Contributing to pyLBA

Thank you for your interest in contributing to pyLBA! We welcome contributions from the community.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/nuttidalab/pyLBA.git
cd pyLBA
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

We use `black` for code formatting:
```bash
black pyLBA/
```

And `flake8` for linting:
```bash
flake8 pyLBA/
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Or use the test runner:
```bash
python run_tests.py
```

## Adding New Models

To add a new accumulator model:

1. Create a new parameter class inheriting from `ModelParameters`
2. Create a new model class inheriting from `AccumulatorModel`
3. Implement the required abstract methods
4. Add tests for your model
5. Update the documentation

Example:
```python
class MyModelParameters(ModelParameters):
    param1: float
    param2: float

class MyModel(AccumulatorModel):
    def __init__(self):
        super().__init__("My Model")
    
    def pdf(self, rt, response, parameters):
        # Implement PDF calculation
        pass
    
    def generate_data(self, n_trials, parameters, **kwargs):
        # Implement data generation
        pass
    
    def get_parameter_class(self):
        return MyModelParameters
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features.

## Code of Conduct

Please be respectful and constructive in all interactions.
