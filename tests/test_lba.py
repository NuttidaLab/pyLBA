"""
Test suite for pyLBA package.
"""

import pytest
import numpy as np
import pandas as pd
from pyLBA import LBAModel, LBAParameters


class TestLBAParameters:
    """Test LBAParameters class."""
    
    def test_creation(self):
        """Test parameter creation."""
        params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)
        assert params.A == 4
        assert params.b == 8
        assert params.v == 1
        assert params.s == 1
        assert params.tau == 0
    
    def test_validation_valid(self):
        """Test validation with valid parameters."""
        params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)
        assert params.validate() is True
    
    def test_validation_invalid_b_less_than_A(self):
        """Test validation fails when b <= A."""
        params = LBAParameters(A=4, b=3, v=1, s=1, tau=0)
        assert params.validate() is False
    
    def test_validation_invalid_negative_A(self):
        """Test validation fails with negative A."""
        params = LBAParameters(A=-1, b=8, v=1, s=1, tau=0)
        assert params.validate() is False
    
    def test_validation_invalid_negative_s(self):
        """Test validation fails with negative s."""
        params = LBAParameters(A=4, b=8, v=1, s=-1, tau=0)
        assert params.validate() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)
        param_dict = params.to_dict()
        expected = {'A': 4, 'b': 8, 'v': 1, 's': 1, 'tau': 0}
        assert param_dict == expected
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        param_dict = {'A': 4, 'b': 8, 'v': 1, 's': 1, 'tau': 0}
        params = LBAParameters.from_dict(param_dict)
        assert params.A == 4
        assert params.b == 8


class TestLBAModel:
    """Test LBAModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = LBAModel()
        self.params = LBAParameters(A=4, b=[6, 10, 20], v=1, s=1, tau=0)
    
    def test_model_creation(self):
        """Test model creation."""
        assert self.model.name == "Linear Ballistic Accumulator"
        assert not self.model.fitted
    
    def test_get_parameter_class(self):
        """Test getting parameter class."""
        param_class = self.model.get_parameter_class()
        assert param_class == LBAParameters
    
    def test_generate_data(self):
        """Test data generation."""
        data = self.model.generate_data(
            n_trials=100, 
            parameters=self.params, 
            n_acc=3, 
            seed=42
        )
        
        assert isinstance(data, pd.DataFrame)
        assert 'rt' in data.columns
        assert 'response' in data.columns
        assert len(data) == 100
        assert data['rt'].min() > 0
        assert data['response'].min() >= 0
        assert data['response'].max() <= 2
    
    def test_generate_data_reproducible(self):
        """Test that data generation is reproducible with same seed."""
        data1 = self.model.generate_data(
            n_trials=50, 
            parameters=self.params, 
            n_acc=3, 
            seed=123
        )
        
        data2 = self.model.generate_data(
            n_trials=50, 
            parameters=self.params, 
            n_acc=3, 
            seed=123
        )
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_generate_data_scalar_parameters(self):
        """Test data generation with scalar parameters."""
        params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)
        data = self.model.generate_data(
            n_trials=50, 
            parameters=params, 
            n_acc=2, 
            seed=42
        )
        
        assert len(data) == 50
        assert data['response'].max() <= 1
    
    def test_pdf_computation(self):
        """Test PDF computation."""
        # Generate some data
        data = self.model.generate_data(
            n_trials=20, 
            parameters=self.params, 
            n_acc=3, 
            seed=42
        )
        
        rt = data['rt'].to_numpy()
        response = data['response'].to_numpy()
        
        # Compute PDF
        pdf = self.model.pdf(rt, response, self.params, n_acc=3)
        
        # Check that we get a tensor
        assert hasattr(pdf, 'eval')
        
        # Evaluate and check properties
        pdf_vals = pdf.eval()
        assert len(pdf_vals) == len(rt)
        assert np.all(pdf_vals > 0)  # All positive probabilities
    
    def test_default_priors(self):
        """Test default priors."""
        priors = self.model.get_default_priors(n_acc=3)
        
        expected_keys = ['A', 'b', 'v', 's', 'tau']
        assert all(key in priors for key in expected_keys)


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        from pyLBA.utils import validate_data
        
        data = pd.DataFrame({
            'rt': [0.5, 0.7, 0.4],
            'response': [0, 1, 0]
        })
        
        # Should not raise any exception
        validate_data(data)
    
    def test_invalid_data_missing_columns(self):
        """Test validation fails with missing columns."""
        from pyLBA.utils import validate_data
        
        data = pd.DataFrame({
            'rt': [0.5, 0.7, 0.4]
        })
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            validate_data(data)
    
    def test_invalid_data_negative_rt(self):
        """Test validation fails with negative RT."""
        from pyLBA.utils import validate_data
        
        data = pd.DataFrame({
            'rt': [0.5, -0.1, 0.4],
            'response': [0, 1, 0]
        })
        
        with pytest.raises(ValueError, match="All reaction times must be positive"):
            validate_data(data)
    
    def test_invalid_data_negative_response(self):
        """Test validation fails with negative response."""
        from pyLBA.utils import validate_data
        
        data = pd.DataFrame({
            'rt': [0.5, 0.7, 0.4],
            'response': [0, -1, 0]
        })
        
        with pytest.raises(ValueError, match="All responses must be non-negative"):
            validate_data(data)


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete workflow: generate -> fit -> predict."""
        model = LBAModel()
        params = LBAParameters(A=4, b=[6, 10], v=1, s=1, tau=0)
        
        # Generate data
        data = model.generate_data(n_trials=100, parameters=params, n_acc=2, seed=42)
        
        # This would test fitting if we had a fast fitting method
        # For now, just test that the model can be created and data generated
        assert len(data) == 100
        assert data['response'].max() <= 1
        
        # Test that model remembers it hasn't been fitted
        assert not model.fitted
        
        # Test predict fails before fitting
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(n_trials=10)


if __name__ == "__main__":
    pytest.main([__file__])
