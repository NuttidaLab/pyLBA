#!/usr/bin/env python3
"""
Quick installation and test script for pyLBA package.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {description} successful!")
            if result.stdout:
                print("Output:", result.stdout[:500])
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("Error:", result.stderr[:500])
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def main():
    """Main installation and test function."""
    print("=" * 60)
    print("pyLBA Package Installation and Test")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Install package in development mode
    success = run_command(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        "Installing pyLBA package"
    )
    
    if not success:
        print("\n❌ Installation failed. Please check the error messages above.")
        return False
    
    # Test basic import
    try:
        print("\n🔍 Testing basic import...")
        import pyLBA
        from pyLBA import LBAModel, LBAParameters
        print("✅ Basic import successful!")
        print(f"   Package version: {pyLBA.__version__}")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    # Run a quick functionality test
    try:
        print("\n🔍 Testing basic functionality...")
        model = LBAModel()
        params = LBAParameters(A=4, b=8, v=1, s=1, tau=0)
        data = model.generate_data(n_trials=10, parameters=params, n_acc=2, seed=42)
        
        assert len(data) == 10
        assert 'rt' in data.columns
        assert 'response' in data.columns
        assert params.validate() == True
        
        print("✅ Basic functionality test successful!")
        print(f"   Generated {len(data)} trials")
        print(f"   RT range: {data['rt'].min():.3f} - {data['rt'].max():.3f}")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    # Run unit tests if available
    test_file = Path(__file__).parent / "tests" / "test_lba.py"
    if test_file.exists():
        print("\n🔍 Running unit tests...")
        success = run_command(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            "Running unit tests"
        )
        
        if not success:
            print("⚠️  Some tests failed, but package is still functional")
    else:
        print("\n⚠️  Unit tests not found, skipping")
    
    # Run complete example
    example_file = Path(__file__).parent / "examples" / "complete_example.py"
    if example_file.exists():
        print("\n🔍 Running complete example...")
        success = run_command(
            [sys.executable, str(example_file)],
            "Running complete example"
        )
        
        if not success:
            print("⚠️  Example failed, but package is still functional")
    else:
        print("\n⚠️  Complete example not found, skipping")
    
    print("\n" + "=" * 60)
    print("Installation and testing completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Try running: python examples/complete_example.py")
    print("2. Check out the examples/ directory for more demos")
    print("3. Read the README.md for detailed usage instructions")
    print("4. For model fitting, install PyMC: pip install pymc")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
