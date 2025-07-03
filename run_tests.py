"""
Test runner for pyLBA package.
"""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run the test suite."""
    test_dir = Path(__file__).parent / "tests"
    
    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    print("Running pyLBA tests...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with return code {result.returncode}")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
