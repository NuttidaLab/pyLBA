"""
Setup configuration for pyLBA package.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyLBA",
    version="0.1.0",
    author="Rudramani Singha",
    author_email="rgs2151@columbia.com",
    description="A Python package for Linear Ballistic Accumulator and other accumulator models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nuttidalab/pyLBA",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "sphinx",
            "sphinx-rtd-theme",
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "nbsphinx",
            "jupyter",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="psychology, neuroscience, cognitive modeling, accumulator models, LBA, reaction time",
    project_urls={
        "Bug Reports": "https://github.com/nuttidalab/pyLBA/issues",
        "Source": "https://github.com/nuttidalab/pyLBA",
        "Documentation": "https://pylba.readthedocs.io/",
    },
)
