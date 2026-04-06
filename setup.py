from setuptools import setup, find_packages


def _parse_requirements(path: str) -> list[str]:
    """Read requirements.txt, skipping comment lines and blank lines."""
    requirements = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


setup(
    name="financial-genomics",
    version="1.0.0",
    description=(
        "Applying genomic sequence analysis to financial time-series: "
        "ACGT alphabet discretization, k-mer grammar extraction, and "
        "LSTM-based regime prediction with walk-forward backtesting."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Financial Genomics Research",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=_parse_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial-genomics=pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
