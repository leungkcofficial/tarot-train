#!/usr/bin/env python3
"""
Setup script for CKD Risk Prediction package
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if not line.startswith('#')]

# Read version from __init__.py
with open('__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"\'')
            break

setup(
    name="ckd_risk_prediction",
    version=version,
    description="Full-stack MLOps repository for AI-driven CKD risk prediction",
    author="CKD Risk Prediction Team",
    author_email="example@example.com",
    url="https://github.com/example/ckd-risk-prediction",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "ckd-ingest=run_pipeline:main",
        ],
    },
)