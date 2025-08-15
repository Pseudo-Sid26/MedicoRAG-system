#!/usr/bin/env python3
"""
Setup script for MedicoRAG System
Optimized for FAISS and Streamlit Cloud deployment
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="medico-rag-system",
    version="2.0.0",
    author="Siddhesh Chavan",
    author_email="siddheshchavan@example.com",
    description="AI-powered medical literature RAG system using FAISS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pseudo-Sid26/MedicoRAG-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "medico-rag=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Pseudo-Sid26/MedicoRAG-system/issues",
        "Source": "https://github.com/Pseudo-Sid26/MedicoRAG-system",
        "Documentation": "https://github.com/Pseudo-Sid26/MedicoRAG-system/blob/main/README.md",
    },
)
