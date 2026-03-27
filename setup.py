#!/usr/bin/env python3
"""
Setup script for Bloodroot Audio Backdoor Attack Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open(this_directory / "requirements.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Skip comments, empty lines, and installation instructions
        if line and not line.startswith("#") and not line.startswith("pip install"):
            requirements.append(line)

setup(
    name="bloodroot",
    version="1.0.0",
    author="Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, Jian-Jiun Ding",
    author_email="",
    description="Watermark-as-Trigger: A Novel Framework for Stealthy Audio Backdoor Attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Bloodroot-Audio-Backdoor",
    project_urls={
        "Paper": "https://arxiv.org/abs/2510.07909",
        "Bug Tracker": "https://github.com/yourusername/Bloodroot-Audio-Backdoor/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_dir={"": "WaterMark"},
    packages=find_packages(where="WaterMark"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "jupyter",
        ],
        "voice_conversion": [
            "parallel_wavegan",
            "paddlepaddle-gpu>=2.2.2",
            "paddlespeech==0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Add command-line tools here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
