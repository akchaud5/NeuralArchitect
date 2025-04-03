from setuptools import setup, find_packages

setup(
    name="nn-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Comprehensive PyTorch-based neural networks framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/nn-project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
