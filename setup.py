
from setuptools import setup, find_packages

setup(
    name="optimizers",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    author="Your Name",
    description="Meta-optimizer using PPO for neural network training",
)
