from setuptools import setup, find_packages

setup(
    name="baja_suspension",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",w
        "scipy", 
        "matplotlib",
        "PyYAML",
    ],
)
