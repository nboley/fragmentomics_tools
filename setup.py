from setuptools import setup

setup(
    name="fragmentomics_tools",
    version="1.0",
    packages=["fragmentomics_tools"],
    # scripts=["bin/build-fragments-h5"],
    install_requires=[
        "dacite",
        "pandas",
        "smart_open",
        "pyliftover",
        "matplotlib",
        "seaborn",
        "sparse",
    ],
)
