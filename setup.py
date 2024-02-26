import numpy as np
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension

extensions = cythonize(
    [
        Extension("sequence", ["fragmentomics_tools/sequence.pyx"]),
    ],
    compiler_directives={"language_level": "3"},
)

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
    include_dirs=[np.get_include()],
    ext_modules=extensions,
)
