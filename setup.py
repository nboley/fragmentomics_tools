import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = cythonize(
    [
        Extension("sequence", ["fragmentomics_tools/sequence.pyx"]),
    ],
    compiler_directives={"language_level": "3"},
)

setup(
    include_dirs=[np.get_include()],
    ext_modules=extensions,
)
