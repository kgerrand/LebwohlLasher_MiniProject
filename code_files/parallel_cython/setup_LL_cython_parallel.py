from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(name="LebwohlLasher_cython_parallel",
      ext_modules=cythonize("LebwohlLasher_cython_parallel.pyx"),
      include_dirs=[numpy.get_include()])

LebwohlLasher_c = Extension("LebwohlLasher_cython_parallel", sources=["LebwohlLasher_cython_parallel"], extra_compile_args=['-O3'])