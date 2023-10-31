from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

setup(name="LebwohlLasher_cython",
      ext_modules=cythonize("LebwohlLasher_cython.pyx"),
      include_dirs=[numpy.get_include()])

LebwohlLasher_c = Extension("LebwohlLasher_cython", sources=["LebwohlLasher_cython"], extra_compile_args=['-O3'])