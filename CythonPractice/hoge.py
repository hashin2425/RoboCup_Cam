from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include  # cimport numpy を使うため

ext = Extension("hoge", sources=["hoge.pyx"], include_dirs=['.', get_include()])
setup(name="hoge", ext_modules=cythonize([ext]))
