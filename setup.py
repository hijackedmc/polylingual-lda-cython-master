# coding=utf-8
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_type = Extension("polylda",
                 sources=["polylda.pyx",
                          "gamma.c"])

setup(name="polylda",
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize([ext_type]))
