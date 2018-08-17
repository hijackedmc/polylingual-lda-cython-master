# coding=utf-8
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_type = Extension("polylda",
                 sources=["polylda.pyx",
                          "gamma.c"])
#
# ext_type = Extension("polylda_type",
#                      sources=["polylda.pyx",
#                               "gamma.c"])
# 如果这样会爆出错误： LINK : error LNK2001: 无法解析的外部符号 initpolylda_type

setup(name="polylda",
      cmdclass = {'build_ext': build_ext},
      ext_modules = cythonize([ext_type]))
# setup(ext_modules = cythonize([ext_type]))
