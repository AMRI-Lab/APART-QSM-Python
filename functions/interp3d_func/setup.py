from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext

from distutils.extension import Extension


import numpy as np

setup(cmdclass = {'build_ext': build_ext},
      ext_modules=cythonize('interp3d.pyx'),
      include_dirs=[np.get_include()])

# run python setup.py build_ext --inplace