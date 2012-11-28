from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("kalmanagent", ["KalmanAgent.pyx"]), Extension("bzrc", ["bzrc.pyx"])]

setup(
  name = 'Kalman Filter App',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
