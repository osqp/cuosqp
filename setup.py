import distutils.sysconfig as sysconfig
import os
import shutil as sh
import sys
from glob import glob
from platform import system
from shutil import copyfile, copy
from subprocess import call, check_output

import numpy
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Add parameters to cmake_args and define_macros
cmake_args = ["-DUNITTESTS=OFF"]
cmake_build_flags = []
define_macros = []
lib_subdir = []

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    cmake_args += ['-G', 'Visual Studio 15 2017']
    # Differentiate between 32-bit and 64-bit
    if sys.maxsize // 2 ** 32 > 0:
        cmake_args[-1] += ' Win64'
    cmake_build_flags += ['--config', 'Release']
    lib_name = 'osqp.lib'
    lib_subdir = ['Release']

else:  # Linux or Mac
    cmake_args += ['-G', 'Unix Makefiles']
    lib_name = 'libosqp.a'

# Pass Python option to CMake and Python interface compilation
cmake_args += ['-DPYTHON=ON']

# Pass CUDA options to CMake
cmake_args += ['-DCUDA_SUPPORT=ON', '-DDFLOAT=ON', '-DDLONG=OFF']

# Pass Python include directory
cmake_args += ['-DPYTHON_INCLUDE_DIRS=%s' % sysconfig.get_python_inc()]

# Pass Python to compiler launched from setup.py
define_macros += [('PYTHON', None)]


# Define osqp and qdldl directories
current_dir = os.getcwd()
osqp_dir = os.path.join('osqp_sources')
osqp_build_dir = os.path.join(osqp_dir, 'build')

# Interface files
include_dirs = [
    os.path.join(osqp_dir, 'include'),      # osqp.h
    os.path.join('extension', 'include'),   # auxiliary .h files
    numpy.get_include()]                    # numpy header files

sources_files = glob(os.path.join('extension', 'src', '*.c'))


# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# External libraries
libraries = ['cublas', 'cusparse', 'cudart', 'stdc++']
if system() == 'Linux':
    libraries += ['rt']
if system() == 'Windows' and sys.version_info[0] == 3:
    # They moved the stdio library to another place.
    # We need to include this to fix the dependency
    libraries += ['legacy_stdio_definitions']

# CUDA libraries
CUDA_PATH = os.environ['CUDA_PATH']
if system() == 'Windows':
    library_dirs = [os.path.join(CUDA_PATH, 'lib', 'x64')]
else:
    library_dirs = [os.path.join(CUDA_PATH, 'lib64')]

# Add OSQP compiled library
extra_objects = [os.path.join('extension', 'src', lib_name)]


class build_ext_osqp(build_ext):
    def build_extensions(self):
        # Compile OSQP using CMake

        # Create build directory
        if os.path.exists(osqp_build_dir):
            sh.rmtree(osqp_build_dir)
        os.makedirs(osqp_build_dir)
        os.chdir(osqp_build_dir)

        try:
            check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build OSQP")

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'osqpstatic'] +
             cmake_build_flags)

        # Change directory back to the python interface
        os.chdir(current_dir)

        # Copy static library to src folder
        lib_origin = [osqp_build_dir, 'out'] + lib_subdir + [lib_name]
        lib_origin = os.path.join(*lib_origin)
        copyfile(lib_origin, os.path.join('extension', 'src', lib_name))

        # Run extension
        build_ext.build_extensions(self)


_osqp = Extension('cuosqp._osqp',
                  define_macros=define_macros,
                  libraries=libraries,
                  library_dirs=library_dirs,
                  include_dirs=include_dirs,
                  extra_objects=extra_objects,
                  sources=sources_files,
                  extra_compile_args=compile_args)

packages = ['cuosqp',
            'cuosqp.tests']


# Read README.rst file
def readme():
    with open('README.rst') as f:
        return f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='cuosqp',
      version='0.1.0',
      author='Michel Schubiger, Goran Banjac, Bartolomeo Stellato',
      author_email='gbanjac@control.ee.ethz.ch',
      description='cuOSQP: CUDA Implementation of the OSQP Solver',
      long_description=readme(),
      package_dir={'cuosqp': 'module'},
      include_package_data=True,  # Include package data from MANIFEST.in
      setup_requires=["numpy >= 1.7"],
      install_requires=requirements,
      license='Apache 2.0',
      url="https://osqp.org/",
      cmdclass={'build_ext': build_ext_osqp},
      packages=packages,
      ext_modules=[_osqp])
