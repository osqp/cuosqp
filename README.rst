Python interface for CUDA implementation of OSQP
================================================


Python wrapper for CUDA implementation of `OSQP <https://osqp.org/>`__.

The OSQP (Operator Splitting Quadratic Program) solver is a numerical
optimization package for solving problems in the form

::

    minimize        0.5 x' P x + q' x

    subject to      l <= A x <= u

where ``x in R^n`` is the optimization variable. The objective function
is defined by a positive semidefinite matrix ``P in S^n_+`` and vector
``q in R^n``. The linear constraints are defined by matrix
``A in R^{m x n}`` and vectors ``l in R^m U {-inf}^m``,
``u in R^m U {+inf}^m``.


Installation
-----------
You need to install the following:

* `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_

* `CMake <https://cmake.org/>`_

* `GCC compiler <https://gcc.gnu.org/>`_ (Linux) or `Build Tools for Visual Studio 2017 <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017>`_ (Windows)

Make sure the environment variable ``CUDA_PATH`` is set to the CUDA Toolkit install directory.

Then run the following commands in your terminal:

::

  git clone --recurse-submodules https://github.com/oxfordcontrol/cuosqp
  cd cuosqp
  python setup.py install





Documentation
-------------

The interface is documented `here <https://osqp.org/docs/interfaces/python.html>`__.


Citing
------

If you use cuosqp for research, please cite our accompanying `paper <https://doi.org/10.1016/j.jpdc.2020.05.021>`__:

::

  @article{cuosqp,
    author  = {Schubiger, M. and Banjac, G. and Lygeros, J.},
    title   = {{GPU} acceleration of {ADMM} for large-scale quadratic programming},
    journal = {Journal of Parallel and Distributed Computing},
    year    = {2020},
    volume  = {144},
    pages   = {55--67},
    doi     = {10.1016/j.jpdc.2020.05.021},
  }

