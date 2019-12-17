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


Documentation
-------------

The interface is documented `here <https://osqp.org/docs/interfaces/python.html>`__.


Citing
------

If you use cuosqp for research, please cite our accompanying `paper <https://arxiv.org/pdf/1912.04263.pdf>`__:

::

    @article{cuosqp,
    author  = {Schubiger, M. and Banjac, G. and Lygeros, J.},
    title   = {{GPU} acceleration of {ADMM} for large-scale quadratic programming},
    journal = {arXiv:1912.04263},
    year    = {2019},
    }

