# Test cuosqp python module
import cuosqp as osqp
import numpy as np
from scipy import sparse
import scipy as sp
from numpy.random import Generator, PCG64

# Unit Test
import unittest
import numpy.testing as nptest


class feasibility_tests(unittest.TestCase):

  def setUp(self):
    """
    Setup equality constrained feasibility problem

        min     0
        st      A x = l = u
    """
    # Set random seed for reproducibility
    rg = Generator(PCG64(1))

    self.n = 30
    self.m = 30
    self.P = sparse.csc_matrix((self.n, self.n))
    self.q = np.zeros(self.n)
    self.A = sparse.random(self.m, self.n, density=1.0, format='csc', random_state=rg)
    self.u = rg.random(self.m)
    self.l = self.u
    self.opts = {'verbose': False,
                  'eps_abs': 5e-05,
                  'eps_rel': 5e-05,
                  'scaling': True,
                  'adaptive_rho': 0,
                  'rho': 10,
                  'max_iter': 5000,
                  'polish': False}
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

  def test_feasibility_problem(self):
    # Solve problem
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(
        res.x,
        np.array([-0.0160104 ,  0.13893361, -0.26093395, -0.46924047, -0.15730985,
                  -0.41690876,  0.30332078,  0.8674208 , -0.20840655,  0.87361543,
                  -0.03207495,  0.0227269 ,  0.02933772, -0.1449326 , -0.54664477,
                   0.19578402,  0.90044367,  0.55150767, -0.57337961, -0.62474418,
                   0.47859095, -0.4826634 , -1.02627427, -0.14334523,  0.16996476,
                   0.24067098,  0.08854844,  0.69244021,  0.51045395, -0.05598347]),
        rtol=1e-3, atol=1e-3)
    nptest.assert_allclose(res.y, np.zeros(self.m), rtol=1e-3, atol=1e-3)
    nptest.assert_allclose(res.info.obj_val, 0., rtol=1e-3, atol=1e-3)
