# Test cuosqp python module
import cuosqp as osqp
import numpy as np
from scipy import sparse
import scipy as sp
from numpy.random import Generator, PCG64

# Unit Test
import unittest
import numpy.testing as nptest


class unconstrained_tests(unittest.TestCase):

  def setUp(self):
    """
    Setup unconstrained quadratic problem
    """
    # Set random seed for reproducibility
    rg = Generator(PCG64(1))

    self.n = 30
    self.m = 0
    P = sparse.diags(rg.random(self.n)) + 0.2*sparse.eye(self.n)
    self.P = P.tocsc()
    self.q = rg.standard_normal(self.n)
    self.A = sparse.csc_matrix((self.m, self.n))
    self.l = np.array([])
    self.u = np.array([])
    self.opts = {'verbose': False,
                  'eps_abs': 1e-06,
                  'eps_rel': 1e-06,
                  'polish': False}
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

  def test_unconstrained_problem(self):
    # Solve problem
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(
        res.x, np.array([
             0.17221215, -1.84085666,  3.23111929,  0.32873825, -3.99110215,
            -1.0375029 , -0.64518994,  0.84374114,  2.19862467, -0.73591755,
            -0.11432888,  1.66275577,  1.28975978,  0.07288708,  1.87750662,
             0.15037534, -0.28584164, -0.05900426,  1.25488928, -1.28429794,
            -0.93771052, -0.66786523,  1.19416376, -0.61965718,  0.4316592 ,
            -0.9506598 ,  1.44596409, -1.91755938,  0.05563106,  1.06737479]),
        rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y, np.array([]))
    nptest.assert_allclose(res.info.obj_val, -17.69727194, rtol=1e-5, atol=1e-5)
