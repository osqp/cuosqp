# Test cuosqp python module
import cuosqp as osqp
import numpy as np
from scipy import sparse
from numpy.random import Generator, PCG64

# Unit Test
import unittest
import numpy.testing as nptest


class polish_tests(unittest.TestCase):

  def setUp(self):
    """
    Setup default options
    """
    self.opts = {'verbose': False,
                 'eps_abs': 1e-04,
                 'eps_rel': 1e-04,
                 'polish': True,
                 'polish_refine_iter': 8}

  def test_polish_simple(self):

    # Simple QP problem
    self.P = sparse.diags([11., 0.], format='csc')
    self.q = np.array([3, 4])
    self.A = sparse.csc_matrix([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
    self.u = np.array([0, 0, -15, 100, 80])
    self.l = -np.inf * np.ones(len(self.u))
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

    # Solve problem
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x, np.array([0., 5.]), rtol=1e-6, atol=1e-6)
    nptest.assert_allclose(res.y, np.array([1.66666667, 0., 1.33333333, 0., 0.]), rtol=1e-6, atol=1e-6)
    nptest.assert_allclose(res.info.obj_val, 20., rtol=1e-6, atol=1e-6)

  def test_polish_unconstrained(self):
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
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

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
        rtol=1e-6, atol=1e-6)
    nptest.assert_allclose(res.y, np.array([]))
    nptest.assert_allclose(res.info.obj_val, -17.69727194, rtol=1e-6, atol=1e-6)

  def test_polish_random(self):

    # Set random seed for reproducibility
    rg = Generator(PCG64(1))

    self.n = 30
    self.m = 50
    Pt = rg.standard_normal((self.n, self.n))
    self.P = sparse.triu(np.dot(Pt.T, Pt), format='csc')
    self.q = rg.standard_normal(self.n)
    self.A = sparse.csc_matrix(rg.standard_normal((self.m, self.n)))
    self.l = -3 + rg.standard_normal(self.m)
    self.u = 3 + rg.standard_normal(self.m)
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

    # Solve problem
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(
        res.x, np.array([
             0.14309944, -0.03539077,  0.27864189,  0.0693045 , -0.40741513,
             0.58500801, -0.05715695,  0.53470081,  0.15764935, -0.10198167,
             0.03584195, -0.28628935, -0.15170641,  0.10532207, -0.48210877,
             0.00868872,  0.48983164, -0.30742672,  0.54240528, -0.17622243,
            -0.38665758, -0.16340594, -0.24741171,  0.26922765,  0.53341687,
            -0.74634085, -1.28463569,  0.02608472, -0.23450606, -0.09142843]),
        rtol=1e-6, atol=1e-6)
    nptest.assert_allclose(
        res.y, np.array([
             0.,  0., 0.11863563,  0.,  0., 0.,  0.,  0.,  0., 0.23223504,
            -0.08489787,  0.,  0.,  0.,  0., 0.0274536 ,  0.,  0.,  0.,  0.,
             0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0., 0.,  0., -0.447373,
             0.,  0., 0.,  0.,  0., -0.12800017,  0., 0.,  0., -0.05106154,
             0.47314221,  0., -0.23398984,  0.,  0.,  0.,  0.]),
        rtol=2e-6, atol=2e-6)
    nptest.assert_allclose(res.info.obj_val, -5.680387544713935, rtol=1e-6, atol=1e-6)
