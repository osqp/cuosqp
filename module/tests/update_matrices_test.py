# Test cuosqp python module
import cuosqp as osqp
import numpy as np
import scipy as sp
from scipy import sparse
from numpy.random import Generator, PCG64

# Unit Test
import unittest
import numpy.testing as nptest


class update_matrices_tests(unittest.TestCase):

  def setUp(self):
    # Set random seed for reproducibility
    rg = Generator(PCG64(1))

    self.n = 5
    self.m = 8
    p = 0.7

    Pt = sparse.random(self.n, self.n, density=p, random_state=rg)
    Pt_new = Pt.copy()
    Pt_new.data += 0.1 * rg.standard_normal(Pt.nnz)

    self.P = sparse.triu(Pt.T.dot(Pt) + sparse.eye(self.n), format='csc')
    self.P_new = sparse.triu(Pt_new.T.dot(Pt_new) + sparse.eye(self.n), format='csc')
    self.q = rg.standard_normal(self.n)
    self.A = sparse.random(self.m, self.n, density=p, format='csc', random_state=rg)
    self.A_new = self.A.copy()
    self.A_new.data += rg.standard_normal(self.A_new.nnz)
    self.l = np.zeros(self.m)
    self.u = 30 + rg.standard_normal(self.m)
    self.opts = {'eps_abs': 1e-06,
                  'eps_rel': 1e-06,
                  'verbose': False}
    self.model = osqp.OSQP()
    self.model.setup(self.P, self.q, self.A, self.l, self.u, **self.opts)

  def test_solve(self):
    # Solve problem
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.00380355, -0.05288448, 0.81394426,
                                     -0.10583796, -0.05073545]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([0, 0., 0., 0., 0., -1.04731486, 0., 0.]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.50827192, rtol=1e-5, atol=1e-5)

  def test_update_P(self):
    # Update matrix P
    Px = self.P_new.data
    Px_idx = np.arange(self.P_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.04720881, -0.06776865, 0.86134936,
                                     -0.11074795, -0.09258846]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([0, 0., 0., 0., 0., -1.0260579, 0., 0.]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.55292652, rtol=1e-5, atol=1e-5)

  def test_update_P_allind(self):
    # Update matrix P
    Px = self.P_new.data
    self.model.update(Px=Px)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.04720881, -0.06776865, 0.86134936,
                                     -0.11074795, -0.09258846]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([0, 0., 0., 0., 0., -1.0260579, 0., 0.]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.55292652, rtol=1e-5, atol=1e-5)

  def test_update_A(self):
    # Update matrix A
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.06691278, 0.08556721, 0.07404456,
                                     -0.18316578, -0.25791392]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.48881119, 0., 0., -0.07316356, 0.,
                                     -0.29852164, 0., -1.14361885]),
                           rtol=1e-4, atol=1e-4)
    nptest.assert_allclose(res.info.obj_val, -0.10000495, rtol=1e-5, atol=1e-5)

  def test_update_A_allind(self):
    # Update matrix A
    Ax = self.A_new.data
    self.model.update(Ax=Ax)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.06691278, 0.08556721, 0.07404456,
                                     -0.18316578, -0.25791392]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.48881119, 0., 0., -0.07316356, 0.,
                                     -0.29852164, 0., -1.14361885]),
                           rtol=1e-4, atol=1e-4)
    nptest.assert_allclose(res.info.obj_val, -0.10000495, rtol=1e-5, atol=1e-5)

  def test_update_P_A_indP_indA(self):
    # Update matrices P and A
    Px = self.P_new.data
    Px_idx = np.arange(self.P_new.nnz)
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.07135987, 0.0912541, 0.07896563,
                                     -0.19533916, -0.27505513]),
                            rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.50970015, 0., 0., -0.07650362, 0.,
                                     -0.29287553, 0., -1.19697277]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.10665138, rtol=1e-5, atol=1e-5)

  def test_update_P_A_indP(self):
    # Update matrices P and A
    Px = self.P_new.data
    Px_idx = np.arange(self.P_new.nnz)
    Ax = self.A_new.data
    self.model.update(Px=Px, Px_idx=Px_idx, Ax=Ax)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.07135987, 0.0912541, 0.07896563,
                                     -0.19533916, -0.27505513]),
                            rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.50970015, 0., 0., -0.07650362, 0.,
                                     -0.29287553, 0., -1.19697277]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.10665138, rtol=1e-5, atol=1e-5)

  def test_update_P_A_indA(self):
    # Update matrices P and A
    Px = self.P_new.data
    Ax = self.A_new.data
    Ax_idx = np.arange(self.A_new.nnz)
    self.model.update(Px=Px, Ax=Ax, Ax_idx=Ax_idx)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.07135987, 0.0912541, 0.07896563,
                                     -0.19533916, -0.27505513]),
                            rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.50970015, 0., 0., -0.07650362, 0.,
                                     -0.29287553, 0., -1.19697277]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.10665138, rtol=1e-5, atol=1e-5)

  def test_update_P_A_allind(self):
    # Update matrices P and A
    Px = self.P_new.data
    Ax = self.A_new.data
    self.model.update(Px=Px, Ax=Ax)
    res = self.model.solve()

    # Assert close
    nptest.assert_allclose(res.x,
                           np.array([ 0.07135987, 0.0912541, 0.07896563,
                                     -0.19533916, -0.27505513]),
                            rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.y,
                           np.array([-0.50970015, 0., 0., -0.07650362, 0.,
                                     -0.29287553, 0., -1.19697277]),
                           rtol=1e-5, atol=1e-5)
    nptest.assert_allclose(res.info.obj_val, -0.10665138, rtol=1e-5, atol=1e-5)

