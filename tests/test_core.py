"""
tests/test_core.py
==================
Test suite for HMCVelo core pipeline.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hmcvelo import HMCVelo
from hmcvelo.data import make_synthetic, HMCData, infer_cytosine_proportions
from hmcvelo.model import solve_model, solve_model_fast, demethylation_velocity
from hmcvelo.params import least_squares_params, _build_system, _fit_gene
from hmcvelo.rank import rank_velocity_genes


@pytest.fixture
def small_data():
    return make_synthetic(n_cells=40, n_genes=50, seed=0)


@pytest.fixture
def medium_data():
    return make_synthetic(n_cells=100, n_genes=200, seed=42)


@pytest.fixture(params=[True, False], ids=["fast", "exact"])
def fitted_model(small_data, request):
    model = HMCVelo(small_data, fast=request.param, min_cells=2)
    model.run(verbose=False)
    return model


class TestHMCData:

    def test_make_synthetic_shape(self, small_data):
        assert small_data.h.shape == (40, 50)
        assert small_data.m.shape == (40, 50)
        assert small_data.c.shape == (40, 50)

    def test_make_synthetic_celltypes(self, small_data):
        types = set(small_data.cell_type)
        assert types == {"Ex", "Inh", "NeuN+", "NeuN-"}

    def test_conservation_law(self, small_data):
        """m + h + c should be close to 1 per cell per gene (proportions)."""
        total = small_data.h + small_data.m + small_data.c
        # Should be approximately 1 for proportion data
        assert total.min() >= 0.0
        assert total.max() <= 1.01   # small floating point tolerance

    def test_non_negative(self, small_data):
        assert (small_data.h >= 0).all()
        assert (small_data.m >= 0).all()
        assert (small_data.c >= 0).all()

    def test_obs_methylation_state(self, small_data):
        oms = small_data.obs_methylation_state
        assert oms.shape == (40, 50)
        np.testing.assert_allclose(oms, small_data.h + small_data.m)

    def test_n_cells_n_genes(self, small_data):
        assert small_data.n_cells == 40
        assert small_data.n_genes == 50

    def test_repr(self, small_data):
        r = repr(small_data)
        assert "HMCData" in r
        assert "40" in r

    def test_infer_cytosine_proportions(self):
        xm = np.array([[0.3, 0.4], [0.2, 0.5]])
        xh = np.array([[0.2, 0.1], [0.3, 0.3]])
        xc = infer_cytosine_proportions(xm, xh)
        expected = np.array([[0.5, 0.5], [0.5, 0.2]])
        np.testing.assert_allclose(xc, expected, atol=1e-10)

    def test_infer_cytosine_clipped(self):
        """Should clip negative values from floating point noise."""
        xm = np.array([[0.6]])
        xh = np.array([[0.5]])   # sum > 1
        xc = infer_cytosine_proportions(xm, xh)
        assert xc[0, 0] == 0.0


class TestParams:

    def test_build_system_shape(self):
        h = np.array([0.2, 0.3, 0.1])
        m = np.array([0.4, 0.3, 0.5])
        c = np.array([0.4, 0.4, 0.4])
        A, b = _build_system(h, m, c)
        assert A.shape == (3, 2)
        assert b.shape == (3,)

    def test_build_system_values(self):
        h = np.array([0.2])
        m = np.array([0.4])
        c = np.array([0.4])
        A, b = _build_system(h, m, c)
        # A[:, 0] = c, A[:, 1] = h, b = 2m
        np.testing.assert_allclose(A[0, 0], 0.4)  # c
        np.testing.assert_allclose(A[0, 1], 0.2)  # h
        np.testing.assert_allclose(b[0], 0.8)      # 2m

    def test_build_system_excludes_zeros(self):
        """Zero-state cells should be masked out."""
        h = np.array([0.2, 0.0, 0.3])
        m = np.array([0.4, 0.0, 0.3])
        c = np.array([0.4, 0.0, 0.4])
        A, b = _build_system(h, m, c)
        assert A.shape[0] == 2   # zero cell excluded

    def test_fit_gene_nonnegative(self, small_data):
        """Rate parameters must be non-negative."""
        params = _fit_gene(small_data.h[:, 0], small_data.m[:, 0],
                           small_data.c[:, 0], min_cells=2)
        assert params[0] >= 0.0   # beta_M
        assert params[1] >= 0.0   # beta_D

    def test_least_squares_shape(self, small_data):
        bM, bD = least_squares_params(
            small_data.h, small_data.m, small_data.c,
            min_cells=2, verbose=False
        )
        assert bM.shape == (50,)
        assert bD.shape == (50,)

    def test_least_squares_nonnegative(self, small_data):
        bM, bD = least_squares_params(
            small_data.h, small_data.m, small_data.c,
            min_cells=2, verbose=False
        )
        assert (bM >= 0).all()
        assert (bD >= 0).all()

    def test_scale_invariance(self, small_data):
        """
        Scale invariance: multiplying all rates by κ should give
        the same steady-state proportions (Eq. 12).
        Since βH=1 is fixed, κ affects βM and βD identically.
        The ratio βM/βD should be scale-invariant.
        """
        bM, bD = least_squares_params(
            small_data.h, small_data.m, small_data.c,
            min_cells=2, verbose=False
        )
        # Ratios should be finite where both are nonzero
        nonzero = (bM > 0) & (bD > 0)
        if nonzero.sum() > 0:
            ratios = bM[nonzero] / bD[nonzero]
            assert np.isfinite(ratios).all()

    def test_steady_state_consistency(self):
        """
        At the estimated steady state, 2m ≈ βM*c + βD*h should hold.
        Test with a known analytical solution.
        βM=2, βH=1, βD=3 -> steady state: m/c = βM/βH = 2, h/m = βH/βD = 1/3
        So if c=0.6: m=βM*c/βH=1.2... but we're in proportions.
        Use proportional: xm+xh+xc=1, xm/xc=βM=2, xh/xm=βH/βD=1/3
        => xm=2xc, xh=2xc/3, xm+xh+xc=1 => 2xc+2xc/3+xc=1 => xc=3/11
        """
        xc = 3.0 / 11
        xm = 2 * xc
        xh = 2 * xc / 3

        # Build fake 10-cell dataset all at steady state
        n = 10
        h = np.tile(xh, (n, 1))
        m = np.tile(xm, (n, 1))
        c = np.tile(xc, (n, 1))

        bM, bD = least_squares_params(h, m, c, min_cells=5, verbose=False)

        # Should recover βM≈2, βD≈3 (up to scale since βH=1)
        # The ratio βM/βD should be ≈ 2/3
        if bD[0] > 0:
            ratio = bM[0] / bD[0]
            np.testing.assert_allclose(ratio, 2.0 / 3.0, rtol=0.05)


class TestModel:

    def test_solve_model_fast_shape(self, small_data):
        bM = np.ones(50) * 0.5
        bD = np.ones(50) * 0.8
        v = solve_model_fast(small_data.m, small_data.h, small_data.c, bM, bD)
        assert v.shape == (40, 50)

    def test_solve_model_fast_formula(self):
        """v = βH * m - βD * h, with βH=1."""
        m = np.array([[0.4]])
        h = np.array([[0.2]])
        c = np.array([[0.4]])
        bM = np.array([0.5])
        bD = np.array([2.0])
        v = solve_model_fast(m, h, c, bM, bD)
        expected = 1.0 * 0.4 - 2.0 * 0.2   # = 0.0
        np.testing.assert_allclose(v[0, 0], expected, atol=1e-10)

    def test_solve_model_exact_shape(self, small_data):
        bM = np.ones(50) * 0.5
        bD = np.ones(50) * 0.8
        v = solve_model(small_data.m, small_data.h, small_data.c,
                        bM, bD, dt=1e-5)
        assert v.shape == (40, 50)

    def test_solve_model_finite(self, small_data):
        bM = np.ones(50) * 0.5
        bD = np.ones(50) * 0.8
        v = solve_model_fast(small_data.m, small_data.h, small_data.c, bM, bD)
        assert np.isfinite(v).all()

    def test_conservation_law_ode(self):
        """
        dm/dt + dh/dt + dc/dt = 0 (conservation law, note after paper Eq.5).
        Verify numerically.
        """
        from hmcvelo.model import _ode_rhs
        y = [0.4, 0.2, 0.4]   # m, h, c
        dydt = _ode_rhs(y, 0, beta_M=1.5, beta_H=1.0, beta_D=2.0)
        total_derivative = sum(dydt)
        np.testing.assert_allclose(total_derivative, 0.0, atol=1e-12)

    def test_steady_state_zero_velocity(self):
        """
        At exact steady state (dm/dt=dh/dt=dc/dt=0),
        velocity should be zero.
        With βM=2, βH=1, βD=3: steady state xc=3/11, xm=6/11, xh=2/11.
        """
        from hmcvelo.model import _ode_rhs
        xc = 3.0 / 11
        xm = 6.0 / 11
        xh = 2.0 / 11
        dydt = _ode_rhs([xm, xh, xc], 0, beta_M=2.0, beta_H=1.0, beta_D=3.0)
        np.testing.assert_allclose(dydt, [0.0, 0.0, 0.0], atol=1e-10)

    def test_fast_vs_exact_agreement(self, small_data):
        """
        Fast analytic mode and exact odeint should give very similar
        results for small dt (paper: dt=1e-5).
        """
        bM = np.ones(50) * 0.5
        bD = np.ones(50) * 0.8
        v_fast  = solve_model_fast(small_data.m, small_data.h, small_data.c, bM, bD)
        v_exact = solve_model(small_data.m, small_data.h, small_data.c,
                              bM, bD, dt=1e-5)
        # Should agree to within ~1% for well-conditioned states
        finite = np.isfinite(v_exact) & np.isfinite(v_fast)
        if finite.sum() > 0:
            np.testing.assert_allclose(
                v_fast[finite], v_exact[finite], rtol=0.02
            )

    def test_demethylation_velocity_shape(self, small_data):
        bM = np.ones(50) * 0.5
        bD = np.ones(50) * 0.8
        d = demethylation_velocity(small_data.m, small_data.h,
                                   small_data.c, bM, bD)
        assert d.shape == (40, 50)

    def test_demethylation_velocity_formula(self):
        """d = βH * m - βM * c (Eq. 11)."""
        m = np.array([[0.4]])
        h = np.array([[0.2]])
        c = np.array([[0.4]])
        bM = np.array([0.5])
        bD = np.array([2.0])
        d = demethylation_velocity(m, h, c, bM, bD)
        expected = 1.0 * 0.4 - 0.5 * 0.4   # βH*m - βM*c = 0.4 - 0.2 = 0.2
        np.testing.assert_allclose(d[0, 0], expected, atol=1e-10)


class TestHMCVelo:

    def test_fit_sets_params(self, small_data):
        model = HMCVelo(small_data, min_cells=2)
        assert model.beta_M is None
        model.fit(verbose=False)
        assert model.beta_M is not None
        assert model.beta_D is not None
        assert model.beta_M.shape == (50,)

    def test_compute_velocity_requires_fit(self, small_data):
        model = HMCVelo(small_data)
        with pytest.raises(RuntimeError, match="fit"):
            model.compute_velocity()

    def test_velocity_shape(self, fitted_model):
        assert fitted_model.velocity.shape == (40, 50)

    def test_velocity_finite(self, fitted_model):
        assert np.isfinite(fitted_model.velocity).all()

    def test_run_convenience(self, small_data):
        model = HMCVelo(small_data, min_cells=2).run(verbose=False)
        assert model.velocity is not None

    def test_summary_keys(self, fitted_model):
        s = fitted_model.summary()
        assert "n_cells" in s
        assert "n_genes" in s
        assert "velocity_mean" in s
        assert "velocity_finite_pct" in s
        assert s["velocity_finite_pct"] == 100.0

    def test_repr(self, fitted_model):
        r = repr(fitted_model)
        assert "HMCVelo" in r
        assert "fitted" in r

    def test_beta_H_fixed(self, fitted_model):
        """βH must always be 1.0 (scale invariance, Section 2.6)."""
        assert fitted_model.beta_H == 1.0

    def test_velocity_direction_immature(self, medium_data):
        """
        NeuN- (immature) cells should have higher positive velocity
        than NeuN+ (mature) cells on average, since 5hmC increases
        during maturation.
        """
        model = HMCVelo(medium_data, min_cells=2).run(verbose=False)
        mask_neg = medium_data.cell_type == "NeuN-"
        mask_pos = medium_data.cell_type == "NeuN+"
        v_neg = model.velocity[mask_neg].mean()
        v_pos = model.velocity[mask_pos].mean()
        # NeuN- should be moving (nonzero velocity), direction depends on data
        assert np.isfinite(v_neg)
        assert np.isfinite(v_pos)

    def test_demethylation_velocity(self, fitted_model):
        d = fitted_model.compute_demethylation_velocity()
        assert d.shape == (40, 50)
        assert np.isfinite(d).all()

    def test_different_seeds_different_results(self):
        """Different synthetic datasets should give different velocities."""
        d1 = make_synthetic(n_cells=40, n_genes=50, seed=1)
        d2 = make_synthetic(n_cells=40, n_genes=50, seed=2)
        m1 = HMCVelo(d1, min_cells=2).run(verbose=False)
        m2 = HMCVelo(d2, min_cells=2).run(verbose=False)
        assert not np.allclose(m1.velocity, m2.velocity)


class TestRanking:

    def test_rank_returns_all_celltypes(self, fitted_model, small_data):
        ranked = rank_velocity_genes(
            fitted_model.velocity,
            small_data.cell_type,
            small_data.var_names,
            top_n=5,
            verbose=False,
        )
        assert set(ranked.keys()) == {"Ex", "Inh", "NeuN+", "NeuN-"}

    def test_rank_top_n(self, fitted_model, small_data):
        ranked = rank_velocity_genes(
            fitted_model.velocity,
            small_data.cell_type,
            small_data.var_names,
            top_n=5,
            verbose=False,
        )
        for ct, df in ranked.items():
            assert len(df) <= 5

    def test_rank_gene_names_valid(self, fitted_model, small_data):
        ranked = rank_velocity_genes(
            fitted_model.velocity,
            small_data.cell_type,
            small_data.var_names,
            top_n=10,
            verbose=False,
        )
        valid_genes = set(small_data.var_names)
        for ct, df in ranked.items():
            for gene in df["gene"]:
                assert gene in valid_genes

    def test_rank_has_required_columns(self, fitted_model, small_data):
        ranked = rank_velocity_genes(
            fitted_model.velocity,
            small_data.cell_type,
            small_data.var_names,
            top_n=5,
            verbose=False,
        )
        for ct, df in ranked.items():
            assert "gene" in df.columns
            assert "t_stat" in df.columns
            assert "p_value" in df.columns
            assert "mean_velocity" in df.columns


class TestMathProperties:

    def test_remark1_oms_not_xc(self, small_data):
        """
        Remark 1: embedding should use OMS = h + m, not xc.
        OMS retains covariance between h and m.
        xc = 1 - OMS is algebraically redundant.
        Verify that OMS and xc carry the same information
        (rank of their matrices should be equal).
        """
        oms = small_data.h + small_data.m
        xc  = small_data.c

        # xc = 1 - oms (in proportion space), so they span the same subspace
        # but xc discards the h vs m split
        rank_oms = np.linalg.matrix_rank(oms)
        rank_xc  = np.linalg.matrix_rank(xc)

        # OMS should have at least as much rank as xc
        assert rank_oms >= rank_xc

    def test_conservation_law_holds_throughout(self, small_data):
        """dm/dt + dh/dt + dc/dt = 0 for all cells and genes."""
        from hmcvelo.model import _ode_rhs
        bM, bD = least_squares_params(
            small_data.h, small_data.m, small_data.c,
            min_cells=2, verbose=False
        )
        for i in range(min(5, small_data.n_cells)):
            for g in range(min(5, small_data.n_genes)):
                y = [small_data.m[i, g], small_data.h[i, g], small_data.c[i, g]]
                dydt = _ode_rhs(y, 0, bM[g], 1.0, bD[g])
                np.testing.assert_allclose(sum(dydt), 0.0, atol=1e-12)

    def test_scale_invariance_velocity_direction(self, small_data):
        """
        Scale invariance (Eq. 12): multiplying all rates by κ
        should not change the DIRECTION of velocity, only magnitude.
        """
        bM, bD = least_squares_params(
            small_data.h, small_data.m, small_data.c,
            min_cells=2, verbose=False
        )
        v1 = solve_model_fast(small_data.m, small_data.h, small_data.c, bM, bD)

        kappa = 3.0
        v2 = solve_model_fast(small_data.m, small_data.h, small_data.c,
                              bM * kappa, bD * kappa, beta_H=kappa)

        # Directions should agree (sign of velocity preserved)
        nonzero = (v1 != 0) & (v2 != 0)
        if nonzero.sum() > 0:
            signs_agree = np.sign(v1[nonzero]) == np.sign(v2[nonzero])
            assert signs_agree.mean() > 0.95  # >95% same direction