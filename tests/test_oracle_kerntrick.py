"""Orakel-Tests (ORACLE-PLAYBOOK §2): jede Kernroutine gegen einen ANDEREN Rechenweg; scikit-learn und scipy nur hier als Fremdreferenz.

(a) duales Perceptron (Kerntrick) = primales im expliziten Merkmalsraum (Tensorpotenz, anderer Code als `poly_features`) = `sklearn.linear_model.Perceptron`;
(b) Polynomkern = Skalarprodukt im Merkmalsraum; (c) größte Trennung gegen Aufzählung aller Stützmengen, `SVC` (freies b) und nichtnegative kleinste Quadrate ((φ, 1));
(d) Gauß-Kern auf XOR gegen die geschlossene Form 1 − exp(−2/σ²); (e) kleinster Eigenwert der Gram-Matrix; (f) Literaturwert Cortes/Vapnik 1995 Gl. 37 (Stufe W)."""

import itertools
import math

import numpy as np
import pytest

import ktp_kernels as KK
import ktp_margin as M
import ktp_perceptron as P
import ktp_scenario as S


def tensor_features(X, d):
    """(x, 1)^{⊗ d}: ⟨φ(x), φ(y)⟩ = ((x, 1)·(y, 1))^d = (x·y + 1)^d (Multinomialsatz ohne Gewichte, dafür redundante Koordinaten)."""
    Xt = np.hstack([np.asarray(X, float), np.ones((len(X), 1))])
    out = Xt
    for _ in range(d - 1):
        out = np.einsum("ni,nj->nij", out, Xt).reshape(len(Xt), -1)
    return out


# ------------------------------------------------------------------ (a) Kerntrick
@pytest.mark.parametrize("make,d", [(lambda s: S.rings(60, s), 2), (lambda s: S.rings(60, s), 3), (lambda s: S.delivery_area(40, s), 2), (lambda s: S.delivery_area(40, s), 3)])
def test_dual_equals_primal_in_the_explicit_feature_space(make, d):
    for seed in range(1, 11):
        inst = make(seed)
        order = S.order_of(inst.n, seed)
        Kt = KK.gram_tilde(("poly", d), inst.X)
        rd = P.run_dual(Kt, inst.y, order, 3000)
        Phi = np.hstack([tensor_features(inst.X, d), np.ones((inst.n, 1))])
        wp, up, ep, cp = P.run_primal(Phi, inst.y, order, 3000)
        assert (rd.updates, rd.epochs, rd.converged) == (up, ep, cp)
        wd = Phi.T @ (rd.alpha * inst.y)
        assert np.linalg.norm(wd - wp) <= 1e-9 * np.linalg.norm(wp)


def test_primal_equals_sklearn_perceptron():
    sk = pytest.importorskip("sklearn.linear_model")
    for d in (2, 3):
        for seed in range(1, 11):
            inst = S.rings(60, seed)
            order = S.order_of(inst.n, seed)
            Phi = np.hstack([KK.poly_features(inst.X, d), np.ones((inst.n, 1))])
            Kt = KK.gram_tilde(("poly", d), inst.X)
            rd = P.run_dual(Kt, inst.y, order, 3000)
            clf = sk.Perceptron(penalty=None, alpha=0.0, fit_intercept=False, max_iter=rd.epochs, tol=None, shuffle=False, eta0=1.0)
            clf.fit(Phi[order], inst.y[order])
            wd = Phi.T @ (rd.alpha * inst.y)
            assert np.linalg.norm(clf.coef_[0] - wd) <= 1e-9 * np.linalg.norm(wd)


# ------------------------------------------------------------------ (b) Kern = Merkmalsraum
@pytest.mark.parametrize("p,d", [(2, 2), (2, 3), (3, 2), (5, 3), (10, 3)])
def test_polynomial_kernel_equals_the_scalar_product_in_the_feature_space(p, d):
    rng = np.random.default_rng(p * 10 + d)
    X = rng.normal(size=(30, p))
    K = KK.kernel_matrix(("poly", d), X)
    for Phi in (tensor_features(X, d), KK.poly_features(X, d)):
        assert np.max(np.abs(Phi @ Phi.T - K) / np.maximum(np.abs(K), 1.0)) < 1e-12
    assert KK.poly_features(X, d).shape[1] == math.comb(p + d, d)


# ------------------------------------------------------------------ (c) größte Trennung
def _enumerate_margin(K, y, free_b):
    """Aufzählung aller Stützmengen: KKT-System je Menge lösen, zulässige nehmen (α > 0, y f ≥ 1); eindeutig für positiv definite K."""
    n = len(y)
    best = None
    for r in range(1, n + 1):
        for S_ in itertools.combinations(range(n), r):
            S_ = list(S_)
            Q = (y[:, None] * y[None, :]) * K
            if free_b:
                A = np.zeros((r + 1, r + 1))
                A[:r, :r] = Q[np.ix_(S_, S_)]
                A[:r, r] = y[S_]
                A[r, :r] = y[S_]
                rhs = np.r_[np.ones(r), 0.0]
            else:
                A, rhs = Q[np.ix_(S_, S_)], np.ones(r)
            try:
                sol = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            a = sol[:r]
            b = sol[r] if free_b else 0.0
            if np.any(a <= 1e-12):
                continue
            alpha = np.zeros(n)
            alpha[S_] = a
            if np.all(Q @ alpha + y * b >= 1 - 1e-9):
                w2 = float(alpha @ Q @ alpha)
                if best is None or abs(w2 - best) < 1e-9:
                    best = w2
    return best


@pytest.mark.parametrize("free_b", [True, False])
def test_max_margin_equals_enumeration_of_all_support_sets(free_b):
    checked = 0
    for seed in range(1, 41):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 7))
        X = rng.normal(size=(n, 2))
        y = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        K = KK.kernel_matrix(("gauss", 1.0), X)
        ref = _enumerate_margin(K if free_b else K + 1.0, y, free_b)   # in (φ, 1) rechnet die Aufzählung mit K̃ = K + 1
        q = M.max_margin(K, y, free_b)
        assert q.certified and ref is not None
        assert q.w2 == pytest.approx(ref, rel=1e-6)
        checked += 1
    assert checked == 40


def test_max_margin_free_b_equals_svc():
    sk = pytest.importorskip("sklearn.svm")
    for make in (lambda s: S.rings(40, s), lambda s: S.moons(40, s, 0.1)):
        for seed in range(1, 6):
            inst = make(seed)
            K = KK.kernel_matrix(("gauss", 1.0), inst.X)
            q = M.max_margin(K, inst.y, True)
            m = sk.SVC(kernel="precomputed", C=1e5, tol=1e-10, max_iter=3_000_000).fit(K, inst.y)
            v = np.zeros(inst.n)
            v[m.support_] = m.dual_coef_[0]
            assert q.certified and q.w2 == pytest.approx(float(v @ K @ v), rel=1e-5)
            assert q.b == pytest.approx(float(m.intercept_[0]), abs=1e-3)


def test_max_margin_in_phi_one_equals_nonnegative_least_squares():
    opt = pytest.importorskip("scipy.optimize")
    for seed in range(1, 6):
        inst = S.rings(30, seed)
        Kt = KK.gram_tilde(("gauss", 1.0), inst.X)
        Q = (inst.y[:, None] * inst.y[None, :]) * Kt
        L = np.linalg.cholesky(Q + 1e-13 * np.eye(inst.n))
        a, _ = opt.nnls(L.T, np.linalg.solve(L, np.ones(inst.n)))
        q = M.max_margin(Kt - 1.0, inst.y, False)
        assert q.certified and q.w2 == pytest.approx(float(a @ Q @ a), rel=1e-5)


def test_max_margin_polynomial_has_a_certificate_although_the_feature_space_is_finite():
    for seed in range(1, 6):
        inst = S.rings(60, seed)
        for free in (True, False):
            q = M.max_margin(KK.kernel_matrix(("poly", 2), inst.X), inst.y, free)
            assert q.certified and q.min_constraint >= 1 - 1e-4 and q.gap <= 1e-6 * max(1.0, q.w2)


def test_non_separable_data_get_no_certificate():
    xor = S.xor_instance()
    K = KK.kernel_matrix(("linear",), xor.X)
    assert not M.max_margin(K, xor.y, False).certified and not M.max_margin(K, xor.y, True).certified


# ------------------------------------------------------------------ (d) Gauß auf XOR
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 4.0])
def test_gauss_margin_on_xor_equals_the_closed_form_svc_and_the_eigenvalue(sigma):
    xor = S.xor_instance()
    u = math.exp(-2.0 / sigma ** 2)
    K = KK.kernel_matrix(("gauss", sigma), xor.X)
    q = M.max_margin(K, xor.y, True)
    assert q.certified and q.margin == pytest.approx(1.0 - u, rel=1e-9)
    assert np.allclose(q.alpha, 1.0 / (1.0 - u) ** 2, rtol=1e-8) and q.w2 == pytest.approx(4.0 / (1.0 - u) ** 2, rel=1e-9)
    y = xor.y
    assert float(y @ K @ y / (y @ y)) == pytest.approx((1.0 - u) ** 2, rel=1e-12)      # Rayleigh-Quotient von y
    sk = pytest.importorskip("sklearn.svm")
    m = sk.SVC(kernel="precomputed", C=1e6, tol=1e-10).fit(K, y)
    v = np.zeros(4)
    v[m.support_] = m.dual_coef_[0]
    assert 2.0 / math.sqrt(float(v @ K @ v)) == pytest.approx(1.0 - u, abs=1e-6)


def test_gauss_kernel_perceptron_corrections_on_xor_in_the_order_a_b_c_d():
    """Handbeispiel Teil 4: 6 / 6 / 10 / 34 Korrekturen für σ = 0,5 / 1 / 2 / 4; Schranke R̃²‖w*‖² = 8,01 / 10,70 / 51,67 / 579,42."""
    xor = S.xor_instance()
    for sigma, ups, bound in ((0.5, 6, 8.01), (1.0, 6, 10.70), (2.0, 10, 51.67), (4.0, 34, 579.42)):
        K = KK.kernel_matrix(("gauss", sigma), xor.X)
        r = P.run_dual(K + 1.0, xor.y, [0, 1, 2, 3], 1000)
        q = M.max_margin(K, xor.y, False)
        assert r.converged and r.updates == ups
        assert M.novikoff_bound(K + 1.0, q) == pytest.approx(bound, abs=0.005) and r.updates <= M.novikoff_bound(K + 1.0, q)


# ------------------------------------------------------------------ (e) Eigenwert
def test_smallest_eigenvalue_is_numpys_and_the_gauss_gram_matrix_is_positive_definite():
    for seed in range(1, 21):
        inst = S.rings(40, seed)
        K = KK.kernel_matrix(("gauss", 0.5), inst.X)
        ev = np.linalg.eigvalsh(K)
        assert KK.smallest_eigenvalue(K) == ev[0] and ev[0] > -40 * np.finfo(float).eps * ev[-1]
        assert ev[0] > 0                                                     # verschiedene Punkte: positiv definit
        Kp = KK.kernel_matrix(("poly", 2), inst.X)
        assert KK.smallest_eigenvalue(Kp) > -40 * np.finfo(float).eps * np.linalg.eigvalsh(Kp)[-1]   # Skalarprodukt: positiv semidefinit


def test_a_doubled_point_makes_the_gauss_gram_matrix_singular_and_labels_untrennbar():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    K = KK.kernel_matrix(("gauss", 1.0), X)
    assert abs(KK.smallest_eigenvalue(K)) < 1e-14
    r = P.run_dual(K + 1.0, np.array([1.0, 1.0, -1.0]), [0, 1, 2], 100)
    assert not r.converged


# ------------------------------------------------------------------ (f) Literatur
def test_cortes_vapnik_1995_eq_37_degree_two_feature_map():
    """Quelle (Stufe W): Polynomkern (u·v + 1)^d, Gl. 37; Grad 2 in 2D mit dem Merkmal (1, √2 u₁, √2 u₂, u₁², u₂², √2 u₁u₂)."""
    rng = np.random.default_rng(3)
    U, V = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))

    def phi(u):
        return np.stack([np.ones(len(u)), math.sqrt(2) * u[:, 0], math.sqrt(2) * u[:, 1], u[:, 0] ** 2, u[:, 1] ** 2, math.sqrt(2) * u[:, 0] * u[:, 1]], axis=1)

    assert np.allclose(phi(U) @ phi(V).T, KK.kernel_matrix(("poly", 2), U, V), rtol=1e-12)
