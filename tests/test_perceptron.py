"""Das duale Perceptron: Mini-Instanzen von Hand, schnelle gegen langsame Fassung, Eigenschaften (feste Zufalls-Seeds), Randfälle (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import ktp_kernels as KK
import ktp_perceptron as P
import ktp_scenario as S


def test_two_points_by_hand(two_points):
    """x = 0 (y = −1), x = 1 (y = +1), linear, K̃ = x·y + 1 = [[1, 1], [1, 2]]. Von Hand (Reihenfolge 0, 1): Epoche 1: Punkt 0 f = 0 → α₀ = 1, s = (−1, −1);
    Punkt 1: y s = −1 → α₁ = 1, s = (0, 1). Epoche 2: Punkt 0: y s = 0 → α₀ = 2, s = (−1, 0); Punkt 1: y s = 0 → α₁ = 2, s = (0, 2).
    Epoche 3: Punkt 0: y s = 0 → α₀ = 3, s = (−1, 1); Punkt 1 ok. Epoche 4 ohne Korrektur: α = (3, 2), 5 Korrekturen, 4 Epochen."""
    X, y = two_points
    Kt = KK.gram_tilde(("linear",), X)
    assert np.array_equal(Kt, [[1.0, 1.0], [1.0, 2.0]])
    r = P.run_dual(Kt, y, [0, 1], 100)
    assert r.converged and r.updates == 5 and r.epochs == 4 and list(r.alpha) == [3.0, 2.0]
    assert list(r.s) == [-1.0, 1.0]


def test_xor_polynomial_by_hand(xor):
    """K = 8I + 11ᵀ, K̃ = 8I + 2·11ᵀ; Reihenfolge a, b, c, d: Korrekturen bei a (f = 0), c (f = 2), d (f = 0) in Epoche 1 und b (f = −2) in Epoche 2."""
    Kt = KK.gram_tilde(("poly", 2), xor.X)
    assert np.array_equal(Kt, 8 * np.eye(4) + 2.0)
    log, r = P.trace_dual(Kt, xor.y, [0, 1, 2, 3], 50)
    fehler = [(s.epoch, "abcd"[s.point], s.f) for s in log if s.error]
    assert fehler == [(1, "a", 0.0), (1, "c", 2.0), (1, "d", 0.0), (2, "b", -2.0)]
    assert r.converged and r.updates == 4 and list(r.alpha) == [1, 1, 1, 1] and r.epochs == 3
    assert list(r.s * xor.y) == [8.0] * 4  # y_i f(x_i) = 8 an allen vier Punkten


def test_xor_linear_returns_to_w_zero_after_one_epoch(xor):
    """Lineares Perceptron auf XOR, Reihenfolge a, b, c, d: 4 Korrekturen, danach wieder w = 0 (Zyklus der Länge einer Epoche)."""
    Kt = KK.gram_tilde(("linear",), xor.X)
    r = P.run_dual(Kt, xor.y, [0, 1, 2, 3], 100, detect_cycle=True)
    assert r.stop == P.STOP_CYCLE and r.cycle_epochs == 1 and r.cycle_updates == 4
    assert list(r.s) == [0.0] * 4  # w = 0, also alle f(x_i) = 0


@pytest.mark.parametrize("seed", range(1, 9))
def test_fast_version_equals_the_point_by_point_trace(seed):
    inst = S.rings(30, seed)
    Kt = KK.gram_tilde(("gauss", 0.5), inst.X)
    order = S.order_of(30, seed)
    r = P.run_dual(Kt, inst.y, order, 1000)
    log, r2 = P.trace_dual(Kt, inst.y, order, 1000)
    assert (r.updates, r.epochs, r.stop) == (r2.updates, r2.epochs, r2.stop) and np.array_equal(r.alpha, r2.alpha)
    assert sum(1 for s in log if s.error) == r.updates


def test_first_error_unit():
    y = np.array([1.0, -1.0, 1.0])
    s = np.array([2.0, 0.0, -1.0])           # y s = 2, 0, −1
    idx = np.array([0, 1, 2])
    assert P.first_error_in_order(y, s, idx, 0) == 1      # der Fall 0 zählt als Fehler
    assert P.first_error_in_order(y, s, idx, 2) == 2
    assert P.first_error_in_order(np.ones(3), np.array([1.0, 1.0, 1.0]), idx, 0) == -1
    assert P.first_error_in_order(y, s, np.array([2, 0, 1]), 0) == 0                 # Reihenfolge zählt: Punkt 2 zuerst


def test_apply_update_keeps_s_equal_to_gram_times_alpha_y():
    inst = S.delivery_area(12, 4)
    Kt = KK.gram_tilde(("poly", 2), inst.X)
    alpha, s = np.zeros(12), np.zeros(12)
    for i in (3, 3, 7, 0, 11, 7):
        P.apply_update(alpha, s, Kt, inst.y, i)
    assert np.allclose(s, Kt @ (alpha * inst.y), rtol=1e-12) and alpha.sum() == 6 and alpha[3] == 2 and alpha[7] == 2


# ------------------------------------------------------------------ Eigenschaften
def _run(inst, spec, order=None, epochs=3000, y=None):
    y = inst.y if y is None else y
    return P.run_dual(KK.gram_tilde(spec, inst.X), y, order or S.order_of(inst.n, 1), epochs)


def test_flipping_all_labels_gives_the_same_alpha_and_negative_f():
    for seed in range(1, 11):
        inst = S.rings(40, seed)
        a = _run(inst, ("gauss", 0.5))
        b = _run(inst, ("gauss", 0.5), y=-inst.y)
        assert np.array_equal(a.alpha, b.alpha) and np.allclose(a.s, -b.s)


def test_renumbering_the_points_only_renumbers_alpha():
    inst = S.rings(30, 5)
    order = S.order_of(30, 5)
    perm = np.array(S.order_of(30, 99))               # neue Nummer k gehört zum alten Punkt perm[k]
    inv = np.argsort(perm)
    a = P.run_dual(KK.gram_tilde(("gauss", 0.5), inst.X), inst.y, order, 1000)
    b = P.run_dual(KK.gram_tilde(("gauss", 0.5), inst.X[perm]), inst.y[perm], [int(inv[i]) for i in order], 1000)
    assert a.updates == b.updates and np.array_equal(a.alpha[perm], b.alpha)


def test_rotation_of_the_input_does_not_change_polynomial_and_gauss_runs():
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for spec in (("poly", 2), ("gauss", 0.5)):
        for seed in range(1, 6):
            inst = S.rings(30, seed)
            a = P.run_dual(KK.gram_tilde(spec, inst.X), inst.y, S.order_of(30, seed), 1000)
            b = P.run_dual(KK.gram_tilde(spec, inst.X @ R.T), inst.y, S.order_of(30, seed), 1000)
            assert (a.updates, a.epochs) == (b.updates, b.epochs) and np.array_equal(a.alpha, b.alpha)


def test_scaling_the_gram_matrix_by_a_positive_constant_changes_no_decision():
    for seed in range(1, 6):
        inst = S.rings(30, seed)
        Kt = KK.gram_tilde(("gauss", 0.5), inst.X)
        a = P.run_dual(Kt, inst.y, S.order_of(30, seed), 1000)
        b = P.run_dual(4.0 * Kt, inst.y, S.order_of(30, seed), 1000)       # Faktor 4: exakt in Gleitkomma
        assert np.array_equal(a.alpha, b.alpha)


def test_learning_rate_has_no_effect_with_start_zero():
    """Start w = 0: eine Lernrate η skaliert w nur; die Entscheidungen y f ≤ 0 bleiben gleich (primale Regel im expliziten Merkmalsraum)."""
    inst = S.rings(30, 3)
    Phi = KK.with_bias_column(KK.poly_features(inst.X, 2))
    order = S.order_of(30, 3)
    w1, u1, e1, c1 = P.run_primal(Phi, inst.y, order, 1000)
    w2, u2, e2, c2 = P.run_primal(0.25 * Phi, inst.y, order, 1000)       # Merkmale mal 1/4 = Lernrate 1/4 in der Regel
    assert (u1, e1, c1) == (u2, e2, c2) and np.allclose(w1 * 0.25, w2)


def test_alpha_is_integral_and_sums_to_the_updates():
    for seed in range(1, 11):
        inst = S.delivery_area(40, seed)
        r = _run(inst, ("poly", 2), S.order_of(40, seed))
        assert np.all(r.alpha >= 0) and np.array_equal(r.alpha, np.round(r.alpha)) and r.alpha.sum() == r.updates
        assert 1 <= r.updates <= 40 * r.epochs


def test_same_seed_same_output_and_gram_is_symmetric():
    inst = S.moons(40, 5, 0.1)
    for spec in (("linear",), ("poly", 3), ("gauss", 0.5), ("sigmoid", 0.1, -1.0)):
        K = KK.kernel_matrix(spec, inst.X)
        assert np.allclose(K, K.T, rtol=1e-12, atol=1e-12)
    a, b = _run(inst, ("gauss", 0.5)), _run(inst, ("gauss", 0.5))
    assert np.array_equal(a.alpha, b.alpha) and np.allclose(np.diag(KK.kernel_matrix(("gauss", 0.5), inst.X)), 1.0)


def test_shuffling_each_epoch_is_a_different_run_but_reproducible():
    inst = S.rings(40, 2)
    Kt = KK.gram_tilde(("gauss", 0.5), inst.X)
    fest = P.run_dual(Kt, inst.y, S.order_of(40, 2), 1000)
    m1 = P.run_dual(Kt, inst.y, S.order_of(40, 2), 1000, shuffle_seed=5)
    m2 = P.run_dual(Kt, inst.y, S.order_of(40, 2), 1000, shuffle_seed=5)
    m3 = P.run_dual(Kt, inst.y, S.order_of(40, 2), 1000, shuffle_seed=6)
    assert np.array_equal(m1.alpha, m2.alpha) and m1.converged and m3.converged
    assert not (np.array_equal(m1.alpha, fest.alpha) and np.array_equal(m3.alpha, fest.alpha))   # auf diesem Zufalls-Seed weichen sie ab


# ------------------------------------------------------------------ Randfälle
def test_one_point_needs_exactly_one_update():
    Kt = np.array([[2.0]])
    r = P.run_dual(Kt, np.array([1.0]), [0], 10)
    assert r.converged and r.updates == 1 and r.epochs == 2 and r.alpha[0] == 1


def test_two_points_with_the_same_label():
    Kt = KK.gram_tilde(("linear",), np.array([[1.0, 0.0], [2.0, 0.0]]))
    r = P.run_dual(Kt, np.array([1.0, 1.0]), [0, 1], 10)
    assert r.converged and r.updates == 1       # der erste Punkt wird korrigiert (f = 0), danach sind beide richtig


def test_duplicate_point_with_other_label_never_converges_and_says_limit():
    X = np.array([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([1.0, -1.0, -1.0])
    r = P.run_dual(KK.gram_tilde(("gauss", 1.0), X), y, [0, 1, 2], 200)
    assert not r.converged and r.stop == P.STOP_EPOCHS and r.epochs == 200


def test_epoch_limit_one_and_update_limit():
    inst = S.rings(20, 1)
    Kt = KK.gram_tilde(("linear",), inst.X)
    r = P.run_dual(Kt, inst.y, S.order_of(20, 1), 1)
    assert r.epochs == 1 and r.stop in (P.STOP_EPOCHS, P.STOP_CONVERGED)
    r = P.run_dual(Kt, inst.y, S.order_of(20, 1), 1000, max_updates=7)
    assert r.stop == P.STOP_UPDATES and r.updates == 7


def test_origin_point_in_the_gauss_kernel_and_one_input_dimension():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    y = np.array([1.0, -1.0, -1.0])
    r = P.run_dual(KK.gram_tilde(("gauss", 0.5), X), y, [0, 1, 2], 100)
    assert r.converged
    X1 = np.array([[0.0], [1.0], [2.0], [3.0]])                       # p = 1
    r = P.run_dual(KK.gram_tilde(("poly", 2), X1), np.array([-1.0, -1.0, 1.0, 1.0]), [0, 1, 2, 3], 100)
    assert r.converged


def test_sigmoid_with_an_indefinite_gram_matrix_runs_without_any_message():
    inst = S.rings(100, 1)
    K = KK.kernel_matrix(("sigmoid", 1.0, 0.0), inst.X)
    assert KK.smallest_eigenvalue(K) < -1e-6                          # indefinit
    r = P.run_dual(K + 1.0, inst.y, S.order_of(100, 1), 200)
    assert r.stop in (P.STOP_CONVERGED, P.STOP_EPOCHS)                # kein Fehler, keine Warnung: nur die Meldung des Laufs


def test_decision_values_and_error_rate_count_zero_as_error():
    assert P.error_rate(np.array([0.0, 1.0, -1.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0])) == 0.5
    Kt = KK.gram_tilde(("linear",), np.array([[1.0, 0.0], [-1.0, 0.0]]))
    f = P.decision_values(Kt, np.array([1.0, 0.0]), np.array([1.0, -1.0]))
    assert list(f) == [2.0, 0.0]
    assert P.weight_norm2(Kt, np.array([1.0, 0.0]), np.array([1.0, -1.0])) == 2.0
