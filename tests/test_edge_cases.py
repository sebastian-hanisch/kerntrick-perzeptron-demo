"""Randfälle der Analysen und des QP (DEMO-PLAYBOOK §12): n = 1 und n = 2, einklassige Daten, Grenzen der Regler, nicht trennbare Daten, jede Auswahl der Seitenleiste.

Die Randfälle des Perceptrons selbst (n = 1, doppelter Punkt, Epochengrenze 1, Sigmoid) stehen in `test_perceptron.py`."""

import itertools

import numpy as np
import pytest

import ktp_constants as C
import ktp_evaluation as ev
import ktp_kernels as KK
import ktp_margin as M
import ktp_perceptron as P


def test_qp_with_exactly_two_points_and_with_one_class_only():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    y = np.array([-1.0, 1.0])
    q = M.max_margin(KK.kernel_matrix(("linear",), X), y, True)
    assert q.certified and q.margin == pytest.approx(1.0) and q.n_support == 2
    one_class = M.max_margin(KK.kernel_matrix(("linear",), X), np.array([1.0, 1.0]), True)
    assert not one_class.certified and one_class.iterations == 0           # ohne beide Klassen gibt es keine Trennung mit freiem b


def test_qp_in_phi_one_with_one_point_is_the_scaled_inverse_norm():
    K = np.array([[3.0]])                                                  # K̃ = 4, y = +1: α = 1/4, ‖w‖² = 1/4, Marge 4
    q = M.max_margin(K, np.array([1.0]), False)
    assert q.certified and q.w2 == pytest.approx(0.25) and q.margin == pytest.approx(4.0)


def test_novikoff_bound_is_never_exceeded_on_small_separable_instances():
    import ktp_scenario as S

    for seed in range(1, 21):
        inst = S.rings(20, seed)
        for spec in (("poly", 2), ("gauss", 0.5), ("gauss", 1.0)):
            K = KK.kernel_matrix(spec, inst.X)
            r = P.run_dual(K + 1.0, inst.y, S.order_of(20, seed), 3000)
            q = M.max_margin(K, inst.y, False)
            assert r.converged and q.certified and r.updates <= M.novikoff_bound(K + 1.0, q) + 1e-9


def test_every_xor_kernel_and_order_gives_a_verdict_and_linear_never_converges():
    for kern in C.XOR_KERNELS:
        for order in ("a, b, c, d", "d, c, b, a", "b, d, a, c"):
            a = ev.analyse_xor(kern, order)
            kind, text = ev.verdict_xor(a)
            assert text and kind in ("success", "warning")
            if kern == "Linear":
                assert a.run.stop == P.STOP_CYCLE and a.all_orders_converged == 0 and kind == "warning"
            else:
                assert a.run.converged and a.all_orders_converged == 24 and a.qp is not None and a.run.updates <= a.bound + 1e-9


def test_xor_formula_is_8_x1_x2_for_the_polynomial_kernel_in_every_order():
    for order in ev.C.XOR_ORDERS:
        assert ev.analyse_xor("Polynom 2", order).formula == "8 x₁x₂"


@pytest.mark.parametrize("n", [C.N_MIN, C.N_MAX])
@pytest.mark.parametrize("daten", C.EX3_DATEN)
def test_dataset_analysis_at_the_slider_ends_for_every_dataset(n, daten):
    a = ev.analyse_dataset(daten, "Gauß σ = 0,5", n, False, 3)
    assert a.inst.n == n and a.run.converged and a.grid[2].shape == (60, 60)
    if n > C.QP_LIVE_MAX_N:
        assert a.bound is None and "nur bis n = 100" in a.qp_note
    else:
        assert a.bound is not None and a.run.updates <= a.bound + 1e-9


def test_dataset_analysis_with_shuffling_and_linear_limit():
    a = ev.analyse_dataset(C.DATEN_RINGE, "Linear", 20, False, 2)
    assert not a.run.converged and a.bound is None and "nicht konvergiert" in a.qp_note and 0 < a.train_error <= 0.5
    b = ev.analyse_dataset(C.DATEN_RINGE, "Gauß σ = 1", 40, True, 2)
    assert b.run.converged


def test_bandwidth_analysis_at_both_ends_of_the_grid():
    lo, hi = ev.analyse_bandwidth(C.SIGMAS[0], 3), ev.analyse_bandwidth(C.SIGMAS[-1], 3)
    assert lo.run.converged and hi.run.converged
    assert lo.offdiag_mean < 1e-2 and hi.offdiag_mean > 0.9                 # Einheitsmatrix gegen Einsmatrix
    assert lo.gram.shape == (100, 100) and np.allclose(np.diag(lo.gram), 1.0)


@pytest.mark.parametrize("etiketten", C.EX5_ETIKETTEN)
@pytest.mark.parametrize("daten", C.EX5_DATEN)
def test_trap_analysis_for_every_label_and_data_choice(etiketten, daten):
    a = ev.analyse_trap(daten, etiketten, "Gauß σ = 0,1", 3000, 5)
    assert a.n == (100 if daten == C.EX5_DATEN[0] else 200)
    assert 0.0 <= a.test_error <= 1.0 and 0.0 <= a.train_error <= 1.0
    if etiketten == "wahr":
        assert a.run is a.run_true and a.train_error == 0.0 and a.run.converged


def test_trap_with_a_tiny_epoch_limit_says_limit_not_untrennbar():
    a = ev.analyse_trap(C.EX5_DATEN[0], "zufällig", "Gauß σ = 0,25", 100, 5)
    kind, text = ev.verdict_trap(a)
    assert not a.run.converged and a.run.epochs == 100 and "Grenze erreicht" in text and "nicht „nicht trennbar“" in text


def test_trap_worst_case_at_the_slider_maximum_still_returns_a_result():
    a = ev.analyse_trap(C.EX5_DATEN[1], "zufällig", "Gauß σ = 0,5", C.EPOCH_CHOICES[-1], 5)      # Liefergebiet n = 200, σ = 0,5, 50 000 Epochen
    assert a.run.stop in (P.STOP_CONVERGED, P.STOP_EPOCHS, P.STOP_UPDATES) and a.run.updates <= C.MAX_UPDATES


def test_cost_analysis_for_every_degree_and_the_ends_of_the_p_slider():
    for d, p in itertools.product(C.DEGREES, (C.P_MIN, C.P_MAX)):
        a = ev.analyse_cost(d, p, 3)
        assert a.same_run and a.rel_w < 1e-9 and any(row[0] == p for row in a.rows)
        assert ev.verdict_cost(a)[0] == "info"


def test_break_even_never_negative_and_monotone_in_p():
    for d in C.DEGREES:
        assert all(KK.break_even_points(p, d) >= 0 for p in range(1, 60))
        assert KK.break_even_points(1000, d) > KK.break_even_points(10, d) > KK.break_even_points(2, d) - 1


def test_frozen_sweep_has_the_sections_and_both_sets():
    sw = ev.load_sweep()
    assert set(sw) == {"meta", "ringe", "bandbreite", "liefergebiet", "monde", "zufall", "d2", "marge", "sigmoid", "kerntrick"}
    assert list(sw["bandbreite"]) == [KK.fmt_num(s) for s in C.SIGMAS]
    for row in sw["bandbreite"].values():
        assert set(row) == {"A", "B"} and row["A"]["runs"] == row["B"]["runs"] == 30
