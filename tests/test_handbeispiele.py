"""Die Handbeispiele der Planung als echte Tests (enge Toleranzen, exakte Brüche) und ihr Abgleich mit den Ergebnissen der App.

`handbeispiel_exakt.py` ist die unveränderte Kopie der 304 Prüfungen (Teil 1 Kern = Merkmalsraum, Teil 2 XOR linear nicht trennbar mit exaktem Zyklus, Teil 3 Kern-Perceptron
auf XOR mit allen 24 Reihenfolgen, Teil 3b zwei Maximalmargen, Teil 4 Gauß-Kern). Hier: jede Teilrechnung einmal, dann die App gegen dieselben Zahlen.
Langläufer: Teil 3 und 4 rechnen mit Brüchen und `SVC` (einige Sekunden, kein Orakel-Test im Sinn der 10-s-Regel)."""

import math

import numpy as np
import pytest

import handbeispiel_exakt as H
import ktp_kernels as KK
import ktp_margin as M
import ktp_perceptron as P
import ktp_scenario as S


def test_teil1_kern_ist_merkmalsraum_exakt_in_bruechen():
    H.teil1()


def test_teil2_xor_linear_nicht_trennbar_und_exakter_zyklus():
    zyk, nz = H.teil2()
    assert (zyk, nz) == (1, 1)         # Zyklus der Länge einer Epoche, ein Zustand am Epochenanfang (w = 0)


def test_teil3_kern_perceptron_auf_xor_und_alle_24_reihenfolgen():
    assert H.teil3() == (4, 4)


def test_teil3b_zwei_maximalmargen_exakt():
    H.teil3b()


def test_teil4_gauss_kern_auf_xor():
    g = H.teil4()
    assert {s: u for s, (u, _) in g.items()} == {0.5: 6, 1.0: 6, 2.0: 10, 4.0: 34}
    assert [round(b, 2) for _, b in g.values()] == [8.01, 10.70, 51.67, 579.42]


def test_alle_304_pruefungen_sind_gezaehlt():
    n0 = H.N[0]
    H.teil1()
    H.teil2()
    H.teil3()
    H.teil3b()
    H.teil4()
    assert H.N[0] - n0 == 304


# ------------------------------------------------------------------ die App gegen dieselben Zahlen
def test_app_two_points_equal_the_hand_values_of_teil3b(two_points):
    X, y = two_points
    K = KK.kernel_matrix(("linear",), X)
    qf, qt = M.max_margin(K, y, True), M.max_margin(K, y, False)
    assert qf.margin == pytest.approx(1.0, rel=1e-9) and list(np.round(qf.alpha, 9)) == [2.0, 2.0] and qf.b == pytest.approx(-1.0)
    assert qt.margin == pytest.approx(2.0 / math.sqrt(5.0), rel=1e-9) and list(np.round(qt.alpha, 9)) == [3.0, 2.0]


def test_app_xor_polynomial_equals_the_hand_values(xor):
    K = KK.kernel_matrix(("poly", 2), xor.X)
    q = M.max_margin(K, xor.y, False)
    assert q.margin == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-9) and np.allclose(q.alpha, 1.0 / 8.0) and q.w2 == pytest.approx(0.5, rel=1e-9)
    assert M.novikoff_bound(K + 1.0, q) == pytest.approx(5.0, rel=1e-9)
    assert M.max_margin(K, xor.y, True).margin == pytest.approx(2.0 * math.sqrt(2.0), rel=1e-9)     # Σαy = 0: beide Maxima gleich
    import itertools

    for perm in itertools.permutations(range(4)):
        r = P.run_dual(K + 1.0, xor.y, list(perm), 100)
        assert r.converged and r.updates == 4 and list(r.alpha) == [1, 1, 1, 1]
    # f(x) = 8 x₁x₂ an rationalen Punkten (Gleitkomma, Gram-Einträge ganzzahlig)
    pts = np.array([(a / 3.0, b / 5.0) for a in range(-6, 7, 2) for b in range(-10, 11, 4)])
    f = P.decision_values(KK.gram_tilde(("poly", 2), pts, xor.X), np.ones(4), xor.y)
    assert np.allclose(f, 8.0 * pts[:, 0] * pts[:, 1], rtol=1e-12, atol=1e-12)


def test_app_found_plane_is_the_maximal_one_on_xor(xor):
    K = KK.kernel_matrix(("poly", 2), xor.X)
    r = P.run_dual(K + 1.0, xor.y, [0, 1, 2, 3], 100)
    qf, qt = M.max_margin(K, xor.y, True), M.max_margin(K, xor.y, False)
    rf, rt = M.found_margin_ratio(K + 1.0, K, xor.y, r.alpha, qf, qt)
    assert rf == pytest.approx(1.0, abs=1e-9) and rt == pytest.approx(1.0, abs=1e-9)        # Verhältnis der Marge 1,000000000 (V10 Teil 2)


def test_app_two_point_perceptron_finds_the_maximal_plane_in_both_measures(two_points):
    X, y = two_points
    K = KK.kernel_matrix(("linear",), X)
    r = P.run_dual(K + 1.0, y, [0, 1], 100)
    assert list(r.alpha) == [3.0, 2.0]                     # dasselbe α wie die größte Trennung in (φ, 1)
    rf, rt = M.found_margin_ratio(K + 1.0, K, y, r.alpha, M.max_margin(K, y, True), M.max_margin(K, y, False))
    assert rf == pytest.approx(1.0, abs=1e-9) and rt == pytest.approx(1.0, abs=1e-9)


def test_margin_in_phi_one_is_never_larger_than_the_free_b_margin_in_the_ratio_sense():
    """Das Maximum mit freiem b enthält das Maximum in (φ, 1): Marge frei ≥ Marge in (φ, 1) (zwei Punkte: 1 gegen 2/√5)."""
    for seed in range(1, 6):
        inst = S.rings(40, seed)
        K = KK.kernel_matrix(("poly", 2), inst.X)
        assert M.max_margin(K, inst.y, True).margin >= M.max_margin(K, inst.y, False).margin * (1 - 1e-9)
