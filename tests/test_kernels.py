"""Kerne: Werte von Hand, Aufwandszahlen (Zähleinheit Multiplikationen je Skalarprodukt), Gleichstand der dualen und der expliziten Auswertung."""

import math

import numpy as np
import pytest

import ktp_kernels as KK


def test_gram_entries_of_the_xor_polynomial_kernel_by_hand(xor):
    """(1,1)·(1,1) = 2 → 9; (1,1)·(1,−1) = 0 → 1; (1,1)·(−1,−1) = −2 → 1: K = 8I + 11ᵀ."""
    K = KK.kernel_matrix(("poly", 2), xor.X)
    assert K[0, 0] == 9 and K[0, 2] == 1 and K[0, 1] == 1 and np.array_equal(K, 8 * np.eye(4) + 1)
    assert np.array_equal(KK.gram_tilde(("poly", 2), xor.X), K + 1)


def test_gauss_kernel_by_hand_on_the_xor_points():
    """Abstand² 4 (a–c) und 8 (a–b): K = u mit u = exp(−2/σ²) bzw. u²; Diagonale 1."""
    xor_X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    K = KK.kernel_matrix(("gauss", 1.0), xor_X)
    u = math.exp(-2.0)
    assert K[0, 2] == pytest.approx(u, rel=1e-14) and K[0, 1] == pytest.approx(u * u, rel=1e-14) and np.allclose(np.diag(K), 1.0)
    assert KK.gamma_of(1.0) == 0.5 and KK.gamma_of(0.5) == 2.0


def test_linear_and_sigmoid_by_hand():
    X = np.array([[1.0, 2.0], [3.0, -1.0]])
    assert KK.kernel_matrix(("linear",), X).tolist() == [[5.0, 1.0], [1.0, 10.0]]
    assert KK.kernel_matrix(("sigmoid", 0.1, -1.0), X)[0, 0] == pytest.approx(math.tanh(0.5 - 1.0), rel=1e-14)
    with pytest.raises(ValueError):
        KK.kernel_matrix(("unbekannt",), X)


def test_feature_dimension_and_multiplication_counts_of_the_plan():
    assert [KK.poly_dimension(p, 2) for p in (2,)] == [6] and KK.poly_dimension(10, 3) == 286
    assert KK.poly_dimension(100, 3) == 176851 and KK.poly_dimension(1000, 3) == 167668501
    assert [KK.kernel_multiplications(p, 3) for p in (2, 10, 100, 1000)] == [4, 12, 102, 1002]
    assert KK.kernel_multiplications(2, 2) == 3


def test_break_even_points_of_the_plan():
    """p = 10, d = 3: 23 verschiedene Punkte kosten 23·12 = 276 < 286, 24 kosten 288 (nicht mehr billiger); mit Bau von φ(x) (2·286 = 572): 47 (564) gegen 48 (576)."""
    assert KK.break_even_points(10, 3) == 23 and 23 * 12 < 286 <= 24 * 12
    assert KK.break_even_points(10, 3, with_feature_build=True) == 47 and 47 * 12 < 572 < 48 * 12
    assert KK.break_even_points(2, 3) == 2            # 2·4 = 8 < 10, 3·4 = 12 ≥ 10: in 2D ist der Kern schon ab 3 Punkten teurer
    assert KK.break_even_points(2, 2) == 1


def test_poly_features_in_two_dimensions_of_degree_two_have_the_hand_weights():
    """Gewichte der Monome (1, x₁, x₂, x₁², x₂², x₁x₂) im Skalarprodukt: (c², 2c, 2c, 1, 1, 2); Summe über Einträge ist (x·y + c)² exakt in Brüchen."""
    from fractions import Fraction as F

    pts = [(F(1, 2), F(3)), (F(-2, 3), F(5, 7)), (F(0), F(0)), (F(7), F(-3, 4))]
    for c in (1, 2):
        w = (c * c, 2 * c, 2 * c, 1, 1, 2)
        for x in pts:
            for y in pts:
                mx = (1, x[0], x[1], x[0] ** 2, x[1] ** 2, x[0] * x[1])
                my = (1, y[0], y[1], y[0] ** 2, y[1] ** 2, y[0] * y[1])
                assert sum(wi * a * b for wi, a, b in zip(w, mx, my)) == (x[0] * y[0] + x[1] * y[1] + c) ** 2
    X = np.array([[1.0, 2.0], [0.5, -1.0]])
    Phi = KK.poly_features(X, 2, c=2.0)
    assert np.allclose(Phi @ Phi.T, (X @ X.T + 2.0) ** 2, rtol=1e-12)


def test_poly_features_refuse_a_huge_space_and_bias_column():
    with pytest.raises(ValueError):
        KK.poly_features(np.zeros((2, 100)), 3)
    assert KK.with_bias_column(np.zeros((3, 2))).shape == (3, 3) and np.all(KK.with_bias_column(np.zeros((3, 2)))[:, -1] == 1)


def test_kernel_names_use_the_german_decimal_comma():
    assert KK.kernel_name(("gauss", 0.25)) == "Gauß σ = 0,25" and KK.kernel_name(("poly", 2)) == "Polynom 2" and KK.kernel_name(("linear",)) == "Linear"
    assert KK.kernel_name(("sigmoid", 0.1, -1.0)) == "Sigmoid a = 0,1, c = −1" and KK.fmt_num(8.0) == "8"


def test_feature_space_flags_and_rotation_invariance():
    assert KK.has_feature_space(("gauss", 1.0)) and not KK.has_feature_space(("sigmoid", 1.0, 0.0))
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 2))
    th = 1.1
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    for spec in (("linear",), ("poly", 3), ("gauss", 0.7)):
        assert np.allclose(KK.kernel_matrix(spec, X), KK.kernel_matrix(spec, X @ R.T), rtol=1e-11, atol=1e-12)
