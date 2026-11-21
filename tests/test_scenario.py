"""Erzeuger: Eigenschaften der Regel, eingefrorene Werte je Erzeuger und Zufalls-Seed (SplitMix64, Bänder wegen `math.cos`/`math.sin` der C-Bibliothek), Labelsoperationen."""

import math

import numpy as np
import pytest

import ktp_constants as C
import ktp_scenario as S


def test_xor_instance_is_the_four_points_of_the_plan(xor):
    assert xor.X.tolist() == [[1, 1], [-1, -1], [1, -1], [-1, 1]] and xor.y.tolist() == [1, 1, -1, -1] and S.XOR_NAMES == "abcd"


def test_rings_have_the_radii_the_gap_and_balanced_classes():
    for seed in range(1, 21):
        inst = S.rings(100, seed)
        r = np.hypot(inst.X[:, 0], inst.X[:, 1])
        assert int((inst.y > 0).sum()) == 50
        assert np.all(np.abs(r[inst.y < 0] - 1.0) <= 0.1 + 1e-12) and np.all(np.abs(r[inst.y > 0] - 2.0) <= 0.1 + 1e-12)
        assert r[inst.y > 0].min() - r[inst.y < 0].max() >= 0.8 - 1e-12              # Lücke 0,8 im Radius


def test_delivery_area_rule_and_half_area():
    inst = S.delivery_area(4000, 3)
    assert np.all(np.abs(inst.X) <= 1.0)
    assert np.array_equal(inst.y > 0, np.hypot(inst.X[:, 0], inst.X[:, 1]) <= C.DELIVERY_RADIUS)
    assert math.pi * C.DELIVERY_RADIUS ** 2 == pytest.approx(2.0, rel=1e-12)           # Kreisfläche = halbe Fläche von [−1, 1]² (4)
    assert (inst.y > 0).mean() == pytest.approx(0.5, abs=4.5 * math.sqrt(0.25 / 4000))     # Anteil +1 mit 4,5 Standardfehlern


def test_moons_are_two_arcs_with_bounded_noise():
    clean = S.moons(200, 4, 0.0)
    up, low = clean.X[clean.y < 0], clean.X[clean.y > 0]
    assert np.allclose(np.hypot(up[:, 0], up[:, 1]), 1.0) and np.all(up[:, 1] >= -1e-12)
    assert np.allclose(np.hypot(low[:, 0] - 1.0, low[:, 1] - 0.5), 1.0) and np.all(low[:, 1] <= 0.5 + 1e-12)
    noisy = S.moons(200, 4, 0.1)
    assert np.max(np.abs(noisy.X - clean.X)) <= 0.1 + 1e-12 and np.max(np.abs(noisy.X - clean.X)) > 0.05


def test_flip_labels_flips_exactly_k_and_keeps_the_true_class():
    base = S.rings(100, 3)
    flipped = S.flip_labels(base, 0.1, 3)
    assert int((flipped.y != base.y).sum()) == 10 and np.array_equal(flipped.y_true, base.y) and np.array_equal(flipped.X, base.X)
    assert int((S.flip_labels(base, 0.0, 3).y != base.y).sum()) == 0


def test_random_labels_keep_the_counts_and_the_points():
    base = S.rings(100, 3)
    rnd = S.random_labels(base, 3)
    assert int((rnd.y > 0).sum()) == 50 and np.array_equal(rnd.X, base.X) and np.array_equal(rnd.y_true, base.y)
    assert int((rnd.y != base.y).sum()) > 20                      # stark vertauscht (erwartet 50)


def test_order_is_a_permutation_fixed_per_seed_and_differs_between_seeds():
    o = S.order_of(50, 7)
    assert sorted(o) == list(range(50)) and o == S.order_of(50, 7) and o != S.order_of(50, 8)


def test_test_points_are_fresh_points_of_the_same_generator():
    inst = S.rings(100, 3)
    test = S.fresh_test_points(C.DATEN_RINGE, 200, 3)
    assert test.n == 200 and not np.array_equal(test.X[:100], inst.X)
    r = np.hypot(test.X[:, 0], test.X[:, 1])
    assert np.all(np.abs(r[test.y_true > 0] - 2.0) <= 0.1 + 1e-12)


# eingefrorene Werte (Zufalls-Seed 3, n = 20): erste und letzte Koordinaten und Summe, relativ 1e-9 (Kosinus und Sinus kommen aus der C-Bibliothek)
FROZEN = {
    "rings": (S.rings(20, 3), [0.786819452494, 0.680174429303], [-1.424982922767, 1.365962752836], -9.338685344, 0),
    "lief": (S.delivery_area(20, 3), [-0.773099315886, 0.400587027186], [-0.243269699213, -0.260621660471], 1.373523434, 4),
    "moons0": (S.moons(20, 3, 0.0), [0.937153793265, 0.348916562762], [1.505058213147, -0.363085280452], 15.895044126, 0),
    "moons1": (S.moons(20, 3, 0.1), [0.977212495983, 0.371511499272], [1.524542043832, -0.389068673665], 15.918155978, 0),
}


@pytest.mark.parametrize("name", list(FROZEN))
def test_frozen_values_per_generator(name):
    inst, first, last, total, ysum = FROZEN[name]
    assert inst.X[0].tolist() == pytest.approx(first, rel=1e-9, abs=1e-9)
    assert inst.X[-1].tolist() == pytest.approx(last, rel=1e-9, abs=1e-9)
    assert float(inst.X.sum()) == pytest.approx(total, rel=1e-9, abs=1e-8) and int(inst.y.sum()) == ysum


def test_frozen_order_and_label_operations():
    assert S.order_of(8, 3) == [5, 1, 0, 7, 6, 3, 4, 2]
    flipped = S.flip_labels(S.rings(20, 3), 0.1, 3)
    assert [int(v) for v in flipped.y] == [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]
    assert [int(v) for v in S.random_labels(S.rings(20, 3), 3).y] == [-1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1]


def test_make_instance_dispatch_and_unknown_name():
    assert S.make_instance(C.DATEN_RINGE, 10, 1).name == "Ringe mit Lücke" and S.make_instance(C.DATEN_LIEFER, 10, 1).name == "Liefergebiet (Kreis)"
    with pytest.raises(ValueError):
        S.make_instance("unbekannt", 10, 1)
