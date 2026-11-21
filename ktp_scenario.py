"""Erzeuger der Instanzen: alle ausgedacht (Modelle, keine Messdaten), Zufallsquelle überall SplitMix64 (`ktp_rng.py`).

Erzeuger: „XOR (vier Punkte)“, „Ringe mit Lücke“ (zwei konzentrische Ringe, Radius 1 und 2, Breite 0,1; Lücke 0,8 im Radius), „Liefergebiet (Kreis)“
(Kunden gleichverteilt in [−1, 1]², Klasse +1 im Kreis mit halber Fläche, keine Lücke), „Monde“ (zwei Halbkreisbögen, beschränktes Rauschen).
Gleichverteilt in [a, b) heißt a + (b − a)·`random()`. Reihenfolgen der Punkte sind `shuffle` der Liste 0 … n−1 mit eigener Strom-Kennung je Zweck."""

import math
from dataclasses import dataclass

import numpy as np

import ktp_constants as C
from ktp_rng import SplitMix64


def _unif(rng: SplitMix64, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


@dataclass(frozen=True)
class Instance:
    """Punkte X (n × 2), Labels y ∈ {−1, +1} (die gelernten), y_true die wahre Klasse nach der Erzeugerregel."""

    name: str
    X: np.ndarray
    y: np.ndarray
    y_true: np.ndarray
    seed: int

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


# ------------------------------------------------------------------ XOR (vier Punkte)
XOR_NAMES = "abcd"


def xor_instance() -> Instance:
    """a = (1, 1), b = (−1, −1) mit y = +1, c = (1, −1), d = (−1, 1) mit y = −1."""
    X = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    return Instance("XOR (vier Punkte)", X, y, y.copy(), 0)


# ------------------------------------------------------------------ Ringe mit Lücke
def rings(n: int, seed: int) -> Instance:
    """n Punkte (n gerade): n/2 auf dem inneren Ring (y = −1, Radius 1 ± 0,1), n/2 auf dem äußeren (y = +1, Radius 2 ± 0,1); Winkel gleichverteilt."""
    n_per = n // 2
    rng = SplitMix64(seed)
    pts, labels = [], []
    for radius, label in ((C.RING_INNER, -1.0), (C.RING_OUTER, 1.0)):
        for _ in range(n_per):
            ang = _unif(rng, 0.0, 2.0 * math.pi)
            r = radius + _unif(rng, -C.RING_WIDTH, C.RING_WIDTH)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
            labels.append(label)
    X = np.array(pts)
    y = np.array(labels)
    return Instance("Ringe mit Lücke", X, y, y.copy(), seed)


# ------------------------------------------------------------------ Liefergebiet (Kreis)
def delivery_area(n: int, seed: int) -> Instance:
    """Kunden gleichverteilt in [−1, 1]²; Klasse +1 innerhalb des Kreises um das Depot (Fläche halb so groß wie das Quadrat), sonst −1; keine Lücke."""
    rng = SplitMix64(seed)
    X = np.array([(_unif(rng, -1.0, 1.0), _unif(rng, -1.0, 1.0)) for _ in range(n)])
    y = np.where(np.sqrt((X ** 2).sum(1)) <= C.DELIVERY_RADIUS, 1.0, -1.0)
    return Instance("Liefergebiet (Kreis)", X, y, y.copy(), seed)


# ------------------------------------------------------------------ Monde
def moons(n: int, seed: int, noise: float) -> Instance:
    """Zwei Halbkreisbögen (oberer: y = −1, unterer: y = +1), Winkel gleichverteilt in [0, π); `noise` ist die Halbbreite eines gleichverteilten
    (also beschränkten) Rauschens je Koordinate. Eigener Erzeuger; `make_moons` ist nur Fremdreferenz im Test."""
    n_per = n // 2
    rng = SplitMix64(seed)
    pts, labels = [], []
    for k, label in ((0, -1.0), (1, 1.0)):
        for _ in range(n_per):
            ang = _unif(rng, 0.0, math.pi)
            if k == 0:
                x1, x2 = math.cos(ang), math.sin(ang)
            else:
                x1, x2 = 1.0 - math.cos(ang), 0.5 - math.sin(ang)
            pts.append((x1 + _unif(rng, -noise, noise), x2 + _unif(rng, -noise, noise)))
            labels.append(label)
    X = np.array(pts)
    y = np.array(labels)
    return Instance("Monde", X, y, y.copy(), seed)


# ------------------------------------------------------------------ Labels ändern
def flip_labels(inst: Instance, fraction: float, seed: int) -> Instance:
    """Genau round(fraction · n) Labels vertauscht (Auswahl ohne Zurücklegen per `shuffle` der Indizes), die wahre Klasse bleibt."""
    idx = list(range(inst.n))
    SplitMix64(seed ^ C.SALT_FLIP).shuffle(idx)
    k = int(round(fraction * inst.n))
    y = inst.y.copy()
    for i in idx[:k]:
        y[i] = -y[i]
    return Instance(inst.name, inst.X, y, inst.y_true, inst.seed)


def random_labels(inst: Instance, seed: int) -> Instance:
    """Zufalls-Labels: die Labels der gleichen Punkte werden per `shuffle` vertauscht (gleiche Zahl +1 und −1 wie vorher, wahre Klasse bleibt)."""
    labels = [float(v) for v in inst.y]
    SplitMix64(seed ^ C.SALT_RANDOM_LABELS).shuffle(labels)
    return Instance(inst.name, inst.X, np.array(labels), inst.y_true, inst.seed)


def order_of(n: int, seed: int) -> list[int]:
    """Feste Reihenfolge der Punkte je Zufalls-Seed: `shuffle` der Liste 0 … n−1."""
    order = list(range(n))
    SplitMix64(seed ^ C.SALT_ORDER).shuffle(order)
    return order


def make_instance(daten: str, n: int, seed: int, noise: float = 0.0) -> Instance:
    """Instanz nach dem Namen aus `C.DATEN`."""
    if daten == C.DATEN_RINGE:
        return rings(n, seed)
    if daten == C.DATEN_LIEFER:
        return delivery_area(n, seed)
    if daten == C.DATEN_MONDE:
        return moons(n, seed, noise)
    raise ValueError(daten)


def fresh_test_points(daten: str, n_test: int, seed: int, noise: float = 0.0) -> Instance:
    """Frische Testpunkte desselben Erzeugers (eigener Zufallsstrom), wahre Klasse nach der Erzeugerregel."""
    return make_instance(daten, n_test, seed ^ C.SALT_TEST, noise)
