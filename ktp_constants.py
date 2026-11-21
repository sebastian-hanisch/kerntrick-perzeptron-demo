"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Kern-Auswahllisten und Presets."""

import math

from ktp_kernels import kernel_name

# ------------------------------------------------------------------ Erzeuger
RING_INNER, RING_OUTER, RING_WIDTH = 1.0, 2.0, 0.1       # Radius 1 und 2, Breite 0,1 (Lücke 0,8 im Radius)
DELIVERY_RADIUS = math.sqrt(2.0 / math.pi)               # Kreis mit halber Fläche des Quadrats [−1, 1]²
MOONS_NOISE = 0.1                                        # Halbbreite des beschränkten Rauschens der Monde mit Rauschen
N_TEST = 1000                                            # frische Testpunkte für den Testfehler gegen die wahre Klasse
D2_NOISE = 0.1                                           # Anteil vertauschter Labels („10 % vertauscht“)

# Strom-Kennungen: je Zweck ein eigener SplitMix64-Strom aus demselben Zufalls-Seed
SALT_ORDER = 0x5851F42D4C957F2D
SALT_FLIP = 0x14057B7EF767814F
SALT_RANDOM_LABELS = 0x2545F4914F6CDD1D
SALT_TEST = 0x6A09E667F3BCC909
SALT_SHUFFLE = 0x3C6EF372FE94F82B

# ------------------------------------------------------------------ Regler
SEED_MAX, DEFAULT_SEED = 999999, 7
EPOCH_CHOICES = (100, 300, 1000, 3000, 10000, 50000)
DEFAULT_EPOCHS = 3000
MAX_UPDATES = 300000                                     # Korrekturgrenze (eine Grenze für alle Läufe)
SPINNER_EPOCHS = 3000                                    # über dieser Epochengrenze läuft die Rechnung mit Spinner und Zeitangabe
N_MIN, N_MAX, N_STEP, DEFAULT_N = 20, 200, 2, 60
P_MIN, P_MAX, DEFAULT_P = 2, 1000, 10
DEGREES = (2, 3)
DEFAULT_DEGREE = 3
SIGMAS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # Bandbreite in den Einheiten der Daten (logarithmisches Gitter)
DEFAULT_SIGMA = 1.0
QP_LIVE_MAX_N = 100                                      # darüber wird das Schranken-QP nicht live gerechnet
SWEEP_SEEDS_RINGS = 30                                   # Zufalls-Seeds der Bandbreitenkurve (eingefroren)

DATEN_RINGE = "Ringe mit Lücke"
DATEN_LIEFER = "Liefergebiet (Kreis)"
DATEN_MONDE = "Monde (ohne Rauschen)"
DATEN_MONDE_R = "Monde (Rauschen 0,1)"
EX3_DATEN = (DATEN_RINGE, DATEN_LIEFER, DATEN_MONDE, DATEN_MONDE_R)
EX5_DATEN = ("Ringe mit Lücke (n = 100)", "Liefergebiet (n = 200)")
EX5_ETIKETTEN = ("wahr", "10 % vertauscht", "zufällig")

# Kerne in den Auswahllisten (Name -> Spezifikation); kein Regler, der unter einer anderen Einstellung wirkungslos wäre
XOR_KERNELS = {kernel_name(k): k for k in (("linear",), ("poly", 2), ("gauss", 0.5), ("gauss", 1.0), ("gauss", 2.0), ("gauss", 4.0))}
EX3_KERNELS = {kernel_name(k): k for k in (("linear",), ("poly", 2), ("poly", 3), ("gauss", 0.25), ("gauss", 0.5), ("gauss", 1.0), ("gauss", 2.0))}
EX5_KERNELS = {kernel_name(k): k for k in (("gauss", 0.05), ("gauss", 0.1), ("gauss", 0.25), ("gauss", 0.5), ("poly", 2),
                                          ("sigmoid", 0.1, -1.0), ("sigmoid", 0.5, -1.0))}


def _all_orders():
    import itertools
    return [", ".join(p) for p in itertools.permutations("abcd")]


XOR_ORDERS = _all_orders()

DEFAULT_XOR_KERNEL = kernel_name(("poly", 2))
DEFAULT_XOR_ORDER = "a, b, c, d"
DEFAULT_EX3_DATEN = DATEN_RINGE
DEFAULT_EX3_KERNEL = kernel_name(("gauss", 1.0))
DEFAULT_EX5_DATEN = EX5_DATEN[0]
DEFAULT_EX5_ETIKETTEN = "wahr"
DEFAULT_EX5_KERNEL = kernel_name(("gauss", 0.25))

COLOR_POS = "#2F6B65"
COLOR_NEG = "#B5543B"
COLOR_LINE = "#4c78a8"
COLOR_BAND = "rgba(76,120,168,0.22)"

# ------------------------------------------------------------------ Presets: nur die Regler, die das Szenario ändert (der Rest bleibt Standard)
PRESETS = {
    "XOR von Hand (vier Punkte)": {},
    "Ringe: Innenstadt gegen Umland": {"ex3_n": 100},
    "Liefergebiet ohne Lücke": {"ex3_daten": DATEN_LIEFER, "ex3_kern": kernel_name(("poly", 2)), "ex3_n": 100},
    "Monde": {"ex3_daten": DATEN_MONDE_R, "ex3_kern": kernel_name(("gauss", 0.5)), "ex3_n": 100},
    "Zufalls-Labels": {"ex5_etiketten": "zufällig", "ex5_kern": kernel_name(("gauss", 0.1))},
    "Schmale Bandbreite (σ = 0,05)": {"ex4_sigma": 0.05},
    "Breite Bandbreite (σ = 8)": {"ex4_sigma": 8.0},
}
PRESET_HELP = {
    "XOR von Hand (vier Punkte)": "Experiment 1: vier Punkte, Kern (x·y + 1)², Reihenfolge a, b, c, d. Das ist die Voreinstellung aller Regler.",
    "Ringe: Innenstadt gegen Umland": "Experiment 3: zwei Ringe (Radius 1 und 2), 100 Punkte, Gauß-Kern σ = 1.",
    "Liefergebiet ohne Lücke": "Experiment 3: Kunden gleichverteilt im Quadrat, Klasse +1 im Kreis mit halber Fläche, 100 Punkte, Polynom 2.",
    "Monde": "Experiment 3: zwei Mondbögen mit Rauschen 0,1, 100 Punkte, Gauß-Kern σ = 0,5.",
    "Zufalls-Labels": "Experiment 5: dieselben Ringe mit zufällig vertauschten Labels, Gauß-Kern σ = 0,1: das Perceptron meldet „konvergiert“.",
    "Schmale Bandbreite (σ = 0,05)": "Experiment 4: sehr schmaler Gauß-Kern, die Gram-Matrix ähnelt der Einheitsmatrix.",
    "Breite Bandbreite (σ = 8)": "Experiment 4: sehr breiter Gauß-Kern, die Gram-Matrix ähnelt der Einsmatrix.",
}
