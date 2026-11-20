"""Kerne und Gram-Matrizen (Einheiten: Kernfunktion, Gram-Matrix, Bias als K + 1, expliziter Merkmalsraum des Polynomkerns).

Ein Kern ist hier eine Spezifikation (Tupel) und wird erst von `kernel_matrix` ausgewertet:
    ("linear",)            K(x, y) = x·y
    ("poly", d)            K(x, y) = (x·y + 1)^d, d = 2, 3 (c = 1 immer)
    ("gauss", sigma)       K(x, y) = exp(−‖x − y‖² / (2 σ²)); γ = 1/(2 σ²) steht nur als Hinweis neben σ
    ("sigmoid", a, c)      K(x, y) = tanh(a x·y + c); nur als Falle (kein Skalarprodukt)

Der Bias steht als Zusatzkoordinate: K̃ = K + 1 (`gram_tilde`). Alles hier ist reine Funktion ohne Zustand."""

import itertools
import math

import numpy as np

KERNEL_ARTEN = ("linear", "poly", "gauss", "sigmoid")


def kernel_matrix(spec: tuple, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Gram-Matrix K[i, j] = K(X[i], Y[j]) (Y = X, wenn nicht angegeben)."""
    X = np.asarray(X, float)
    same = Y is None
    Y = X if same else np.asarray(Y, float)
    art = spec[0]
    if art == "linear":
        K = X @ Y.T
    elif art == "poly":
        K = (X @ Y.T + 1.0) ** int(spec[1])
    elif art == "gauss":
        sigma = float(spec[1])
        d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        K = np.exp(-d2 / (2.0 * sigma ** 2))
    elif art == "sigmoid":
        K = np.tanh(float(spec[1]) * (X @ Y.T) + float(spec[2]))
    else:
        raise ValueError(f"unbekannter Kern: {spec!r}")
    return np.ascontiguousarray((K + K.T) / 2.0) if same else K  # exakt symmetrisch (a + a)/2 = a: das Perceptron liest Zeilen statt Spalten


def gram_tilde(spec: tuple, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """K̃ = K + 1: der Bias als Zusatzkoordinate 1 im Merkmalsraum."""
    return kernel_matrix(spec, X, Y) + 1.0


def gamma_of(sigma: float) -> float:
    """γ = 1/(2 σ²), einmal festgelegt (die Bandbreite σ ist der Regler)."""
    return 1.0 / (2.0 * float(sigma) ** 2)


def kernel_name(spec: tuple) -> str:
    art = spec[0]
    if art == "linear":
        return "Linear"
    if art == "poly":
        return f"Polynom {int(spec[1])}"
    if art == "gauss":
        return f"Gauß σ = {fmt_num(spec[1])}"
    if art == "sigmoid":
        return f"Sigmoid a = {fmt_num(spec[1])}, c = {fmt_num(spec[2])}"
    raise ValueError(spec)


def fmt_num(x: float) -> str:
    """Zahl mit deutschem Dezimalkomma, ohne überflüssige Nullen (0,25; 8; −1)."""
    s = f"{float(x):.10g}".replace("-", "−")
    return s.replace(".", ",")


def has_feature_space(spec: tuple) -> bool:
    """Linear, Polynom und Gauß sind Skalarprodukte (Gauß: unendlichdimensional); das Sigmoid ist im Allgemeinen keines."""
    return spec[0] in ("linear", "poly", "gauss")


# ------------------------------------------------------------------ Aufwand und expliziter Merkmalsraum des Polynomkerns
def poly_dimension(p: int, d: int) -> int:
    """Zahl der Merkmale D = C(p + d, d) des Polynomkerns (x·y + 1)^d bei p Eingabekoordinaten."""
    return math.comb(p + d, d)


def kernel_multiplications(p: int, d: int) -> int:
    """Multiplikationen je Kernauswertung (x·y + 1)^d: p für das Skalarprodukt und d − 1 für die Potenz (die Addition von 1 zählt nicht)."""
    return p + d - 1


def break_even_points(p: int, d: int, with_feature_build: bool = False) -> int:
    """Größte Zahl n_SV verschiedener Punkte mit α > 0, bei der die duale Auswertung f(x) noch billiger ist als das Skalarprodukt im expliziten Merkmalsraum.

    Voraussetzung: es zählt nur die Auswertung an EINEM neuen Punkt, nicht das Training. Ohne Bau: der explizite Weg kostet D Multiplikationen (φ(x) liegt
    vor), der duale Weg n_SV · (p + d − 1). Mit Bau von φ(x) (Annahme: mindestens eine Multiplikation je Merkmal, untere Abschätzung) kostet der explizite
    Weg 2 D. Gleichstand zählt als nicht billiger."""
    explicit = poly_dimension(p, d) * (2 if with_feature_build else 1)
    k = kernel_multiplications(p, d)
    n = (explicit - 1) // k  # größtes n mit n * k < explicit
    return int(n)


def poly_features(X: np.ndarray, d: int, c: float = 1.0) -> np.ndarray:
    """Expliziter Merkmalsraum des Polynomkerns (x·y + c)^d (Multinomialsatz): φ(x)·φ(y) = (x·y + c)^d. Dimension C(p + d, d); nur für kleines p gedacht."""
    X = np.asarray(X, float)
    n, p = X.shape
    if poly_dimension(p, d) > 20000:
        raise ValueError("expliziter Merkmalsraum zu groß")
    cols = []
    for combo in itertools.combinations_with_replacement(range(p + 1), d):  # Index p steht für die Konstante c
        counts = [combo.count(i) for i in range(p + 1)]
        coeff = math.factorial(d)
        for k in counts:
            coeff //= math.factorial(k)
        weight = math.sqrt(coeff * c ** counts[p])
        col = np.full(n, weight)
        for i in range(p):
            if counts[i]:
                col = col * X[:, i] ** counts[i]
        cols.append(col)
    return np.stack(cols, axis=1)


def with_bias_column(Phi: np.ndarray) -> np.ndarray:
    """(φ, 1): der Bias als Zusatzkoordinate; Skalarprodukt = K + 1."""
    return np.hstack([Phi, np.ones((Phi.shape[0], 1))])


def smallest_eigenvalue(K: np.ndarray) -> float:
    """Kleinster Eigenwert der symmetrischen Gram-Matrix (positiv semidefinit: ≥ −n·eps·λmax; Gauß bei verschiedenen Punkten positiv definit)."""
    return float(np.linalg.eigvalsh((K + K.T) / 2.0)[0])
