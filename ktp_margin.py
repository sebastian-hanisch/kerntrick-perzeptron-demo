"""Größte Trennung und Novikoff-Schranke: ein kleines duales QP mit Aktivmengenverfahren und Zertifikat (nur numpy, kein Löser von außen).

Duales Problem (harte Marge): min ½ αᵀQα − 1ᵀα, α ≥ 0 mit Q_ij = y_i y_j K_ij, bei freiem Versatz b zusätzlich yᵀα = 0.
 * **Marge in (φ, 1)** (Bias als Koordinate, Gram-Matrix K̃ = K + 1, keine Gleichung): so rechnet die Schranke R̃²‖w*‖² ≥ Korrekturen.
 * **Marge in φ mit freiem b** (das Maß von Stück 2): Gram-Matrix K, Gleichung yᵀα = 0.
Marge heißt 2/‖w‖ (Breite zwischen den Klassen); der Abstand eines Punktes zur Ebene ist 1/‖w‖ bei y f = 1.
Zertifikat: Dualitätslücke |Σα − αᵀQα| ≤ 1e-6·max(1, ‖w*‖²) und kleinste Randbedingung y f ≥ 1 − 1e-4 (sonst `certified = False`, dann gibt es keine Zahl).
Das Verfahren (Lawson–Hanson-artig, die Teilprobleme werden exakt gelöst) liefert für nicht trennbare Daten kein Zertifikat; es endet an der Iterationsgrenze."""

from dataclasses import dataclass

import numpy as np

GAP_TOL = 1e-6
CONSTRAINT_TOL = 1e-4
VIOLATION_TOL = 1e-9
RIDGE = 1e-12  # relativer Zuschlag auf die Diagonale der Teilprobleme (Q ist bei endlichdimensionalem Merkmalsraum singulär)


@dataclass
class QPResult:
    alpha: np.ndarray        # duale Lösung (α ≥ 0)
    b: float                 # Versatz (0 in (φ, 1))
    w2: float                # ‖w*‖² = αᵀQα
    gap: float               # |Σα − αᵀQα|
    min_constraint: float    # kleinstes y_i f(x_i), soll ≥ 1 sein
    iterations: int
    certified: bool
    n_support: int

    @property
    def margin(self) -> float:
        """Marge 2/‖w*‖ (Breite)."""
        return 2.0 / float(np.sqrt(self.w2)) if self.w2 > 0 else float("inf")

    @property
    def half_margin(self) -> float:
        """Abstand zur Ebene 1/‖w*‖."""
        return 1.0 / float(np.sqrt(self.w2)) if self.w2 > 0 else float("inf")


def _solve_kkt(Q: np.ndarray, y: np.ndarray, F: np.ndarray, free_b: bool):
    """Teilproblem auf der Menge F: Q_FF z (+ y_F b) = 1, yᵀz = 0 bei freiem b. Rückgabe (z, b)."""
    QFF = Q[np.ix_(F, F)] + RIDGE * float(np.max(np.diag(Q))) * np.eye(len(F))
    k = len(F)
    if free_b:
        M = np.zeros((k + 1, k + 1))
        M[:k, :k] = QFF
        M[:k, k] = y[F]
        M[k, :k] = y[F]
        rhs = np.r_[np.ones(k), 0.0]
    else:
        M, rhs = QFF, np.ones(k)
    try:
        sol = np.linalg.solve(M, rhs)
        if not np.all(np.isfinite(sol)) or np.linalg.norm(M @ sol - rhs) > 1e-7 * (1.0 + np.linalg.norm(rhs)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(M, rhs, rcond=1e-13)[0]
    return (sol[:k], float(sol[k])) if free_b else (sol, 0.0)


def solve_dual_qp(K: np.ndarray, y: np.ndarray, free_b: bool, max_iter: int | None = None) -> QPResult:
    """Größte Trennung (harte Marge) für die Gram-Matrix K (in (φ, 1): K̃ übergeben und free_b = False; in φ mit freiem b: K und free_b = True)."""
    y = np.asarray(y, float)
    n = len(y)
    Q = (y[:, None] * y[None, :]) * K
    cap = max_iter if max_iter is not None else 8 * n + 100
    alpha = np.zeros(n)
    b = 0.0
    inF = np.zeros(n, dtype=bool)
    it = 0
    if free_b:
        pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
        if len(pos) == 0 or len(neg) == 0:
            return QPResult(alpha, 0.0, 0.0, 0.0, 0.0, 0, False, 0)
        d = np.diag(K)
        best = (np.inf, 0, 0)
        for i in pos:  # nächstes Paar im Merkmalsraum als Start
            dist = d[i] + d[neg] - 2.0 * K[i, neg]
            j = int(np.argmin(dist))
            if dist[j] < best[0]:
                best = (float(dist[j]), int(i), int(neg[j]))
        inF[best[1]] = inF[best[2]] = True
    while it < cap:
        it += 1
        if inF.any():
            F = np.flatnonzero(inF)
            z, bz = _solve_kkt(Q, y, F, free_b)
            if np.all(z > 0.0):
                alpha = np.zeros(n)
                alpha[F] = z
                b = bz
            else:
                # Schritt vom alten (zulässigen) α zur Lösung z, solange α ≥ 0 bleibt (Lawson–Hanson, innere Schleife)
                old = alpha[F]
                neg_mask = z <= 0.0
                denom = old[neg_mask] - z[neg_mask]
                theta = float(np.min(old[neg_mask] / np.where(denom > 0, denom, np.inf)))
                theta = min(max(theta, 0.0), 1.0)
                new = old + theta * (z - old)
                alpha = np.zeros(n)
                alpha[F] = new
                b = b + theta * (bz - b) if free_b else 0.0
                drop = F[new <= 1e-14]
                if len(drop) == 0:  # numerisch: kleinsten Eintrag entfernen, damit es weitergeht
                    drop = F[[int(np.argmin(new))]]
                alpha[drop] = 0.0
                inF[drop] = False
                if free_b and (not (y[inF] > 0).any() or not (y[inF] < 0).any()):
                    inF[:] = False
                    alpha[:] = 0.0
                    b = 0.0
                    pos, neg = np.flatnonzero(y > 0), np.flatnonzero(y < 0)
                    d = np.diag(K)
                    i = int(pos[np.argmin(d[pos])])
                    j = int(neg[np.argmin(d[neg] - 2.0 * K[i, neg])])
                    inF[i] = inF[j] = True
                continue
        # Verletzungen der Randbedingung y f ≥ 1 außerhalb von F
        f_y = Q @ alpha + y * b  # = y_i f(x_i)
        viol = 1.0 - f_y
        viol[inF] = -np.inf
        j = int(np.argmax(viol))
        if viol[j] <= VIOLATION_TOL:
            break
        inF[j] = True
    v = alpha * y
    w2 = float(v @ K @ v)
    f = K @ v + b
    min_c = float(np.min(y * f))
    gap = abs(float(alpha.sum()) - w2)
    certified = bool(it < cap and min_c >= 1.0 - CONSTRAINT_TOL and gap <= GAP_TOL * max(1.0, w2) and np.all(alpha >= 0.0)
                     and (not free_b or abs(float(v.sum())) <= 1e-6 * max(1.0, float(alpha.sum()))))
    return QPResult(alpha, b, w2, gap, min_c, it, certified, int((alpha > 0).sum()))


def max_margin(K: np.ndarray, y: np.ndarray, free_b: bool, max_iter: int | None = None) -> QPResult:
    """Größte Trennung: freier Versatz mit K, sonst Bias als Koordinate mit K̃ = K + 1."""
    return solve_dual_qp(K if free_b else K + 1.0, y, free_b, max_iter)


def novikoff_bound(Kt: np.ndarray, qp: QPResult) -> float:
    """R̃²‖w*‖² mit R̃² = max K̃(x, x) und ‖w*‖ der größten Trennung in (φ, 1): obere Schranke der Korrekturen, wenn trennbar (Rechnung dieser README)."""
    return float(np.max(np.diag(Kt)) * qp.w2)


def found_margin_ratio(Kt: np.ndarray, K: np.ndarray, y: np.ndarray, alpha: np.ndarray, qp_free: QPResult | None, qp_tilde: QPResult | None):
    """Marge der vom Perceptron gefundenen Trennung im Verhältnis zur größten, zwei Maße. Rückgabe (Verhältnis in φ mit freiem b, Verhältnis in (φ, 1)).

    Gefunden: w = Σ α_j y_j φ̃(x_j), f = K̃ (α ∘ y); Abstand der Punkte zur Ebene min y f/‖·‖. In φ mit freiem b zählt nur w_φ (Norm mit K), der Versatz
    b = Σ α_j y_j geht nicht in die Norm ein. Fehlt ein Zertifikat, steht None."""
    v = alpha * y
    f = Kt @ v
    min_yf = float(np.min(y * f))
    ratio_free = ratio_tilde = None
    if qp_free is not None and qp_free.certified and min_yf > 0:
        w_phi = float(np.sqrt(max(v @ K @ v, 0.0)))
        if w_phi > 0:
            ratio_free = (min_yf / w_phi) / qp_free.half_margin
    if qp_tilde is not None and qp_tilde.certified and min_yf > 0:
        w_t = float(np.sqrt(max(v @ Kt @ v, 0.0)))
        if w_t > 0:
            ratio_tilde = (min_yf / w_t) / qp_tilde.half_margin
    return ratio_free, ratio_tilde
