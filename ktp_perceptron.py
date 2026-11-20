"""Das Perceptron in dualer Form (Kerntrick) und primal im expliziten Merkmalsraum; je Regel eine kleine Funktion.

Regel (primal): w ← w + y_i φ̃(x_i), wenn y_i ⟨w, φ̃(x_i)⟩ ≤ 0 (Korrektur; der Fall 0 zählt als Fehler). Startwert w = 0, deshalb ist die Lernrate ohne Wirkung.
Dual: α_i zählt die Korrekturen des Punktes i, w = Σ_j α_j y_j φ̃(x_j), f(x) = Σ_j α_j y_j K̃(x_j, x), Korrektur α_i ← α_i + 1.
**Der Kerntrick ist die Ersetzung ⟨φ̃(x), φ̃(x′)⟩ → K̃(x, x′) und sonst nichts.** K̃ = K + 1 (Bias als Zusatzkoordinate).
Zähleinheiten: Korrekturen (α_i += 1), Epochen (ein Durchlauf in fester Reihenfolge), konvergiert = eine Epoche ohne Korrektur.
„Grenze erreicht“ heißt nicht „nicht trennbar“."""

from dataclasses import dataclass, field

import numpy as np

import ktp_constants as C
from ktp_rng import SplitMix64

STOP_CONVERGED = "konvergiert"
STOP_EPOCHS = "Epochengrenze"
STOP_UPDATES = "Korrekturgrenze"
STOP_CYCLE = "Zyklus"


@dataclass
class RunResult:
    alpha: np.ndarray            # Zahl der Korrekturen je Punkt (ganzzahlig, als float geführt)
    s: np.ndarray                # s = K̃ (α ∘ y): die Werte f(x_i) an den Trainingspunkten
    updates: int                 # Korrekturen insgesamt (Σ α)
    epochs: int                  # durchlaufene Epochen (bei Konvergenz inklusive der letzten ohne Korrektur)
    stop: str                    # STOP_*
    cycle_epochs: int | None = None   # Länge des Zyklus in Epochen, nur bei STOP_CYCLE
    cycle_updates: int | None = None  # Korrekturen in einem Zyklus

    @property
    def converged(self) -> bool:
        return self.stop == STOP_CONVERGED


def first_error_in_order(y: np.ndarray, s: np.ndarray, idx: np.ndarray, pos: int) -> int:
    """Position k ≥ pos in der Reihenfolge `idx` des ersten Punktes mit y_i s_i ≤ 0 (Fehler), sonst −1. Eine Einheit der Regel: nur die Prüfung."""
    seg = idx[pos:]
    bad = np.flatnonzero(y[seg] * s[seg] <= 0.0)
    return int(pos + bad[0]) if bad.size else -1


def apply_update(alpha: np.ndarray, s: np.ndarray, Kt: np.ndarray, y: np.ndarray, i: int) -> None:
    """Korrektur des Punktes i: α_i += 1 und s += y_i K̃[:, i] (s = K̃ (α ∘ y) wird mitgeführt; eine Prüfung kostet danach O(1))."""
    alpha[i] += 1.0
    s += y[i] * Kt[i]  # K̃ ist symmetrisch: die Zeile i ist die Spalte i (zusammenhängend im Speicher)


def run_dual(Kt: np.ndarray, y: np.ndarray, order: list[int], max_epochs: int, max_updates: int = C.MAX_UPDATES,
             shuffle_seed: int | None = None, detect_cycle: bool = False) -> RunResult:
    """Duales Perceptron. `order`: feste Reihenfolge je Epoche; mit `shuffle_seed` wird sie in jeder Epoche neu gemischt (SplitMix64, `shuffle`).
    `detect_cycle`: kehrt der Zustand s am Epochenanfang exakt wieder, ist es ein Zyklus (nur für kleine, ganzzahlige Beispiele gedacht)."""
    n = len(y)
    alpha = np.zeros(n)
    s = np.zeros(n)
    updates = 0
    rng = SplitMix64(shuffle_seed) if shuffle_seed is not None else None
    idx = np.array(order, dtype=int)
    seen: dict[bytes, tuple[int, int]] = {}
    for epoch in range(1, max_epochs + 1):
        if detect_cycle:
            key = s.tobytes()
            if key in seen:
                return RunResult(alpha, s, updates, epoch - 1, STOP_CYCLE, cycle_epochs=epoch - seen[key][0], cycle_updates=updates - seen[key][1])
            seen[key] = (epoch, updates)
        if rng is not None:
            lst = list(range(n))
            rng.shuffle(lst)
            idx = np.array(lst, dtype=int)
        pos = 0
        mistakes = 0
        while pos < n:
            k = first_error_in_order(y, s, idx, pos)
            if k < 0:
                break
            apply_update(alpha, s, Kt, y, int(idx[k]))
            updates += 1
            mistakes += 1
            pos = k + 1
            if updates >= max_updates:
                return RunResult(alpha, s, updates, epoch, STOP_UPDATES)
        if mistakes == 0:
            return RunResult(alpha, s, updates, epoch, STOP_CONVERGED)
    return RunResult(alpha, s, updates, max_epochs, STOP_EPOCHS)


@dataclass
class TraceStep:
    epoch: int
    point: int
    f: float
    error: bool
    alpha: tuple


def trace_dual(Kt: np.ndarray, y: np.ndarray, order: list[int], max_epochs: int) -> tuple[list[TraceStep], RunResult]:
    """Die Regel Punkt für Punkt, ohne mitgeführten Vektor s (f wird jedes Mal neu summiert): Protokoll und Bezugsrechnung für `run_dual`."""
    n = len(y)
    alpha = np.zeros(n)
    log: list[TraceStep] = []
    updates = 0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for i in order:
            f = float(sum(alpha[j] * y[j] * Kt[j, i] for j in range(n)))
            err = y[i] * f <= 0.0
            if err:
                alpha[i] += 1.0
                updates += 1
                mistakes += 1
            log.append(TraceStep(epoch, int(i), f, bool(err), tuple(alpha)))
        if mistakes == 0:
            s = Kt @ (alpha * y)
            return log, RunResult(alpha, s, updates, epoch, STOP_CONVERGED)
    s = Kt @ (alpha * y)
    return log, RunResult(alpha, s, updates, max_epochs, STOP_EPOCHS)


def run_primal(Phi: np.ndarray, y: np.ndarray, order: list[int], max_epochs: int, max_updates: int = C.MAX_UPDATES):
    """Primales Perceptron im expliziten Merkmalsraum Phi (Bias als letzte Spalte schon enthalten): Bezugsrechnung für den Kerntrick.
    Rückgabe (w, Korrekturen, Epochen, konvergiert)."""
    w = np.zeros(Phi.shape[1])
    updates = 0
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for i in order:
            if y[i] * float(w @ Phi[i]) <= 0.0:
                w = w + y[i] * Phi[i]
                updates += 1
                mistakes += 1
                if updates >= max_updates:
                    return w, updates, epoch, False
        if mistakes == 0:
            return w, updates, epoch, True
    return w, updates, max_epochs, False


def decision_values(K_test_train: np.ndarray, alpha: np.ndarray, y: np.ndarray) -> np.ndarray:
    """f(x) = Σ_j α_j y_j K̃(x_j, x) an neuen Punkten (K̃ = K + 1, Zeilen: neue Punkte)."""
    return K_test_train @ (alpha * y)


def weight_norm2(Kt: np.ndarray, alpha: np.ndarray, y: np.ndarray) -> float:
    """‖w‖² = (α ∘ y)ᵀ K̃ (α ∘ y), nur mit der Gram-Matrix gerechnet."""
    v = alpha * y
    return float(v @ Kt @ v)


def error_rate(f: np.ndarray, y: np.ndarray) -> float:
    """Anteil der Punkte mit y f ≤ 0 (der Fall f = 0 zählt wie in der Regel als Fehler)."""
    return float(np.mean(y * f <= 0.0))
