"""Auswertung: Analysen je Experiment (reine Funktionen, keine Streamlit-Aufrufe) und die Verdikte der App als reine Funktionen `verdict_*(...) -> (Art, Text)`.

Jede Zahl im Verdikt stammt entweder aus der Rechnung dieser Instanz (live) oder aus `ktp_sweep.json` (eingefroren, mit `tools/ktp_make_sweep.py` neu gemessen);
die Schwellen stehen als Konstanten hier, jede Zahl im Text ist in `tests/test_verdicts.py` an handgesetzten Werten geprüft."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

import ktp_constants as C
import ktp_kernels as KK
import ktp_margin as M
import ktp_perceptron as P
import ktp_scenario as S

SWEEP_PATH = Path(__file__).resolve().parent / "ktp_sweep.json"


class Settings(NamedTuple):
    """Alle Regler (Schlüssel wie in `ktp_presets.SETTING_SPECS`); hashbar für `st.cache_data`."""

    xor_kern: str
    xor_order: str
    ex2_grad: int
    ex2_p: int
    ex3_daten: str
    ex3_kern: str
    ex3_n: int
    ex3_mischen: bool
    ex4_sigma: float
    ex5_daten: str
    ex5_etiketten: str
    ex5_kern: str
    ex5_epochen: int
    seed_input: int


def default_settings() -> Settings:
    import ktp_presets as P_

    return Settings(**{k: spec.default for k, spec in P_.SETTING_SPECS.items()})


_SWEEP_CACHE: dict | None = None


def load_sweep() -> dict:
    """Die eingefrorenen Messreihen (ktp_sweep.json)."""
    global _SWEEP_CACHE
    if _SWEEP_CACHE is None:
        _SWEEP_CACHE = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    return _SWEEP_CACHE


# ------------------------------------------------------------------ Zahlenformat
def de(x, nd: int = 0) -> str:
    """Zahl mit deutschem Dezimalkomma und Leerzeichen als Tausendertrenner (146 490; 0,18)."""
    if isinstance(x, (int, np.integer)) or (nd == 0 and float(x).is_integer() and abs(x) < 1e15):
        s = f"{int(round(float(x))):,}".replace(",", " ")
    else:
        s = f"{float(x):,.{nd}f}".replace(",", " ")
        s = s.replace(".", ",")
    return s.replace("-", "−")


def sci(x: float, nd: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit Dezimalkomma (3,5·10⁻¹⁶)."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** exp
    if round(abs(mant), nd) >= 10:
        mant, exp = mant / 10.0, exp + 1
    sup = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{de(mant, nd)}·10{sup}"


def pct(x: float, nd: int = 0) -> str:
    return de(100.0 * x, nd) + " %"


def parse_order(label: str) -> list[int]:
    return [C_XOR_NAMES.index(ch) for ch in label.replace(",", "").replace(" ", "")]


C_XOR_NAMES = S.XOR_NAMES


# ------------------------------------------------------------------ Experiment 1: XOR von Hand
@dataclass
class XorAnalysis:
    kern: str
    spec: tuple
    order: list[int]
    K: np.ndarray
    Kt: np.ndarray
    run: P.RunResult
    trace: list
    w_after_epoch: list          # nur lineare Form: w = (w1, w2, b) nach jeder Epoche des Protokolls
    all_orders_updates: list     # Korrekturen je Reihenfolge der 24 (None bei Zyklus/Grenze)
    all_orders_same_alpha: bool
    all_orders_converged: int
    qp: M.QPResult | None
    bound: float | None
    formula: str | None


def _formula_poly2(alpha: np.ndarray, y: np.ndarray, X: np.ndarray) -> str | None:
    """f(x) = Σ α_j y_j ((x_j·x + 1)² + 1) als Polynom in x₁, x₂, aus Funktionswerten an einem Gitter angepasst (nur zur Anzeige)."""
    pts = np.array([(a, b) for a in (-1.5, -0.5, 0.5, 1.5) for b in (-1.0, 0.0, 1.0)])
    f = KK.gram_tilde(("poly", 2), pts, X) @ (alpha * y)
    basis = np.stack([np.ones(len(pts)), pts[:, 0], pts[:, 1], pts[:, 0] ** 2, pts[:, 1] ** 2, pts[:, 0] * pts[:, 1]], axis=1)
    coef = np.linalg.lstsq(basis, f, rcond=None)[0]
    names = ["", "x₁", "x₂", "x₁²", "x₂²", "x₁x₂"]
    parts = []
    for c, nm in zip(coef, names):
        if abs(c) < 1e-9:
            continue
        mag = de(abs(c), 0 if abs(c - round(c)) < 1e-9 else 3)
        term = (mag if nm == "" else (nm if abs(abs(c) - 1) < 1e-9 else f"{mag} {nm}"))
        parts.append(("− " if c < 0 else ("+ " if parts else "")) + term)
    return " ".join(parts).replace("+ − ", "− ") if parts else "0"


def analyse_xor(kern_label: str, order_label: str) -> XorAnalysis:
    inst = S.xor_instance()
    spec = C.XOR_KERNELS[kern_label]
    order = parse_order(order_label)
    K = KK.kernel_matrix(spec, inst.X)
    Kt = K + 1.0
    run = P.run_dual(Kt, inst.y, order, 400, C.MAX_UPDATES, detect_cycle=True)
    log, _ = P.trace_dual(Kt, inst.y, order, max(run.epochs, 1))
    w_after = []
    if spec[0] == "linear":
        Phi = KK.with_bias_column(inst.X)
        per_epoch = len(order)
        for e in range(0, len(log), per_epoch):
            alpha = np.array(log[e + per_epoch - 1].alpha)
            w_after.append(tuple(float(v) for v in Phi.T @ (alpha * inst.y)))
    ups, alphas, conv = [], [], 0
    import itertools

    for perm in itertools.permutations(range(4)):
        r = P.run_dual(Kt, inst.y, list(perm), 400, C.MAX_UPDATES, detect_cycle=True)
        if r.converged:
            conv += 1
            ups.append(r.updates)
            alphas.append(tuple(r.alpha))
        else:
            ups.append(None)
    qp = bound = formula = None
    if run.converged:
        qp = M.max_margin(K, inst.y, False)
        if qp.certified:
            bound = M.novikoff_bound(Kt, qp)
        else:
            qp = None
        if spec == ("poly", 2):
            formula = _formula_poly2(run.alpha, inst.y, inst.X)
    return XorAnalysis(kern_label, spec, order, K, Kt, run, log, w_after, ups, len(set(alphas)) == 1 and conv == 24, conv, qp, bound, formula)


# ------------------------------------------------------------------ Experiment 2: Kosten und Kerntrick
@dataclass
class CostAnalysis:
    d: int
    p: int
    rows: list            # (p, D, Multiplikationen des Kerns, Gleichstand ohne Bau, Gleichstand mit Bau)
    updates_dual: int
    updates_primal: int
    same_run: bool
    rel_w: float
    n_support: int        # verschiedene korrigierte Punkte des Laufs (p = 2)
    D2: int
    k2: int


def cost_rows(d: int, ps) -> list:
    return [(p, KK.poly_dimension(p, d), KK.kernel_multiplications(p, d), KK.break_even_points(p, d), KK.break_even_points(p, d, True)) for p in ps]


def analyse_cost(d: int, p: int, seed: int) -> CostAnalysis:
    ps = sorted({2, 10, 100, 1000, p})
    rows = cost_rows(d, ps)
    inst = S.rings(60, seed)
    order = S.order_of(inst.n, seed)
    Kt = KK.gram_tilde(("poly", d), inst.X)
    rd = P.run_dual(Kt, inst.y, order, 3000)
    Phi = KK.with_bias_column(KK.poly_features(inst.X, d))
    wp, up, ep, cp = P.run_primal(Phi, inst.y, order, 3000)
    wd = Phi.T @ (rd.alpha * inst.y)
    rel = float(np.linalg.norm(wd - wp) / max(np.linalg.norm(wp), 1e-300))
    return CostAnalysis(d, p, rows, rd.updates, up, bool(rd.updates == up and rd.epochs == ep and rd.converged == cp), rel, int((rd.alpha > 0).sum()),
                        KK.poly_dimension(2, d), KK.kernel_multiplications(2, d))


# ------------------------------------------------------------------ Experiment 3: Datensätze
DATEN_NOISE = {C.DATEN_RINGE: 0.0, C.DATEN_LIEFER: 0.0, C.DATEN_MONDE: 0.0, C.DATEN_MONDE_R: C.MOONS_NOISE}


def make_ex3_instance(daten: str, n: int, seed: int) -> S.Instance:
    if daten in (C.DATEN_MONDE, C.DATEN_MONDE_R):
        return S.moons(n, seed, DATEN_NOISE[daten])
    return S.make_instance(daten, n, seed)


@dataclass
class DatasetAnalysis:
    inst: S.Instance
    spec: tuple
    run: P.RunResult
    K: np.ndarray
    Kt: np.ndarray
    grid: tuple
    train_error: float
    n_support: int
    qp_tilde: M.QPResult | None
    qp_free: M.QPResult | None
    bound: float | None
    ratio_free: float | None
    ratio_tilde: float | None
    qp_note: str
    noise: float = 0.0


def decision_grid(inst: S.Instance, spec: tuple, alpha: np.ndarray, m: int = 60):
    X = inst.X
    pad = 0.15 * max(float(np.ptp(X[:, 0])), float(np.ptp(X[:, 1])), 1.0)
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, m)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, m)
    G = np.array([(a, b) for b in ys for a in xs])
    F = (KK.gram_tilde(spec, G, X) @ (alpha * inst.y)).reshape(m, m)
    return xs, ys, F


def analyse_dataset(daten: str, kern_label: str, n: int, mischen: bool, seed: int) -> DatasetAnalysis:
    inst = make_ex3_instance(daten, n, seed)
    spec = C.EX3_KERNELS[kern_label]
    K = KK.kernel_matrix(spec, inst.X)
    Kt = K + 1.0
    run = P.run_dual(Kt, inst.y, S.order_of(n, seed), C.DEFAULT_EPOCHS, C.MAX_UPDATES, shuffle_seed=(seed ^ C.SALT_SHUFFLE) if mischen else None)
    grid = decision_grid(inst, spec, run.alpha)
    qp_t = qp_f = bound = rf = rt = None
    note = ""
    if not run.converged:
        note = "keine Schranke: der Lauf ist nicht konvergiert"
    elif n > C.QP_LIVE_MAX_N:
        note = f"Schranke und größte Marge werden live nur bis n = {C.QP_LIVE_MAX_N} gerechnet"
    else:
        qp_t = M.max_margin(K, inst.y, False)
        qp_f = M.max_margin(K, inst.y, True)
        if qp_t.certified:
            bound = M.novikoff_bound(Kt, qp_t)
        else:
            note = "Maximum-Margin-Trennung nicht zertifiziert (keine Zahl)"
        if qp_t.certified or qp_f.certified:
            rf, rt = M.found_margin_ratio(Kt, K, inst.y, run.alpha, qp_f, qp_t)
    return DatasetAnalysis(inst, spec, run, K, Kt, grid, P.error_rate(run.s, inst.y), int((run.alpha > 0).sum()), qp_t, qp_f, bound, rf, rt, note, DATEN_NOISE[daten])


# ------------------------------------------------------------------ Experiment 4: Bandbreite
@dataclass
class BandwidthAnalysis:
    sigma: float
    run: P.RunResult
    gram: np.ndarray          # nach Klasse und Winkel geordnet
    offdiag_mean: float
    n: int


def analyse_bandwidth(sigma: float, seed: int, n: int = 100) -> BandwidthAnalysis:
    inst = S.rings(n, seed)
    Kt = KK.gram_tilde(("gauss", sigma), inst.X)
    run = P.run_dual(Kt, inst.y, S.order_of(n, seed), 20000, C.MAX_UPDATES)
    ang = np.arctan2(inst.X[:, 1], inst.X[:, 0])
    perm = np.lexsort((ang, inst.y))
    K = (Kt - 1.0)[np.ix_(perm, perm)]
    off = K[~np.eye(n, dtype=bool)]
    return BandwidthAnalysis(sigma, run, K, float(off.mean()), n)


def sweep_curve() -> dict:
    """Eingefrorene Bandbreitenkurve: Median (Mittel der beiden Satz-Mediane), Minimum und Maximum über beide Sätze."""
    sw = load_sweep()["bandbreite"]
    sigmas = list(C.SIGMAS)
    med, lo, hi = [], [], []
    for s in sigmas:
        row = sw[KK.fmt_num(s)]
        meds = [row[k]["updates"]["median"] for k in ("A", "B")]
        med.append(float(np.mean(meds)))
        lo.append(min(row[k]["updates"]["min"] for k in ("A", "B")))
        hi.append(max(row[k]["updates"]["max"] for k in ("A", "B")))
    return {"sigmas": sigmas, "median": med, "min": lo, "max": hi}


# ------------------------------------------------------------------ Experiment 5: die Falle
@dataclass
class TrapAnalysis:
    daten: str
    etiketten: str
    spec: tuple
    epochs_limit: int
    run: P.RunResult
    run_true: P.RunResult
    train_error: float
    test_error: float
    min_eigenvalue: float
    indefinite: bool
    n: int


def analyse_trap(daten_label: str, etiketten: str, kern_label: str, epochs: int, seed: int) -> TrapAnalysis:
    if daten_label == C.EX5_DATEN[0]:
        n, daten = 100, C.DATEN_RINGE
        base = S.rings(n, seed)
    else:
        n, daten = 200, C.DATEN_LIEFER
        base = S.delivery_area(n, seed)
    if etiketten == "10 % vertauscht":
        inst = S.flip_labels(base, C.D2_NOISE, seed)
    elif etiketten == "zufällig":
        inst = S.random_labels(base, seed)
    else:
        inst = base
    spec = C.EX5_KERNELS[kern_label]
    K = KK.kernel_matrix(spec, inst.X)
    Kt = K + 1.0
    order = S.order_of(n, seed)
    run = P.run_dual(Kt, inst.y, order, epochs, C.MAX_UPDATES)
    run_true = run if etiketten == "wahr" else P.run_dual(Kt, base.y, order, epochs, C.MAX_UPDATES)
    test = S.fresh_test_points(daten, C.N_TEST, seed)
    f = P.decision_values(KK.gram_tilde(spec, test.X, inst.X), run.alpha, inst.y)
    eig = KK.smallest_eigenvalue(K)
    lam_max = float(np.linalg.eigvalsh((K + K.T) / 2)[-1])
    return TrapAnalysis(daten_label, etiketten, spec, epochs, run, run_true, P.error_rate(run.s, inst.y), P.error_rate(f, test.y_true), eig,
                        bool(eig < -100.0 * np.finfo(float).eps * lam_max), n)


# ------------------------------------------------------------------ Verdikte (reine Funktionen: (Art, Text))
def _is_half(v: float) -> bool:
    """Halbzahl (x,5): Median einer geraden Zahl ganzzahliger Werte."""
    v = float(v)
    return v != int(v) and v * 2 == int(v * 2)


def _band(rows: dict, path: tuple, nd: int = 0) -> str:
    """„a bis b“ der beiden Satz-Werte (Satz A und B), bei gleichen Werten nur „a“. Mit nd = 0 werden Halbzahlen (Median zweier ganzer Zahlen) nicht verschluckt."""
    vals = []
    for k in ("A", "B"):
        v = rows[k]
        for p in path:
            v = v[p]
        vals.append(v)
    nd_eff = 1 if nd == 0 and any(_is_half(v) for v in vals) else nd
    lo, hi = round(min(vals), nd_eff), round(max(vals), nd_eff)

    def one(v):
        return de(v, 1) if nd == 0 and _is_half(v) else de(v, nd_eff if nd else 0)

    return one(lo) if lo == hi else f"{one(lo)} bis {one(hi)}"



def _order_text(order: list[int]) -> str:
    return ", ".join(S.XOR_NAMES[i] for i in order)


def verdict_xor(a: XorAnalysis) -> tuple[str, str]:
    """Experiment 1: linear Zyklus (kein Konvergieren), sonst Korrekturen, α, Marge und Schranke dieses Kerns."""
    if a.run.stop == P.STOP_CYCLE:
        w0 = " (w = 0 wie am Anfang)" if not np.any(a.run.s) else ""
        return "warning", (
            f"**{a.kern}, Reihenfolge {_order_text(a.order)}:** nach {de(a.run.cycle_epochs)} Epoche{'' if a.run.cycle_epochs == 1 else 'n'} mit {de(a.run.cycle_updates)} Korrekturen "
            f"kehrt der Zustand zurück{w0}: ein Zyklus, das Perceptron konvergiert nie (exakt, Rechnung mit ganzen Zahlen). Von Hand: die Randbedingungen von a und b summieren sich "
            f"zu 2b ≥ 2, die von c und d zu −2b ≥ 2; keine Gerade erfüllt beides."
        )
    if not a.run.converged:
        return "warning", f"**{a.kern}:** keine Konvergenz nach {de(a.run.epochs)} Epochen (Grenze erreicht)."
    alpha = ", ".join(de(v) for v in a.run.alpha)
    ups = sorted({u for u in a.all_orders_updates if u is not None})
    teil = (f"in {a.all_orders_converged} von 24 Reihenfolgen konvergiert es, " +
            (f"jedes Mal mit {de(ups[0])} Korrekturen" if len(ups) == 1 else f"mit {de(ups[0])} bis {de(ups[-1])} Korrekturen") +
            (" und demselben α" if a.all_orders_same_alpha else ""))
    text = f"**{a.kern}, Reihenfolge {_order_text(a.order)}:** {de(a.run.updates)} Korrekturen in {de(a.run.epochs)} Epochen (die letzte ohne Korrektur), α = ({alpha}); {teil}."
    if a.formula is not None:
        text += f" Die Trennfunktion ist f(x) = {a.formula}."
    if a.qp is not None and a.bound is not None:
        text += f" Maximum-Margin-Trennung in (φ, 1): Marge 2/‖w*‖ = {de(a.qp.margin, 3)}, Schranke R̃²‖w*‖² = {de(a.bound, 2)} ≥ {de(a.run.updates)} Korrekturen."
    if a.spec[0] == "gauss":
        u = float(np.exp(-2.0 / a.spec[1] ** 2))
        text += f" Geschlossene Form der Marge auf XOR: 1 − exp(−2/σ²) = {de(1.0 - u, 6)}."
    return "success", text


def verdict_cost(a: CostAnalysis) -> tuple[str, str]:
    """Experiment 2: Aufwand je Skalarprodukt (nur Auswertung an einem neuen Punkt) und der Gleichstand von dualer und primaler Form."""
    kt = load_sweep()["kerntrick"]
    runs = sum(r["instances"] for r in kt.values())
    equal = sum(r["equal_runs"] for r in kt.values())
    maxrel = max(r["max_rel_w_dual_vs_primal"] for r in kt.values())
    D, k = KK.poly_dimension(a.p, a.d), KK.kernel_multiplications(a.p, a.d)
    n0, n1 = KK.break_even_points(a.p, a.d), KK.break_even_points(a.p, a.d, True)
    cheaper = a.n_support * a.k2 < a.D2
    text = (
        f"**Polynom {a.d} bei p = {de(a.p)}:** {de(D)} Merkmale gegen {de(k)} Multiplikationen je Skalarprodukt, bei der Auswertung an einem neuen Punkt mit fertig vorliegendem φ(x) "
        f"(nicht im Training). Die duale Form ist billiger, solange höchstens {de(n0)} verschiedene Punkte korrigiert wurden; wird der Bau von φ(x) mit mindestens einer Multiplikation je "
        f"Merkmal mitgezählt (Annahme, untere Abschätzung), höchstens {de(n1)}. Dual und primal in diesem Lauf (Ringe, n = 60, p = 2): {de(a.updates_dual)} und {de(a.updates_primal)} Korrekturen, "
        f"Normalen relativ {sci(a.rel_w)} auseinander. In 2D (D = {de(a.D2)}) hat dieser Lauf {de(a.n_support)} verschiedene korrigierte Punkte zu je {de(a.k2)} Multiplikationen, "
        f"also {de(a.n_support * a.k2)} gegen {de(a.D2)}: der Kern ist hier {'billiger' if cheaper else 'gleich teuer' if a.n_support * a.k2 == a.D2 else 'teurer'}. Messung (Polynom 2 und 3, Ringe und Liefergebiet, 50 Instanzen je Zeile): "
        f"gleiche Korrekturen und Epochen in {de(equal)} von {de(runs)} Läufen, w relativ höchstens {sci(maxrel)}. Der Gewinn des Kerntricks ist Rechenaufwand bei großem D und der Zugang zu "
        f"unendlichdimensionalen Merkmalsräumen, nicht Geschwindigkeit in 2D."
    )
    return "info", text


def _frozen_row_for(a: DatasetAnalysis) -> tuple[str, dict] | None:
    """Eingefrorene Messzeile (n = 100 bzw. 20 Zufalls-Seeds je Satz) zur Kombination aus Datensatz und Kern, sonst None."""
    sw = load_sweep()
    name = KK.kernel_name(a.spec)
    if a.inst.name == "Ringe mit Lücke" and name in sw["ringe"]:
        return "Ringe mit Lücke, n = 100, 30 Zufalls-Seeds je Satz, Grenze 3 000 Epochen", sw["ringe"][name]
    if a.inst.name == "Liefergebiet (Kreis)" and f"n100 {name}" in sw["liefergebiet"]:
        return "Liefergebiet (Kreis), n = 100, 30 Zufalls-Seeds je Satz, Grenze 3 000 Epochen", sw["liefergebiet"][f"n100 {name}"]
    if a.inst.name == "Monde":
        key = f"Rauschen {KK.fmt_num(a.noise)} {name}"
        if key in sw["monde"]:
            return f"Monde, Rauschen {KK.fmt_num(a.noise)}, n = 100, 20 Zufalls-Seeds je Satz, Grenze 2 000 Epochen", sw["monde"][key]
    return None


def verdict_dataset(a: DatasetAnalysis) -> tuple[str, str]:
    """Experiment 3: dieser Lauf (Korrekturen, Schranke, Marge der gefundenen Trennung) und die eingefrorene Messzeile, falls es sie gibt."""
    r = a.run
    name = KK.kernel_name(a.spec)
    kind = "success"
    if r.converged:
        text = f"**{name}, {a.inst.name}, n = {de(a.inst.n)}:** konvergiert nach {de(r.epochs)} Epochen mit {de(r.updates)} Korrekturen ({de(a.n_support)} verschiedene Punkte mit α > 0)."
        if a.bound is not None:
            if r.updates <= a.bound + 1e-9:
                text += f" Schranke R̃²‖w*‖² = {de(a.bound, 1)} ≥ {de(r.updates)} Korrekturen (Rechnung dieser Seite, diese Instanz)."
            else:
                kind = "warning"
                text += f" Die Schranke R̃²‖w*‖² = {de(a.bound, 1)} wird überschritten: das widerspräche dem Satz, das Zertifikat der Maximum-Margin-Trennung ist zu prüfen."
        elif a.qp_note:
            text += f" {a.qp_note[0].upper()}{a.qp_note[1:]}."
        if a.ratio_free is not None:
            text += (f" Die gefundene Trennung hat {pct(a.ratio_free)} der größten Marge (Maß mit freiem b; diese Instanz und diese Reihenfolge): das Perceptron nimmt die erste "
                     f"Trennung, die es findet.")
    else:
        kind = "warning"
        text = (f"**{name}, {a.inst.name}, n = {de(a.inst.n)}:** Grenze erreicht ({r.stop}) nach {de(r.epochs)} Epochen und {de(r.updates)} Korrekturen, Trainingsfehler {pct(a.train_error)}. "
                f"„Grenze erreicht“ heißt nicht „nicht trennbar“.")
    row = _frozen_row_for(a)
    if row is not None:
        desc, rows = row
        text += (f" Eingefrorene Messung ({desc}): konvergiert in {de(rows['A']['converged'])} und {de(rows['B']['converged'])} von {de(rows['A']['runs'])} Läufen, "
                 f"Korrekturen im Median {_band(rows, ('updates', 'median'))}.")
    else:
        text += " Für diese Kombination liegt keine eingefrorene Messzeile vor (gemessen sind die Kerne und Datensätze der Tabelle unten)."
    return kind, text


def bandwidth_totals() -> tuple[int, int, int, int]:
    """(Läufe, davon konvergiert, zertifizierte Läufe, davon mit überschrittener Schranke) über alle σ und beide Sätze der eingefrorenen Kurve."""
    sw = load_sweep()["bandbreite"]
    runs = sum(sw[s][k]["runs"] for s in sw for k in "AB")
    conv = sum(sw[s][k]["converged"] for s in sw for k in "AB")
    cert = sum(sw[s][k]["certified"] for s in sw for k in "AB")
    viol = sum(sw[s][k]["bound_violations"] for s in sw for k in "AB")
    return runs, conv, cert, viol


def verdict_bandwidth(a: BandwidthAnalysis) -> tuple[str, str]:
    """Experiment 4: dieser Lauf und die U-Form der eingefrorenen Kurve."""
    sw = load_sweep()["bandbreite"]
    flat = [sw[KK.fmt_num(s)][k]["updates"]["median"] for s in (0.25, 0.5, 1.0) for k in "AB"]
    flat_txt = de(min(flat)) if min(flat) == max(flat) else f"{de(min(flat))} bis {de(max(flat))}"
    runs, conv, cert, viol = bandwidth_totals()
    lo, hi = sw[KK.fmt_num(0.02)], sw[KK.fmt_num(8.0)]
    if a.run.converged:
        own = f"{de(a.run.updates)} Korrekturen in {de(a.run.epochs)} Epochen"
    else:
        own = f"Grenze erreicht ({a.run.stop}) nach {de(a.run.epochs)} Epochen und {de(a.run.updates)} Korrekturen"
    text = (f"**σ = {KK.fmt_num(a.sigma)} (γ = {de(KK.gamma_of(a.sigma), 3)}):** {own} (Ringe, n = {de(a.n)}, diese Instanz); der mittlere Außerdiagonaleintrag der Gram-Matrix ist {de(a.offdiag_mean, 3)}. "
            f"Eingefrorene Messung (Ringe, n = 100, Gauß-Kern, Mediane der Korrekturen über je 30 Zufalls-Seeds, zwei Sätze): σ = 0,02: {_band(lo, ('updates', 'median'))}; "
            f"σ = 0,25 bis 1: {flat_txt}; σ = 8: {_band(hi, ('updates', 'median'))}. Außerhalb von σ = 0,25 bis 1 wächst die Zahl auf beiden Seiten (σ = 0,1: {_band(sw[KK.fmt_num(0.1)], ('updates', 'median'))}; σ = 2: {_band(sw[KK.fmt_num(2.0)], ('updates', 'median'))}). "
            f"Die Meldung „konvergiert“ ist in {de(conv)} von {de(runs)} Läufen dieselbe, die Schranke R̃²‖w*‖² wird in {de(cert - viol)} von {de(cert)} zertifizierten Läufen nicht überschritten.")
    return "info", text


def trap_summary_for(a: TrapAnalysis) -> str:
    """Eingefrorene Zeile D1 (zufällige Labels auf den Ringen) zum gewählten Gauß-Kern, sonst leer."""
    if a.etiketten != "zufällig" or a.daten != C.EX5_DATEN[0] or a.spec[0] != "gauss":
        return ""
    row = load_sweep()["zufall"].get(KK.fmt_num(a.spec[1]))
    if row is None:
        return ""
    conv = sum(row[k]["converged"] for k in "AB")
    w3 = sum(row[k]["within_3000"] for k in "AB")
    return (f" Eingefrorene Messung (Ringe, n = 100, zufällige Labels, 60 Zufalls-Seeds in zwei Sätzen, σ = {KK.fmt_num(a.spec[1])}): bei der Messgrenze von 50 000 Epochen "
            f"konvergiert das Perceptron in {de(conv)} von 60, mit 3 000 Epochen in {de(w3)} von 60; Korrekturen im Median {_band(row, ('updates', 'median'))} gegen "
            f"{_band(row, ('true_updates_median',))} mit den wahren Labels.")


def verdict_trap(a: TrapAnalysis) -> tuple[str, str]:
    """Experiment 5: was die Meldung „konvergiert“ sagt und was nicht."""
    r = a.run
    kern = KK.kernel_name(a.spec)
    if r.converged:
        meldung = f"Meldung: „konvergiert nach {de(r.epochs)} Epochen“."
        kind = "success"
    else:
        meldung = (f"Meldung: „Grenze erreicht“ ({r.stop}) nach {de(r.epochs)} Epochen und {de(r.updates)} Korrekturen; das heißt nicht „nicht trennbar“, "
                   f"die Grenze entscheidet, was hier steht.")
        kind = "warning"
    vergleich = ""
    if a.etiketten != "wahr":
        vergleich = f" (mit den wahren Labels auf denselben Punkten: {de(a.run_true.updates)}{'' if a.run_true.converged else ', Grenze erreicht'})"
    text = (f"**{kern}, Labels {a.etiketten}, {a.daten}:** {meldung} Korrekturen: {de(r.updates)}{vergleich}. "
            f"Trainingsfehler {pct(a.train_error, 1)}, Testfehler gegen die wahre Klasse {pct(a.test_error, 1)} ({de(C.N_TEST)} frische Punkte).")
    if a.etiketten == "zufällig" and r.converged and a.run_true.converged and a.run_true.updates:
        kind = "warning"
        text += (f" Dieselbe Meldung „konvergiert“ wie bei wahren Labels, aber das {de(r.updates / a.run_true.updates, 1)}-Fache an Korrekturen: aus der Meldung allein ist der Unterschied nicht zu sehen.")
    if a.indefinite:
        kind = "warning"
        text += f" Die Gram-Matrix ist indefinit (kleinster Eigenwert {de(a.min_eigenvalue, 3)}): das Sigmoid ist hier kein Skalarprodukt, und der Lauf meldet dazu nichts."
    else:
        text += f" Kleinster Eigenwert der Gram-Matrix: {sci(a.min_eigenvalue)}."
    text += trap_summary_for(a)
    return kind, text


def verdict_d2() -> tuple[str, str]:
    """Ehrlicher Negativbefund (eingefroren): Konvergenzmeldung und Testfehler bei 10 % vertauschten Labels (Liefergebiet, n = 200, 40 Zufalls-Seeds je Zeile)."""
    d2 = load_sweep()["d2"]
    gauss = [(k, v["alle"]) for k, v in d2.items() if k.startswith("Gauß")]
    means = [r["test_error_mean"] for _, r in gauss]
    ses = [r["test_error_se"] for _, r in gauss]
    diffs = [(abs(gauss[i + 1][1]["test_error_mean"] - gauss[i][1]["test_error_mean"]),
              abs(gauss[i + 1][1]["test_error_mean"] - gauss[i][1]["test_error_mean"]) / float(np.hypot(ses[i], ses[i + 1])), i) for i in range(len(gauss) - 1)]
    big = max(diffs)
    conv_rows = [r["test_error_mean"] for _, r in gauss if r["converged"] == 40]
    never_rows = [r["test_error_mean"] for _, r in gauss if r["converged"] == 0]
    mix_rows = [(k, r) for k, r in gauss if 0 < r["converged"] < 40]
    poly = d2["Polynom 2"]["alle"]
    teil = "".join(f", in einem Teil ({de(r['converged'])} von 40)" for _, r in mix_rows)
    ordnung = "innerhalb der Streuung" if big[1] < 2.0 else "eine Tendenz, kein Befund"
    text = (
        f"**Ehrlicher Negativbefund:** die Meldung „konvergiert“ trennt gute und schlechte Läufe nicht. Der Testfehler gegen die wahre Klasse liegt in den {len(gauss)} Gauß-Zeilen bei "
        f"{de(min(means), 3)} bis {de(max(means), 3)} (Standardfehler {de(min(ses), 3)} bis {de(max(ses), 3)}), ob das Perceptron in allen 40{teil} oder in keinem der 40 Läufe konvergiert. Differenzen benachbarter Gauß-Zeilen: höchstens {de(big[0], 3)} ({de(big[1], 1)} Standardfehler der Differenz, {ordnung}). "
        f"Die Zeilen ohne Konvergenz ({'; '.join(de(v, 3) for v in never_rows)}) sind {'nicht besser' if min(never_rows) >= max(conv_rows) else 'zum Teil besser'} als die mit Konvergenz "
        f"({'; '.join(de(v, 3) for v in conv_rows)}). Polynom 2 (konvergiert nie): {de(poly['test_error_mean'], 3)} ({de(poly['test_error_se'], 3)}). Ein Schaden durch das Auswendiglernen ist in diesen "
        f"Zeilen nicht erkennbar; ein Vergleich mit abgestimmtem C fehlt (nicht gemessen), deshalb steht hier kein Satz „Auswendiglernen schadet“."
    )
    return "info", text


def assumptions_table_markdown() -> str:
    """Tabelle „Wo die Annahmen enden“ mit den eingefrorenen Zahlen."""
    sw = load_sweep()
    z = sw["zufall"]["0,25"]
    not3000 = 60 - sum(z[k]["within_3000"] for k in "AB")
    pd = sum(z[k]["gram_positive_definite"] for k in "AB")
    sig = sw["sigmoid"]
    sig_conv = "; ".join(f"{name.replace('Sigmoid ', '')}: {de(v['converged'])} von 30" for name, v in sig.items())
    sig_ind = min(v["indefinite"] for v in sig.values())
    ratios = {s: _band(sw["zufall"][s], ("ratio_median_updates",), nd) for s, nd in (("0,05", 1), ("0,1", 1), ("0,25", 0))}
    cos = _band(sw["marge"]["Ringe Gauß σ = 1"], ("cos_min_per_instance", "median"), 2)
    ub = sw["bandbreite"]
    rows = [
        ("Die Punkte sind im Merkmalsraum trennbar",
         f"Sonst konvergiert das Perceptron nie: bei XOR mit linearem Kern kehrt der Zustand nach einer Epoche mit 4 Korrekturen zu w = 0 zurück (exakt). Aber „Grenze erreicht“ heißt nicht „nicht trennbar“: "
         f"bei zufälligen Labels und σ = 0,25 meldet die Seite mit 3 000 Epochen in {de(not3000)} von 60 Zufalls-Seeds „Grenze erreicht“, obwohl alle {de(pd)} trennbar sind (Gram-Matrix positiv definit)."),
        ("Der Kern ist ein Skalarprodukt (Gram-Matrix positiv semidefinit)",
         f"Sigmoid auf den Ringen (n = 100 > d = 2): die Gram-Matrix ist in {de(sig_ind)} von 30 Zufalls-Seeds indefinit, in allen {len(sig)} Parametersätzen; das Perceptron läuft ohne Meldung, konvergiert in "
         f"{sig_conv}. Konvergiert es, ist die Trennung der Trainingspunkte gültig, aber die Schranke hat keinen Merkmalsraum mehr. Eine Prüfung des Kerns gehört nicht zu dieser Seite."),
        ("Feste Reihenfolge je Zufalls-Seed",
         f"Die Zahlen gelten für diese Regel. Die gefundene Trennung hängt an der Reihenfolge: der kleinste Kosinus der Normalen zweier Reihenfolgen derselben Instanz liegt im Median bei {cos} (Ringe, Gauß σ = 1, "
         f"30 Instanzen je Satz, 8 Reihenfolgen). Mischen in jeder Epoche ist als Schalter angeboten und nicht gemessen."),
        ("Die Bandbreite passt zu den Daten",
         f"Nur an den Ringen gemessen: die Zahl der Korrekturen ist U-förmig in σ ({_band(ub[KK.fmt_num(0.02)], ('updates', 'median'))} bei σ = 0,02, {_band(ub[KK.fmt_num(1.0)], ('updates', 'median'))} bei σ = 1, "
         f"{_band(ub[KK.fmt_num(8.0)], ('updates', 'median'))} bei σ = 8, Mediane). Liefergebiet und Monde: nicht gemessen."),
        ("Die Labels stimmen",
         f"Mit zufälligen Labels auf denselben Ringen braucht das Perceptron mehr Korrekturen als mit den wahren Labels; Verhältnis der Mediane: {ratios['0,05']} (σ = 0,05), {ratios['0,1']} (σ = 0,1), {ratios['0,25']} (σ = 0,25), und meldet dasselbe „konvergiert“."),
        ("Nutzen im Betrieb",
         "Nicht belegt. Die Punkte sind ausgedacht; in den gelesenen Quellen gab es keine Messung, ob Kernmethoden in Logistik, Lager oder Fertigung bessere Entscheidungen liefern."),
    ]
    head = "| Annahme | Was passiert, wenn sie verletzt ist (gemessen, eingefroren) |\n|---|---|\n"
    return head + "\n".join(f"| **{a}** | {b} |" for a, b in rows)


def limits_markdown() -> str:
    """Annahmen, Grenzen, bekannte Probleme (dieselben drei Teile wie in der README)."""
    return (
        "**Annahmen.** Alle Punkte sind ausgedacht (Modelle, keine Messdaten). Der Bias steht als Zusatzkoordinate: K̃ = K + 1. Start bei w = 0, deshalb ist die Lernrate ohne Wirkung. "
        "Feste Reihenfolge der Punkte je Zufalls-Seed. „Konvergiert“ heißt: eine Epoche ohne Korrektur; der Fall y f = 0 zählt als Fehler. „Marge“ heißt 2/‖w‖ (Breite zwischen den Klassen; "
        "viele Lehrtexte nennen 1/‖w‖ Marge, das ist hier der Abstand zur Ebene). Die Schranke R̃²‖w*‖² ist eine Rechnung dieser Seite, keine Quellenaussage.\n\n"
        "**Grenzen.** Nur zweidimensionale Beispiele; größere n und höhere Dimension sind nicht gemessen. Es gibt keinen Regler für die Hyperparameter C, λ oder ν und keinen Vergleich mit einem abgestimmten Verfahren, "
        "deshalb keine Aussage „Auswendiglernen schadet“. Die Bandbreite ist nur an den Ringen gemessen. Der Kern wird nicht auf Zulässigkeit geprüft (nur der kleinste Eigenwert wird gezeigt). "
        "Mischen der Reihenfolge je Epoche ist als Schalter angeboten, aber nicht gemessen. Ein Nutzen von Kernmethoden in Logistik, Lager oder Fertigung ist nicht belegt.\n\n"
        "**Bekannte Probleme und Unsicherheiten.** Live-Modell und Studie sind verschieden: die eingefrorenen Tabellen fassen 20 bis 60 Zufalls-Seeds zusammen, die Seite rechnet eine Instanz; "
        "ein einzelner Zufalls-Seed ist kein Befund. Die Zahlen stammen aus festen Zufalls-Seeds; Rundung der Exponential-, Sinus- und Kosinusfunktion der Rechenumgebung kann eine Entscheidung bei "
        f"f nahe 0 verschieben, die Tests prüfen darum Bänder. Schranke und größte Marge werden live nur bei Konvergenz und bis n = {C.QP_LIVE_MAX_N} gerechnet und tragen ein Zertifikat "
        "(Dualitätslücke und kleinste Randbedingung); ohne Zertifikat steht keine Zahl. Die Monde haben beschränktes (gleichverteiltes) Rauschen, nicht Gauß-Rauschen."
    )


MATH_LESEHILFE = (
    "**Lesehilfe.** *Gegeben:* Punkte $x_i \\in \\mathbb{R}^2$ mit Labels $y_i \\in \\{-1, +1\\}$, $i = 1, \\dots, n$, und eine Kernfunktion $K$. "
    "*Gesucht:* Zähler $\\alpha_i \\in \\{0, 1, 2, \\dots\\}$, mit denen $f$ jeden Punkt richtig einordnet ($y_i f(x_i) > 0$). *Maß:* Zahl der Korrekturen (Summe der $\\alpha_i$) und Zahl der Epochen."
)
MATH_SYMBOLE = """
| Symbol | Bedeutung |
|---|---|
| $x, y$ | Punkt in $\\mathbb{R}^p$ (hier $p = 2$) und Label $y_i \\in \\{-1, +1\\}$; $y$ in $K(x, y)$ meint einen zweiten Punkt |
| $\\varphi$ | Merkmalsabbildung; wird nie ausgerechnet, nur über $K(x, x') = \\langle \\varphi(x), \\varphi(x') \\rangle$ benutzt |
| $K$, $\\tilde K$ | Kernfunktion; $\\tilde K = K + 1$ mit dem Bias als Zusatzkoordinate |
| $\\alpha_i$ | Zahl der Korrekturen des Punktes $i$ |
| $w$ | Normale im Merkmalsraum, $w = \\sum_j \\alpha_j y_j \\tilde\\varphi(x_j)$ mit $\\tilde\\varphi = (\\varphi, 1)$ |
| $f$ | Entscheidungsfunktion $f(x) = \\sum_j \\alpha_j y_j \\tilde K(x_j, x)$ |
| $\\tilde R$ | $\\tilde R^2 = \\max_i \\tilde K(x_i, x_i)$ |
| $\\lVert w^* \\rVert$ | Norm der Maximum-Margin-Trennung in $(\\varphi, 1)$; Marge $= 2 / \\lVert w^* \\rVert$ (Abstand zur Ebene $1 / \\lVert w^* \\rVert$) |
| $\\sigma$ | Bandbreite des Gauß-Kerns, Längenmaß in den Einheiten der Daten; $\\gamma = 1 / (2\\sigma^2)$ |
| $D$, $p$, $d$ | Zahl der Merkmale, Eingabedimension, Grad des Polynomkerns; $D = \\binom{p + d}{d}$ |
"""
MATH_RULE_INTRO = (
    "**Modell (vollständig).** Start $\\alpha = 0$ (also $w = 0$). Die Punkte werden je Epoche in einer festen Reihenfolge durchlaufen (je Zufalls-Seed eine feste Zufallsfolge; "
    "als Schalter in jeder Epoche neu gemischt). Eine Epoche ohne Korrektur beendet den Lauf (*konvergiert*); sonst endet er an der Epochen- oder Korrekturgrenze. Der Fall $y_i f(x_i) = 0$ zählt als Fehler."
)
MATH_EQUATIONS = [
    r"\tilde K(x, x') = K(x, x') + 1",
    r"f(x) = \sum_{j=1}^{n} \alpha_j \, y_j \, \tilde K(x_j, x)",
    r"y_i f(x_i) \le 0 \;\Longrightarrow\; \alpha_i \leftarrow \alpha_i + 1",
    r"K(x, x') = x \cdot x', \qquad K(x, x') = (x \cdot x' + 1)^d, \qquad K(x, x') = \exp\!\Big(-\frac{\lVert x - x' \rVert^2}{2\sigma^2}\Big)",
    r"(x \cdot x' + 1)^d = \sum_{|k| = d} \binom{d}{k_0, \dots, k_p} x_1^{k_1} \cdots x_p^{k_p} \, x_1'^{k_1} \cdots x_p'^{k_p} \qquad (k_0 + \dots + k_p = d)",
]
MATH_AFTER = (
    "Der **Kerntrick** ist die Ersetzung von $\\langle \\tilde\\varphi(x), \\tilde\\varphi(x') \\rangle$ durch $\\tilde K(x, x')$; die Regel selbst bleibt gleich (duale Form: $\\alpha_i$ zählt, wie oft Punkt $i$ korrigiert wurde). "
    "Der Polynomkern hat $D = \\binom{p+d}{d}$ Merkmale (Multinomialsatz, die Summe oben); eine Kernauswertung kostet $p + d - 1$ Multiplikationen. "
    "**Maximum-Margin-Trennung und Schranke** (Rechnung dieser Seite; die Schranke ist hier eine Rechnung, keine Quellenaussage): $w^*$ löse $\\min \\tfrac12 \\lVert w \\rVert^2$ unter $y_i \\langle w, \\tilde\\varphi(x_i) \\rangle \\ge 1$; "
    "das duale Problem mit $Q_{ij} = y_i y_j \\tilde K(x_i, x_j)$ ist"
)
MATH_BOUND_EQUATIONS = [
    r"\min_{\alpha \ge 0} \; \tfrac12\, \alpha^{\top} Q\, \alpha - \mathbf{1}^{\top} \alpha, \qquad \lVert w^* \rVert^2 = \alpha^{*\top} Q\, \alpha^{*}",
    r"\lVert w_k \rVert^2 \le k\,\tilde R^2, \qquad \langle w_k, w^* \rangle \ge k \quad\Longrightarrow\quad k^2 \le \lVert w_k \rVert^2 \lVert w^* \rVert^2 \le k\,\tilde R^2 \lVert w^* \rVert^2 \quad\Longrightarrow\quad k \le \tilde R^2 \lVert w^* \rVert^2",
    r"\text{XOR, Gauß-Kern:}\quad \text{Marge} = \frac{2}{\lVert w^* \rVert} = 1 - e^{-2/\sigma^2}, \qquad \text{XOR, } (x \cdot x' + 1)^2:\quad \alpha = (1,1,1,1),\; f(x) = 8\,x_1 x_2",
]
MATH_LITERATUR = (
    "**Literatur** (nur Gleichungen und Befunde in eigenen Worten, die Stufe der Quelle steht dabei). Cortes, C., Vapnik, V. (1995): Support-Vector Networks, Machine Learning 20, 273–297 "
    "(Volltext gelesen: Kern als Skalarprodukt bei erfüllter Mercer-Bedingung, Polynomkern Gl. 37, Marge 2/|w| Gl. 13). Hofmann, T., Schölkopf, B., Smola, A. J. (2008): Kernel methods in machine learning, "
    "Annals of Statistics 36, 1171–1220, Abschnitt 2.3.3 (Volltext gelesen: Aizerman u. a. gaben einen Konvergenzbeweis, mit dem der Konvergenzsatz des Perceptrons anwendbar wurde). "
    "Aizerman, M. A., Braverman, E. M., Rozonoer, L. I. (1964), Automation and Remote Control 25 (nur der Abstract gelesen). Nicht gelesen und nicht zitiert als gelesen: Boser/Guyon/Vapnik 1992 (nur Abstract), "
    "Mercer 1909, Novikoff 1962, Rosenblatt 1958. Die Schranke $k \\le \\tilde R^2 \\lVert w^* \\rVert^2$ steht in keiner gelesenen Quelle und ist hier nachgerechnet."
)
