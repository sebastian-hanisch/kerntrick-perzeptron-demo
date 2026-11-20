"""Rechnet die Messreihen der Demo mit den App-Strömen (SplitMix64) neu und schreibt `ktp_sweep.json` (eingefroren); `--check` rechnet neu und vergleicht.

Aufruf (Demo-Ordner, Test-venv, OMP_NUM_THREADS=1):
    python tools/ktp_make_sweep.py            # schreibt ktp_sweep.json
    python tools/ktp_make_sweep.py --check    # rechnet neu, vergleicht mit ktp_sweep.json (Ganzzahlen und Zahlen auf 1e-6 relativ), Exit 1 bei Abweichung
    python tools/ktp_make_sweep.py --only zufall,marge   # nur einzelne Abschnitte (nur zusammen mit --check sinnvoll, schreibt dann nichts)

Zufalls-Seeds: Satz A = 1 … 30, Satz B = 1001 … 1030 (frisch); je Zahl steht die Zahl der Läufe daneben. Messgrenze der Zufallsetiketten: 50 000 Epochen,
Korrekturgrenze 300 000 (wie in der App). Keine Sekunden als Befund. sklearn wird hier nur für die Fremdreferenz des Kerntricks benutzt (nicht zur Laufzeit der App)."""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import ktp_constants as C  # noqa: E402
import ktp_kernels as KK  # noqa: E402
import ktp_margin as M  # noqa: E402
import ktp_perceptron as P  # noqa: E402
import ktp_scenario as S  # noqa: E402

JSON_PATH = ROOT / "ktp_sweep.json"
SET_A = tuple(range(1, 31))
SET_B = tuple(range(1001, 1031))
SETS = {"A": SET_A, "B": SET_B}
SETS20 = {"A": tuple(range(1, 21)), "B": tuple(range(1001, 1021))}
N_TEST_D2 = 4000


def stat(v) -> dict:
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"min": None, "median": None, "max": None, "n": 0}
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()), "n": int(len(v))}


def run(inst, spec, max_epochs, seed, max_updates=C.MAX_UPDATES, shuffle_seed=None):
    Kt = KK.gram_tilde(spec, inst.X)
    return P.run_dual(Kt, inst.y, S.order_of(inst.n, seed), max_epochs, max_updates, shuffle_seed=shuffle_seed), Kt


def converged_stats(results):
    conv = [r for r in results if r.converged]
    return {"converged": len(conv), "runs": len(results), "updates": stat([r.updates for r in results]),
            "epochs": stat([r.epochs for r in results]),
            "stops": {k: sum(1 for r in results if r.stop == k) for k in (P.STOP_CONVERGED, P.STOP_EPOCHS, P.STOP_UPDATES)}}


# ------------------------------------------------------------------ Abschnitte
def sec_ringe():
    """Ringe mit Lücke, n = 100, Grenze 3 000 Epochen: lineares Perceptron gegen Kerne."""
    out = {}
    for spec in (("linear",), ("poly", 2), ("poly", 3), ("gauss", 0.25), ("gauss", 0.5), ("gauss", 1.0), ("gauss", 2.0)):
        row = {}
        for name, seeds in SETS.items():
            res = [run(S.rings(100, s), spec, 3000, s)[0] for s in seeds]
            row[name] = converged_stats(res)
        out[KK.kernel_name(spec)] = row
    return out


def sec_bandbreite():
    """Gauß-Kern auf den Ringen (n = 100), Grenze 20 000 Epochen: Korrekturen, Epochen, Schranke R̃²‖w*‖² (zertifiziert) gegen σ."""
    out = {}
    for sigma in C.SIGMAS:
        row = {}
        for name, seeds in SETS.items():
            res, bounds, viol, cert = [], [], 0, 0
            for s in seeds:
                inst = S.rings(100, s)
                r, Kt = run(inst, ("gauss", sigma), 20000, s)
                res.append(r)
                if r.converged:
                    q = M.max_margin(Kt - 1.0, inst.y, False)
                    if q.certified:
                        cert += 1
                        b = M.novikoff_bound(Kt, q)
                        bounds.append(b)
                        viol += int(r.updates > b + 1e-9)
            d = converged_stats(res)
            d.update({"certified": cert, "bound": stat(bounds), "bound_violations": viol})
            row[name] = d
        out[KK.fmt_num(sigma)] = row
    return out


def sec_liefergebiet():
    """Liefergebiet (Kreis) ohne Lücke: Polynom 2 und 3 bei n = 100; Polynom 2 bei n = 200 mit Schranke."""
    out = {}
    for spec in (("poly", 2), ("poly", 3)):
        row = {}
        for name, seeds in SETS.items():
            row[name] = converged_stats([run(S.delivery_area(100, s), spec, 3000, s)[0] for s in seeds])
        out[f"n100 {KK.kernel_name(spec)}"] = row
    row = {}
    for name, seeds in SETS20.items():
        res, ratios, viol, cert = [], [], 0, 0
        for s in seeds:
            inst = S.delivery_area(200, s)
            r, Kt = run(inst, ("poly", 2), 3000, s)
            res.append(r)
            if r.converged:
                q = M.max_margin(Kt - 1.0, inst.y, False)
                if q.certified:
                    cert += 1
                    b = M.novikoff_bound(Kt, q)
                    ratios.append(b / r.updates)
                    viol += int(r.updates > b + 1e-9)
        d = converged_stats(res)
        d.update({"certified": cert, "bound_over_updates": stat(ratios), "bound_violations": viol})
        row[name] = d
    out["n200 Polynom 2"] = row
    return out


def sec_monde():
    out = {}
    for noise in (0.0, 0.1):
        for spec in (("linear",), ("poly", 2), ("poly", 3), ("gauss", 0.25), ("gauss", 0.5), ("gauss", 1.0)):
            row = {}
            for name, seeds in SETS20.items():
                row[name] = converged_stats([run(S.moons(100, s, noise), spec, 2000, s)[0] for s in seeds])
            out[f"Rauschen {KK.fmt_num(noise)} {KK.kernel_name(spec)}"] = row
    return out


def sec_zufall():
    """D1: Ringe (n = 100) mit zufällig vertauschten Etiketten, Gauß-Kern, Messgrenze 50 000 Epochen; 60 Zufalls-Seeds (Satz A + B)."""
    out = {}
    for sigma in (0.05, 0.1, 0.25, 0.5):
        row = {}
        for name, seeds in SETS.items():
            res, true_res, eig, solves = [], [], [], 0
            for s in seeds:
                inst = S.random_labels(S.rings(100, s), s)
                r, Kt = run(inst, ("gauss", sigma), 50000, s)
                res.append(r)
                true_res.append(run(S.rings(100, s), ("gauss", sigma), 50000, s)[0])
                K = Kt - 1.0
                eig.append(KK.smallest_eigenvalue(K))
                try:
                    c = np.linalg.solve(K, inst.y)
                    solves += int(np.max(np.abs(inst.y * (K @ c) - 1.0)) < 1e-3)
                except np.linalg.LinAlgError:
                    pass
            d = converged_stats(res)
            d["within_3000"] = sum(1 for r in res if r.converged and r.epochs <= 3000)
            d["true_updates_median"] = float(np.median([r.updates for r in true_res]))
            d["true_converged"] = sum(1 for r in true_res if r.converged)
            d["gram_positive_definite"] = sum(1 for e in eig if e > 0.0)
            d["gram_min_eigenvalue"] = float(min(eig))
            d["interpolation_solved"] = solves
            out_row_ratio = d["updates"]["median"] / d["true_updates_median"] if d["true_updates_median"] else None
            d["ratio_median_updates"] = out_row_ratio
            row[name] = d
        out[KK.fmt_num(sigma)] = row
    return out


def sec_d2():
    """D2: Liefergebiet n = 200 mit 10 % vertauschten Etiketten, Grenze 3 000 Epochen, Testfehler gegen die wahre Klasse (4 000 frische Punkte)."""
    out = {}
    for spec in (("gauss", 0.05), ("gauss", 0.1), ("gauss", 0.2), ("gauss", 0.4), ("gauss", 0.8), ("poly", 2)):
        row = {}
        allte = []
        for name, seeds in SETS20.items():
            res, tr, te = [], [], []
            for s in seeds:
                inst = S.flip_labels(S.delivery_area(200, s), C.D2_NOISE, s)
                r, Kt = run(inst, spec, 3000, s)
                res.append(r)
                tr.append(P.error_rate(r.s, inst.y))
                test = S.fresh_test_points(C.DATEN_LIEFER, N_TEST_D2, s)
                f = P.decision_values(KK.gram_tilde(spec, test.X, inst.X), r.alpha, inst.y)
                te.append(P.error_rate(f, test.y_true))
            d = converged_stats(res)
            d.update({"train_error_mean": float(np.mean(tr)), "test_error_mean": float(np.mean(te)), "test_error_min": float(min(te)), "test_error_max": float(max(te))})
            row[name] = d
            allte += te
        row["alle"] = {"test_error_mean": float(np.mean(allte)), "test_error_se": float(np.std(allte, ddof=1) / np.sqrt(len(allte))), "runs": len(allte),
                       "converged": sum(row[k]["converged"] for k in SETS20)}
        out[KK.kernel_name(spec)] = row
    return out


def sec_marge():
    """H: Marge der vom Perceptron gefundenen Trennung im Verhältnis zur größten, zwei Maße; 30 Instanzen × 8 feste Reihenfolgen je Satz."""
    cases = [("Ringe Gauß σ = 1", lambda s: S.rings(100, s), ("gauss", 1.0)), ("Ringe Gauß σ = 0,5", lambda s: S.rings(100, s), ("gauss", 0.5)),
             ("Ringe Polynom 2", lambda s: S.rings(100, s), ("poly", 2)), ("Liefergebiet n = 100 Polynom 2", lambda s: S.delivery_area(100, s), ("poly", 2))]
    out = {}
    for label, mk, spec in cases:
        row = {}
        for name, seeds in SETS.items():
            rf, rt, cos_min, runs, skipped = [], [], [], 0, 0
            for s in seeds:
                inst = mk(s)
                K = KK.kernel_matrix(spec, inst.X)
                Kt = K + 1.0
                qf = M.max_margin(K, inst.y, True)
                qt = M.max_margin(K, inst.y, False)
                if not (qf.certified and qt.certified):
                    skipped += 1
                    continue
                alphas = []
                for k in range(8):
                    order = S.order_of(inst.n, s * 8 + k)
                    r = P.run_dual(Kt, inst.y, order, 3000)
                    if not r.converged:
                        continue
                    runs += 1
                    a, b = M.found_margin_ratio(Kt, K, inst.y, r.alpha, qf, qt)
                    rf.append(a)
                    rt.append(b)
                    alphas.append(r.alpha)
                vs = [a * inst.y for a in alphas]
                cs = [float(v1 @ Kt @ v2 / np.sqrt((v1 @ Kt @ v1) * (v2 @ Kt @ v2))) for i, v1 in enumerate(vs) for v2 in vs[i + 1:]]
                if cs:
                    cos_min.append(min(cs))
            row[name] = {"runs": runs, "instances_skipped": skipped, "ratio_free_b": stat(rf), "ratio_phi1": stat(rt), "cos_min_per_instance": stat(cos_min)}
        out[label] = row
    return out


def sec_sigmoid():
    """Sigmoid tanh(a x·y + c) auf den Ringen (n = 100 > d = 2), Grenze 1 000 Epochen: indefinite Gram-Matrizen, Konvergenz ohne Meldung."""
    out = {}
    for a, c in [(1.0, -1.0), (0.5, -1.0), (0.1, -1.0), (1.0, 0.0), (0.1, 0.0), (1.0, 1.0), (0.1, 1.0)]:
        res, indef = [], 0
        for s in SET_A:
            inst = S.rings(100, s)
            r, Kt = run(inst, ("sigmoid", a, c), 1000, s)
            res.append(r)
            K = Kt - 1.0
            ev = np.linalg.eigvalsh((K + K.T) / 2)
            indef += int(ev[0] < -100 * np.finfo(float).eps * ev[-1])
        d = converged_stats(res)
        d["indefinite"] = indef
        out[KK.kernel_name(("sigmoid", a, c))] = d
    return out


def sec_kerntrick():
    """Dual gegen primal im expliziten Merkmalsraum (und gegen sklearn.linear_model.Perceptron), Polynomkerne; 50 Instanzen je Zeile."""
    try:
        from sklearn.linear_model import Perceptron
    except ImportError:  # pragma: no cover
        Perceptron = None
    out = {}
    for label, mk, d in [("Ringe Polynom 2", lambda s: S.rings(100, s), 2), ("Ringe Polynom 3", lambda s: S.rings(100, s), 3),
                         ("Liefergebiet n = 100 Polynom 2", lambda s: S.delivery_area(100, s), 2), ("Liefergebiet n = 100 Polynom 3", lambda s: S.delivery_area(100, s), 3)]:
        equal = skl_equal = 0
        maxrel = 0.0
        ups = []
        for s in range(1, 51):
            inst = mk(s)
            order = S.order_of(inst.n, s)
            Kt = KK.gram_tilde(("poly", d), inst.X)
            rd = P.run_dual(Kt, inst.y, order, 3000)
            Phi = KK.with_bias_column(KK.poly_features(inst.X, d))
            wp, up, ep, cp = P.run_primal(Phi, inst.y, order, 3000)
            wd = Phi.T @ (rd.alpha * inst.y)
            equal += int(rd.updates == up and rd.epochs == ep and rd.converged == cp)
            maxrel = max(maxrel, float(np.linalg.norm(wd - wp) / np.linalg.norm(wp)))
            ups.append(rd.updates)
            if Perceptron is not None:
                clf = Perceptron(penalty=None, alpha=0.0, fit_intercept=False, max_iter=ep, tol=None, shuffle=False, eta0=1.0)
                clf.fit(Phi[order], inst.y[order])
                skl_equal += int(np.array_equal(clf.coef_[0], wp))
        out[label] = {"instances": 50, "equal_runs": equal, "max_rel_w_dual_vs_primal": maxrel, "sklearn_equal": skl_equal, "updates": stat(ups)}
    return out


SECTIONS = {"ringe": sec_ringe, "bandbreite": sec_bandbreite, "liefergebiet": sec_liefergebiet, "monde": sec_monde, "zufall": sec_zufall,
            "d2": sec_d2, "marge": sec_marge, "sigmoid": sec_sigmoid, "kerntrick": sec_kerntrick}


def compute(only=None) -> dict:
    data = {"meta": {"seeds_A": [SET_A[0], SET_A[-1]], "seeds_B": [SET_B[0], SET_B[-1]], "max_updates": C.MAX_UPDATES,
                     "erzeuger": "SplitMix64 (ktp_rng.py)"}}
    for name, fn in SECTIONS.items():
        if only and name not in only:
            continue
        t = time.time()
        data[name] = fn()
        print(f"  {name}: fertig", flush=True)
        del t  # keine Zeiten als Befund
    return data


def _flat(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flat(f"{prefix}/{k}", v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flat(f"{prefix}[{i}]", v, out)
    else:
        out[prefix] = obj


def compare(old: dict, new: dict, rel: float = 1e-6) -> list[str]:
    a, b = {}, {}
    _flat("", old, a)
    _flat("", new, b)
    diffs = []
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            diffs.append(f"{k}: fehlt auf einer Seite")
            continue
        x, y = a[k], b[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(x - y) > rel * max(abs(x), abs(y), 1e-12) and abs(x - y) > 1e-12:
                diffs.append(f"{k}: {x} gegen {y}")
        elif x != y:
            diffs.append(f"{k}: {x!r} gegen {y!r}")
    return diffs


def main(argv):
    only = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
    if "--check" in argv:
        old = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        new = compute(only)
        if only:
            old = {k: v for k, v in old.items() if k in new}
        diffs = compare(old, new)
        for d in diffs[:40]:
            print("ABWEICHUNG", d)
        print(f"{len(diffs)} Abweichungen")
        return 1 if diffs else 0
    data = compute(only)
    if only:
        print("nur Teilabschnitte gerechnet, nichts geschrieben")
        return 0
    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
