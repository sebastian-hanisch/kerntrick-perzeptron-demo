"""Jede Zahl der README (Tabelle „Zahl → Test“): (1) steht die Zahl der eingefrorenen Messreihe `ktp_sweep.json` im Text der README, (2) liegt eine Neurechnung mit wenigen Zufalls-Seeds
im Band der Datei. Die vollständige Neurechnung beider Sätze macht `tools/ktp_make_sweep.py --check` (Messung dieser README, nicht Teil der Suite).

Zufallsergebnisse stehen als Band [Minimum, Maximum] über die Läufe der Datei; Ganzzahlen und Brüche der Handbeispiele stehen exakt in `test_handbeispiele.py`."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ktp_constants as C
import ktp_evaluation as ev
import ktp_kernels as KK
import ktp_margin as M
import ktp_perceptron as P
import ktp_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
SW = ev.load_sweep()
de, band = ev.de, ev._band


def _range(rows: dict, key: str = "updates"):
    """[Minimum, Maximum] über alle Läufe beider Sätze."""
    return min(rows[k][key]["min"] for k in "AB"), max(rows[k][key]["max"] for k in "AB")


def _run(inst, spec, epochs, seed, updates=C.MAX_UPDATES):
    return P.run_dual(KK.gram_tilde(spec, inst.X), inst.y, S.order_of(inst.n, seed), epochs, updates)


# ------------------------------------------------------------------ (1) die Zahlen der README stehen in der Datei
def _must_be_in_readme(*parts):
    for part in parts:
        assert part in README, part


def test_readme_numbers_ringe():
    r = SW["ringe"]
    assert [r["Linear"][k]["converged"] for k in "AB"] == [0, 0] and [r["Linear"][k]["runs"] for k in "AB"] == [30, 30]
    _must_be_in_readme(f"Korrekturen nach 3 000 Epochen im Median {band(r['Linear'], ('updates', 'median'))}",
                       f"Gauß σ = 0,25 / 0,5 / 1: 60 von 60 mit {band(r['Gauß σ = 0,25'], ('updates', 'median'))} / {band(r['Gauß σ = 0,5'], ('updates', 'median'))} / {band(r['Gauß σ = 1'], ('updates', 'median'))} Korrekturen im Median",
                       f"Polynom 2: {band(r['Polynom 2'], ('updates', 'median'))}; Polynom 3: {band(r['Polynom 3'], ('updates', 'median'))}")
    for k in ("Gauß σ = 0,25", "Gauß σ = 0,5", "Gauß σ = 1", "Polynom 2", "Polynom 3"):
        assert [r[k][s]["converged"] for s in "AB"] == [30, 30]
    lo = min(_range(r[k])[0] for k in ("Gauß σ = 0,25", "Gauß σ = 0,5", "Gauß σ = 1"))
    hi = max(_range(r[k])[1] for k in ("Gauß σ = 0,25", "Gauß σ = 0,5", "Gauß σ = 1"))
    _must_be_in_readme(f"alle 60 Läufe zwischen {de(lo)} und {de(hi)}")


def test_readme_numbers_liefergebiet_and_monde():
    lg = SW["liefergebiet"]
    p2, p200 = lg["n100 Polynom 2"], lg["n200 Polynom 2"]
    assert [p2[k]["converged"] for k in "AB"] == [30, 30] and [p200[k]["converged"] for k in "AB"] == [20, 20]
    assert [p200[k]["certified"] for k in "AB"] == [20, 20] and [p200[k]["bound_violations"] for k in "AB"] == [0, 0]
    _must_be_in_readme(f"60 von 60 mit {band(p2, ('updates', 'median'))} Korrekturen im Median", f"40 von 40 mit {band(p200, ('updates', 'median'))}",
                       f"das {band(p200, ('bound_over_updates', 'median'), 1)}".replace(" bis ", "- bis ") + "-Fache")
    mo = SW["monde"]
    for noise in ("0", "0,1"):
        for kern in ("Linear", "Polynom 2"):
            assert [mo[f"Rauschen {noise} {kern}"][k]["converged"] for k in "AB"] == [0, 0]
        assert [mo[f"Rauschen {noise} Polynom 3"][k]["converged"] for k in "AB"] == [20, 20]

    def meds(kern):
        v = [mo[f"Rauschen {n} {kern}"][k]["updates"]["median"] for n in ("0", "0,1") for k in "AB"]
        return min(v), max(v)

    p3, g25, g5, g1 = meds("Polynom 3"), meds("Gauß σ = 0,25"), meds("Gauß σ = 0,5"), meds("Gauß σ = 1")
    _must_be_in_readme(f"Polynom 3: 40 von 40 je Rauschstufe (20 je Satz) mit {de(p3[0], 1)} bis {de(p3[1])}", f"Gauß σ = 0,25 und 0,5: {de(min(g25[0], g5[0]))} bis {de(max(g25[1], g5[1]))}", f"σ = 1: {de(g1[0])} bis {de(g1[1])} Korrekturen im Median")


def test_readme_table_by_bandwidth_equals_the_frozen_curve():
    bw = SW["bandbreite"]
    for s, v in bw.items():
        lo, hi = _range(v)
        row = f"| {s} | {band(v, ('updates', 'median'))} | {de(lo)} bis {de(hi)} | {band(v, ('epochs', 'median'))} | {band(v, ('bound', 'median'), 1)} |"
        assert row in README, row
        assert [v[k]["converged"] for k in "AB"] == [30, 30] and [v[k]["certified"] for k in "AB"] == [30, 30] and [v[k]["bound_violations"] for k in "AB"] == [0, 0]
    _must_be_in_readme("in 540 von 540 Läufen dieselbe Meldung", "In 540 von 540 zertifizierten Läufen", "alle 540 Läufe konvergieren")
    assert ev.bandwidth_totals() == (540, 540, 540, 0)
    _must_be_in_readme(f"σ = 0,1: {band(bw['0,1'], ('updates', 'median'))}, σ = 0,05: {band(bw['0,05'], ('updates', 'median'))}, σ = 0,02: {band(bw['0,02'], ('updates', 'median'))}")


def test_readme_table_zufallsetiketten_equals_the_frozen_rows():
    for s, v in SW["zufall"].items():
        a, b = v["A"], v["B"]
        rng = (min(a["updates"]["min"], b["updates"]["min"]), max(a["updates"]["max"], b["updates"]["max"]))
        ratio = band(v, ("ratio_median_updates",), 0 if s == "0,25" else 1)
        row_start = f"| {s} | {de(a['converged'] + b['converged'])} von 60 | {de(a['within_3000'] + b['within_3000'])} von 60 |"
        assert row_start in README, row_start
        line = next(ln for ln in README.splitlines() if ln.startswith(row_start))
        assert f"| {de(rng[0])} bis {de(rng[1])} |" in line and f"| {band(v, ('true_updates_median',))} |" in line
        assert f"| {de(a['stops']['Korrekturgrenze'] + b['stops']['Korrekturgrenze'])} | {de(a['stops']['Epochengrenze'] + b['stops']['Epochengrenze'])} |" in line
        assert (s == "0,5") or ratio in line
        assert [v[k]["gram_positive_definite"] for k in "AB"] == [30, 30] and [v[k]["interpolation_solved"] for k in "AB"] == [30, 30]      # alle 60 trennbar
    z25 = SW["zufall"]["0,25"]
    assert 60 - sum(z25[k]["within_3000"] for k in "AB") == 21
    _must_be_in_readme("in 21 von 60 Zufalls-Seeds „Grenze erreicht“, obwohl alle 60 trennbar sind (bei 50 000 Epochen: 2 von 60)", "die Korrekturgrenze in 41 von 60 Läufen greift", "Bei σ = 0,25 liegt der größte Wert bei 166 997 Korrekturen")
    assert z25["A"]["updates"]["max"] == 166997 or z25["B"]["updates"]["max"] == 166997


def test_readme_table_d2_equals_the_frozen_rows():
    for k, v in SW["d2"].items():
        al = v["alle"]
        row = (f"| {k} | {de(al['converged'])} von 40 | {band(v, ('updates', 'median'))} | {de((v['A']['train_error_mean'] + v['B']['train_error_mean']) / 2, 3)} | "
               f"{de(al['test_error_mean'], 3)} ({de(al['test_error_se'], 3)}) |")
        assert row in README, row
    assert [SW["d2"][k]["alle"]["converged"] for k in SW["d2"]] == [40, 40, 20, 0, 0, 0]
    _must_be_in_readme("Testfehler gegen die wahre Klasse in den Gauß-Zeilen 0,158 bis 0,172", "(1,9 Standardfehler der Differenz, innerhalb der Streuung)", "(0,170; 0,165) sind nicht besser als die mit Konvergenz (0,158; 0,163)")
    assert "1,9 Standardfehler der Differenz" in ev.verdict_d2()[1] and "höchstens 0,009" in ev.verdict_d2()[1]


def test_readme_numbers_margin_and_order():
    mg = SW["marge"]
    g1, g5, p2, lg = mg["Ringe Gauß σ = 1"], mg["Ringe Gauß σ = 0,5"], mg["Ringe Polynom 2"], mg["Liefergebiet n = 100 Polynom 2"]
    f3 = lambda v, key: band(v, (key, "median"), 3)
    _must_be_in_readme(f"im Median {de(g1['A']['ratio_free_b']['median'], 3)} und {de(g1['B']['ratio_free_b']['median'], 3)} der größten Marge",
                       f"{de(g5['A']['ratio_free_b']['median'], 3)} (σ = 0,5)", f"Polynom 2: {f3(p2, 'ratio_free_b')} mit freiem b gegen {f3(p2, 'ratio_phi1')} mit dem Bias als Koordinate",
                       f"{f3(lg, 'ratio_free_b')}")
    _must_be_in_readme(f"{f3(g1, 'cos_min_per_instance')} (Gauß σ = 1), {f3(g5, 'cos_min_per_instance')} (σ = 0,5), {f3(p2, 'cos_min_per_instance')} (Polynom 2), {f3(lg, 'cos_min_per_instance')} (Liefergebiet)")
    for v in mg.values():
        assert v["A"]["runs"] >= 237 and v["B"]["runs"] >= 237 and v["A"]["instances_skipped"] == 0 and v["B"]["instances_skipped"] == 0
    assert p2["A"]["ratio_phi1"]["median"] > p2["A"]["ratio_free_b"]["median"] and p2["B"]["ratio_phi1"]["median"] > p2["B"]["ratio_free_b"]["median"]      # Polynom 2: (φ, 1) überschätzt


def test_readme_numbers_sigmoid_and_kerntrick():
    sig = SW["sigmoid"]
    assert all(v["indefinite"] == 30 and v["runs"] == 30 for v in sig.values()) and len(sig) == 7
    assert [sig[k]["converged"] for k in ("Sigmoid a = 1, c = −1", "Sigmoid a = 0,5, c = −1", "Sigmoid a = 0,1, c = −1")] == [0, 9, 30]
    assert [sig[k]["converged"] for k in sig if k not in ("Sigmoid a = 0,5, c = −1", "Sigmoid a = 0,1, c = −1")] == [0, 0, 0, 0, 0]
    assert sig["Sigmoid a = 0,1, c = −1"]["updates"]["median"] == 41
    _must_be_in_readme("in 30 von 30 Zufalls-Seeds indefinit in allen 7 Parametersätzen", "0, 9 und 30 von 30 konvergieren", "(dort 41 Korrekturen)")
    kt = SW["kerntrick"]
    assert sum(r["equal_runs"] for r in kt.values()) == 200 == sum(r["instances"] for r in kt.values()) == sum(r["sklearn_equal"] for r in kt.values())
    _must_be_in_readme("in 200 von 200 Instanzen (4 Zeilen zu 50)", f"w relativ höchstens {ev.sci(max(r['max_rel_w_dual_vs_primal'] for r in kt.values()))}")


def test_the_test_count_in_the_readme_equals_the_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    m = re.search(r"Die Testzahl: (\d+) Tests", README)
    assert m and int(m.group(1)) == n, (m and m.group(1), n)


# ------------------------------------------------------------------ (2) Neurechnung im Band der Datei (wenige Zufalls-Seeds, Satz A)
def test_recompute_ringe_linear_and_kernels_inside_the_frozen_band():
    lo, hi = _range(SW["ringe"]["Linear"])
    for seed in (1, 2, 3):
        r = _run(S.rings(100, seed), ("linear",), 3000, seed)
        assert not r.converged and lo <= r.updates <= hi
    for name, spec in (("Gauß σ = 0,5", ("gauss", 0.5)), ("Gauß σ = 1", ("gauss", 1.0)), ("Polynom 2", ("poly", 2)), ("Polynom 3", ("poly", 3))):
        lo, hi = _range(SW["ringe"][name])
        for seed in range(1, 11):
            r = _run(S.rings(100, seed), spec, 3000, seed)
            assert r.converged and lo <= r.updates <= hi, (name, seed)


def test_recompute_the_bandwidth_curve_at_three_widths():
    for sigma in (0.02, 1.0, 8.0):
        lo, hi = _range(SW["bandbreite"][KK.fmt_num(sigma)])
        for seed in range(1, 6):
            r = _run(S.rings(100, seed), ("gauss", sigma), 20000, seed)
            assert r.converged and lo <= r.updates <= hi, (sigma, seed)


def test_recompute_the_bound_is_never_exceeded():
    for sigma in (0.25, 1.0):
        for seed in range(1, 11):
            inst = S.rings(100, seed)
            K = KK.kernel_matrix(("gauss", sigma), inst.X)
            r = P.run_dual(K + 1.0, inst.y, S.order_of(100, seed), 3000)
            q = M.max_margin(K, inst.y, False)
            assert r.converged and q.certified and r.updates <= M.novikoff_bound(K + 1.0, q) + 1e-9
    bw = SW["bandbreite"]["1"]
    assert all(bw[k]["bound"]["min"] > 30 for k in "AB")


def test_recompute_random_labels_converge_with_many_more_corrections():
    lo, hi = _range(SW["zufall"]["0,1"])
    for seed in range(1, 7):
        base = S.rings(100, seed)
        r = _run(S.random_labels(base, seed), ("gauss", 0.1), 50000, seed)
        r0 = _run(base, ("gauss", 0.1), 50000, seed)
        assert r.converged and lo <= r.updates <= hi and r.updates > 4 * r0.updates, seed


def test_recompute_random_labels_sigma_025_inside_the_frozen_counts():
    z = SW["zufall"]["0,25"]
    lo, hi = _range(z)
    conv = within = 0
    for seed in range(1, 11):
        r = _run(S.random_labels(S.rings(100, seed), seed), ("gauss", 0.25), 50000, seed)
        conv += int(r.converged)
        within += int(r.converged and r.epochs <= 3000)
        assert lo <= r.updates <= hi or not r.converged
    frac_conv = sum(z[k]["converged"] for k in "AB") / 60.0
    frac_within = sum(z[k]["within_3000"] for k in "AB") / 60.0
    sd = lambda p: 4.5 * np.sqrt(p * (1 - p) / 10)                       # 4,5 Standardfehler für 10 Läufe
    assert abs(conv / 10 - frac_conv) <= max(sd(frac_conv), 0.2) and abs(within / 10 - frac_within) <= sd(frac_within)


def test_recompute_ten_percent_flipped_labels_keep_the_testerror_inside_the_frozen_spread():
    for seed in (1, 2, 3):
        inst = S.flip_labels(S.delivery_area(200, seed), C.D2_NOISE, seed)
        spec = ("gauss", 0.1)
        r = _run(inst, spec, 3000, seed)
        test = S.fresh_test_points(C.DATEN_LIEFER, 4000, seed)
        f = P.decision_values(KK.gram_tilde(spec, test.X, inst.X), r.alpha, inst.y)
        te = P.error_rate(f, test.y_true)
        row = SW["d2"]["Gauß σ = 0,1"]
        assert r.converged and 0.05 <= te <= 0.44 and abs(te - row["alle"]["test_error_mean"]) < 6 * 0.0285 + 0.1     # Streubereich je Zufalls-Seed 0,05 bis 0,44
        assert P.error_rate(r.s, inst.y) == 0.0                                                                       # Trainingsfehler 0 bei Konvergenz


def test_recompute_margin_ratios_inside_the_frozen_band():
    for label, make, spec in (("Ringe Gauß σ = 1", lambda s: S.rings(100, s), ("gauss", 1.0)), ("Ringe Polynom 2", lambda s: S.rings(100, s), ("poly", 2))):
        lo, hi = _range(SW["marge"][label], "ratio_free_b")
        free, tilde = [], []
        for seed in (1, 2, 3):
            inst = make(seed)
            K = KK.kernel_matrix(spec, inst.X)
            qf, qt = M.max_margin(K, inst.y, True), M.max_margin(K, inst.y, False)
            for k in range(2):
                r = P.run_dual(K + 1.0, inst.y, S.order_of(100, seed * 8 + k), 3000)
                a, b = M.found_margin_ratio(K + 1.0, K, inst.y, r.alpha, qf, qt)
                assert 0.0 < a <= 1.0 + 1e-9 and 0.0 < b <= 1.0 + 1e-9 and lo * 0.5 <= a <= hi * 1.001
                free.append(a)
                tilde.append(b)
        assert 0.01 < float(np.median(free)) < 0.45
        if "Polynom" in label:
            assert float(np.median(tilde)) > float(np.median(free))


def test_recompute_sigmoid_is_indefinite_and_runs_without_a_message():
    sig = SW["sigmoid"]["Sigmoid a = 0,1, c = −1"]
    lo, hi = sig["updates"]["min"], sig["updates"]["max"]
    for seed in range(1, 6):
        inst = S.rings(100, seed)
        K = KK.kernel_matrix(("sigmoid", 0.1, -1.0), inst.X)
        ev_min = KK.smallest_eigenvalue(K)
        assert ev_min < -100 * np.finfo(float).eps * np.linalg.eigvalsh(K)[-1]
        r = P.run_dual(K + 1.0, inst.y, S.order_of(100, seed), 1000)
        assert r.converged and lo <= r.updates <= hi


def test_recompute_moons_polynomial_two_never_converges_and_gauss_does():
    for seed in (1, 2, 3):
        inst = S.moons(100, seed, 0.1)
        assert not _run(inst, ("poly", 2), 2000, seed).converged
        r = _run(inst, ("gauss", 0.5), 2000, seed)
        assert r.converged and r.updates <= 40


def test_exact_cost_numbers_of_the_readme():
    assert (KK.poly_dimension(10, 3), KK.kernel_multiplications(10, 3), KK.break_even_points(10, 3), KK.break_even_points(10, 3, True)) == (286, 12, 23, 47) and 24 * 12 == 288 > 286


def test_readme_range_of_the_bound_medians_by_bandwidth():
    meds = [SW["bandbreite"][s][k]["bound"]["median"] for s in SW["bandbreite"] for k in "AB"]
    _must_be_in_readme(f"zwischen {de(min(meds), 1)} und {de(round(max(meds)))}")
