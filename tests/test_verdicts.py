"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text, jede Schwelle und jeder Zweig stehen hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Analyse-Objekte; hier sind es Stellvertreter mit den Feldern, die das Verdikt liest. Die Zahlen aus `ktp_sweep.json`
(eingefroren) werden gegen die Datei selbst geprüft (Zahl im Text = Zahl in der Datei); ihre Neumessung prüft `tests/test_claims.py` bzw. `tools/ktp_make_sweep.py --check`."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

import ktp_evaluation as ev
import ktp_margin as M
import ktp_perceptron as P


def _run(updates, epochs, stop=P.STOP_CONVERGED, alpha=None, s=None, cyc=(None, None)):
    return P.RunResult(np.array([1.0, 1.0, 1.0, 1.0] if alpha is None else alpha), np.zeros(4) if s is None else s, updates, epochs, stop, cyc[0], cyc[1])


def _qp(w2):
    return M.QPResult(np.zeros(4), 0.0, w2, 0.0, 1.0, 5, True, 4)


def _xor(kern="Polynom 2", spec=("poly", 2), run=None, ups=None, same=True, conv=24, qp=None, bound=None, formula=None, order=(0, 1, 2, 3)):
    return NS(kern=kern, spec=spec, order=list(order), run=run or _run(4, 3), all_orders_updates=ups or [4] * 24, all_orders_same_alpha=same, all_orders_converged=conv,
              qp=qp, bound=bound, formula=formula)


# ------------------------------------------------------------------ Hilfsfunktionen
def test_german_number_formats():
    assert ev.de(146490) == "146 490" and ev.de(0.18, 2) == "0,18" and ev.de(-1.5, 1) == "−1,5" and ev.de(8.0) == "8" and ev.de(3) == "3"
    assert ev.pct(0.1812) == "18 %" and ev.pct(0.0123, 1) == "1,2 %"
    assert ev.sci(3.5e-16) == "3,5·10⁻¹⁶" and ev.sci(2.7e-14) == "2,7·10⁻¹⁴" and ev.sci(0) == "0" and ev.sci(9.97e-5) == "1,0·10⁻⁴"
    assert ev.parse_order("c, a, d, b") == [2, 0, 3, 1]


def test_band_shows_both_set_medians_rounded_and_half_integers():
    rows = {"A": {"m": 15.0}, "B": {"m": 15.0}}
    assert ev._band(rows, ("m",)) == "15"
    rows = {"A": {"m": 14.5}, "B": {"m": 16.0}}
    assert ev._band(rows, ("m",)) == "14,5 bis 16"
    rows = {"A": {"m": 0.8224}, "B": {"m": 0.8194}}
    assert ev._band(rows, ("m",), 2) == "0,82"
    assert ev._band({"A": {"m": 3.2}, "B": {"m": 3.3}}, ("m",), 1) == "3,2 bis 3,3"


# ------------------------------------------------------------------ Experiment 1
def test_xor_linear_cycle_is_a_warning_with_the_cycle_numbers():
    kind, text = ev.verdict_xor(_xor("Linear", ("linear",), _run(0, 1, P.STOP_CYCLE, s=np.zeros(4), cyc=(1, 4))))
    assert kind == "warning"
    assert "Linear, Reihenfolge a, b, c, d" in text and "nach 1 Epoche mit 4 Korrekturen kehrt der Zustand zurück (w = 0 wie am Anfang)" in text
    assert "konvergiert nie" in text and "2b ≥ 2" in text and "−2b ≥ 2" in text


def test_xor_cycle_plural_and_nonzero_state():
    kind, text = ev.verdict_xor(_xor("Linear", ("linear",), _run(0, 2, P.STOP_CYCLE, s=np.array([1.0, 0, 0, 0]), cyc=(2, 6)), order=(3, 2, 1, 0)))
    assert "nach 2 Epochen mit 6 Korrekturen" in text and "w = 0" not in text and "Reihenfolge d, c, b, a" in text


def test_xor_polynomial_success_names_every_number():
    kind, text = ev.verdict_xor(_xor(qp=_qp(0.5), bound=5.0, formula="8 x₁x₂"))
    assert kind == "success"
    assert "4 Korrekturen in 3 Epochen" in text and "α = (1, 1, 1, 1)" in text and "in 24 von 24 Reihenfolgen" in text
    assert "jedes Mal mit 4 Korrekturen und demselben α" in text and "f(x) = 8 x₁x₂" in text
    assert "Marge 2/‖w*‖ = 2,828" in text and "Schranke R̃²‖w*‖² = 5,00 ≥ 4 Korrekturen" in text


def test_xor_gauss_range_of_updates_and_closed_form_margin():
    ups = [4] * 12 + [6] * 12
    kind, text = ev.verdict_xor(_xor("Gauß σ = 1", ("gauss", 1.0), _run(6, 3, alpha=[2, 1, 2, 1]), ups=ups, same=False, qp=_qp(4 / (1 - np.exp(-2)) ** 2), bound=10.70))
    assert "mit 4 bis 6 Korrekturen" in text and "demselben α" not in text and "α = (2, 1, 2, 1)" in text
    assert "1 − exp(−2/σ²) = 0,864665" in text and "Marge 2/‖w*‖ = 0,865" in text and "Schranke R̃²‖w*‖² = 10,70 ≥ 6" in text


def test_xor_not_converged_is_a_limit_warning():
    kind, text = ev.verdict_xor(_xor(run=_run(900, 300, P.STOP_EPOCHS)))
    assert kind == "warning" and "keine Konvergenz nach 300 Epochen (Grenze erreicht)" in text


# ------------------------------------------------------------------ Experiment 2
def _cost(updates_dual=19, updates_primal=19, n_support=19, d=3, p=10, rel=3.5e-16, d2=10, k2=4):
    return NS(d=d, p=p, updates_dual=updates_dual, updates_primal=updates_primal, n_support=n_support, rel_w=rel, D2=d2, k2=k2)


def test_cost_verdict_numbers_and_the_two_break_even_points():
    kind, text = ev.verdict_cost(_cost())
    assert kind == "info"
    assert "Polynom 3 bei p = 10:** 286 Merkmale gegen 12 Multiplikationen" in text
    assert "höchstens 23 verschiedene Punkte" in text and "höchstens 47" in text
    assert "19 und 19 Korrekturen" in text and "3,5·10⁻¹⁶" in text
    assert "19 verschiedene korrigierte Punkte zu je 4 Multiplikationen, also 76 gegen 10: der Kern ist hier teurer" in text
    assert "200 von 200 Läufen" in text and "nicht Geschwindigkeit in 2D" in text and "nicht im Training" in text


def test_cost_verdict_cheaper_branch_at_two_support_points():
    _, text = ev.verdict_cost(_cost(n_support=2))
    assert "also 8 gegen 10: der Kern ist hier billiger" in text
    _, text = ev.verdict_cost(_cost(n_support=3))                      # 3·4 = 12 ≥ 10: teurer
    assert "teurer" in text
    _, text = ev.verdict_cost(_cost(n_support=2, d=2, d2=6, k2=3))     # Polynom 2, p = 2: 2·3 = 6 = 6: Gleichstand, nicht „billiger“ und nicht „teurer“
    assert "also 6 gegen 6: der Kern ist hier gleich teuer" in text


def test_cost_verdict_for_large_p_uses_the_exact_dimension():
    _, text = ev.verdict_cost(_cost(p=1000, d=3))
    assert "167 668 501 Merkmale gegen 1 002 Multiplikationen" in text


# ------------------------------------------------------------------ Experiment 3
def _dataset(converged=True, updates=15, epochs=2, bound=37.1, ratio=0.18, n_support=14, name="Ringe mit Lücke", n=60, spec=("gauss", 1.0), note="", noise=0.0, stop=P.STOP_CONVERGED, train=0.33):
    return NS(run=_run(updates, epochs, stop if not converged or stop != P.STOP_CONVERGED else P.STOP_CONVERGED), spec=spec, inst=NS(name=name, n=n), n_support=n_support,
              bound=bound, ratio_free=ratio, qp_note=note, noise=noise, train_error=train)


def test_dataset_converged_with_bound_ratio_and_frozen_row():
    kind, text = ev.verdict_dataset(_dataset())
    assert kind == "success"
    assert "Gauß σ = 1, Ringe mit Lücke, n = 60:** konvergiert nach 2 Epochen mit 15 Korrekturen (14 verschiedene Punkte mit α > 0)" in text
    assert "Schranke R̃²‖w*‖² = 37,1 ≥ 15 Korrekturen" in text and "hat 18 % der größten Marge" in text
    assert "(Ringe mit Lücke, n = 100, 30 Zufalls-Seeds je Satz, Grenze 3 000 Epochen): konvergiert in 30 und 30 von 30 Läufen, Korrekturen im Median 15" in text


def test_dataset_bound_violation_is_a_warning():
    kind, text = ev.verdict_dataset(_dataset(updates=40, bound=37.1))
    assert kind == "warning" and "wird überschritten" in text


def test_dataset_without_bound_names_the_reason_and_without_frozen_row_says_so():
    kind, text = ev.verdict_dataset(_dataset(bound=None, ratio=None, note="größte Trennung nicht zertifiziert (keine Zahl)", spec=("gauss", 4.0)))
    assert "Größte Trennung nicht zertifiziert (keine Zahl)." in text and "hat 18 %" not in text
    assert "keine eingefrorene Messzeile" in text


def test_dataset_limit_reached_is_a_warning_and_never_says_not_separable():
    kind, text = ev.verdict_dataset(_dataset(converged=False, stop=P.STOP_EPOCHS, updates=88558, epochs=3000, spec=("linear",), bound=None, ratio=None, train=0.33))
    assert kind == "warning"
    assert "Grenze erreicht (Epochengrenze) nach 3 000 Epochen und 88 558 Korrekturen, Trainingsfehler 33 %" in text and "heißt nicht „nicht trennbar“" in text
    assert "konvergiert in 0 und 0 von 30 Läufen" in text                    # eingefrorene Zeile: lineares Perceptron auf den Ringen


def test_dataset_moons_frozen_row_depends_on_the_noise():
    _, t0 = ev.verdict_dataset(_dataset(name="Monde", spec=("gauss", 0.5), noise=0.0))
    _, t1 = ev.verdict_dataset(_dataset(name="Monde", spec=("gauss", 0.5), noise=0.1))
    assert "Monde, Rauschen 0, n = 100, 20 Zufalls-Seeds je Satz, Grenze 2 000 Epochen" in t0 and "Monde, Rauschen 0,1, n = 100" in t1


# ------------------------------------------------------------------ Experiment 4
def test_bandwidth_verdict_reads_the_frozen_curve():
    a = NS(sigma=1.0, run=_run(15, 2), n=100, offdiag_mean=0.2594)
    kind, text = ev.verdict_bandwidth(a)
    assert kind == "info"
    assert "σ = 1 (γ = 0,500):** 15 Korrekturen in 2 Epochen" in text and "Außerdiagonaleintrag der Gram-Matrix ist 0,259" in text
    assert "σ = 0,02: 71 bis 72" in text and "σ = 0,25 bis 1: 15 bis 16" in text and "σ = 0,1: 20; σ = 2: 24" in text and "σ = 8: 977,5 bis 1 248" in text
    assert "in 540 von 540 Läufen dieselbe" in text and "540 von 540 zertifizierten Läufen nicht überschritten" in text


def test_bandwidth_verdict_limit_branch():
    a = NS(sigma=8.0, run=_run(5000, 3000, P.STOP_EPOCHS), n=100, offdiag_mean=0.99)
    assert "Grenze erreicht (Epochengrenze) nach 3 000 Epochen und 5 000 Korrekturen" in ev.verdict_bandwidth(a)[1]


# ------------------------------------------------------------------ Experiment 5
def _trap(etiketten="zufällig", spec=("gauss", 0.1), conv=True, updates=526, true_updates=20, true_conv=True, indefinite=False, eig=9.84e-3, train=0.0, test=0.471,
          daten="Ringe mit Lücke (n = 100)", stop=P.STOP_CONVERGED, epochs=69):
    return NS(etiketten=etiketten, spec=spec, daten=daten, run=_run(updates, epochs, stop), run_true=_run(true_updates, 3, P.STOP_CONVERGED if true_conv else P.STOP_EPOCHS),
              train_error=train, test_error=test, min_eigenvalue=eig, indefinite=indefinite)


def test_trap_random_labels_converge_with_the_same_message_and_more_corrections():
    kind, text = ev.verdict_trap(_trap())
    assert kind == "warning"
    assert "Meldung: „konvergiert nach 69 Epochen“" in text and "Korrekturen: 526 (mit den wahren Labels auf denselben Punkten: 20)" in text
    assert "Trainingsfehler 0,0 %, Testfehler gegen die wahre Klasse 47,1 % (1 000 frische Punkte)" in text
    assert "das 26,3-Fache an Korrekturen: aus der Meldung allein ist der Unterschied nicht zu sehen" in text and "Kleinster Eigenwert der Gram-Matrix: 9,8·10⁻³" in text
    assert "60 Zufalls-Seeds in zwei Sätzen, σ = 0,1): bei der Messgrenze von 50 000 Epochen konvergiert das Perceptron in 60 von 60, mit 3 000 Epochen in 58 von 60" in text
    assert "Korrekturen im Median 359 bis 379 gegen 20 mit den wahren Labels" in text


def test_trap_true_labels_have_no_comparison_and_no_frozen_row():
    kind, text = ev.verdict_trap(_trap(etiketten="wahr", true_updates=18, updates=18, epochs=3, test=0.0))
    assert kind == "success" and "mit den wahren Labels" not in text and "Eingefrorene Messung" not in text and "Fache" not in text


def test_trap_limit_reached_says_it_is_not_not_separable():
    kind, text = ev.verdict_trap(_trap(spec=("gauss", 0.25), stop=P.STOP_EPOCHS, epochs=3000, updates=88000))
    assert kind == "warning" and "Grenze erreicht“ (Epochengrenze) nach 3 000 Epochen und 88 000 Korrekturen" in text and "das heißt nicht „nicht trennbar“" in text
    assert "konvergiert das Perceptron in 58 von 60, mit 3 000 Epochen in 39 von 60" in text            # σ = 0,25: eingefroren


def test_trap_indefinite_gram_matrix_is_named_and_the_run_says_nothing():
    kind, text = ev.verdict_trap(_trap(etiketten="wahr", spec=("sigmoid", 0.1, -1.0), indefinite=True, eig=-75.102, updates=43, epochs=3))
    assert kind == "warning" and "kleinster Eigenwert −75,102" in text and "meldet dazu nichts" in text


def test_d2_negative_finding_names_the_ranges():
    kind, text = ev.verdict_d2()
    assert kind == "info"
    assert "in den 5 Gauß-Zeilen bei 0,158 bis 0,172" in text and "(Standardfehler 0,003 bis 0,011)" in text
    assert "in allen 40, in einem Teil (20 von 40) oder in keinem der 40 Läufe" in text and "höchstens 0,009" in text
    assert "Die Zeilen ohne Konvergenz (0,170; 0,165) sind nicht besser als die mit Konvergenz (0,158; 0,163)" in text
    assert "Polynom 2 (konvergiert nie): 0,187 (0,014)" in text and "kein Satz „Auswendiglernen schadet“" in text


def test_assumption_table_and_limit_texts_carry_the_frozen_numbers():
    table = ev.assumptions_table_markdown()
    assert "in 21 von 60 Zufalls-Seeds „Grenze erreicht“, obwohl alle 60 trennbar sind" in table
    assert "in 30 von 30 Zufalls-Seeds indefinit, in allen 7 Parametersätzen" in table and "a = 0,5, c = −1: 9 von 30" in table and "a = 0,1, c = −1: 30 von 30" in table
    assert "Nicht belegt" in table and table.count("\n| **") == 6
    limits = ev.limits_markdown()
    assert "**Annahmen.**" in limits and "**Grenzen.**" in limits and "**Bekannte Probleme und Unsicherheiten.**" in limits and "bis n = 100" in limits
