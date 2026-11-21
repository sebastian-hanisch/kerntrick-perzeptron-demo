"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Schnellstart-Button, fährt jeden Slider an seine Grenzen und
prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId` bei Charts mit gleichem Inhalt und `StreamlitAPIException` bei Slidern mit berechneten Grenzen).
Dazu je Schnellstart-Szenario die erwartete Verdikt-Zeile (ein Szenario ohne Effekt darf keinen Effekt melden; kein Test greift per Index auf einen Expander)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ktp_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _verdicts(at: AppTest) -> list[str]:
    return [e.value for kind in ("success", "info", "warning") for e in getattr(at, kind)]


def test_app_loads_without_exception():
    at = _fresh_app()
    assert len(at.get("plotly_chart")) == 5 and len(_verdicts(at)) >= 6          # fünf Experimente, je ein Verdikt (Experiment 5 zusätzlich der Negativbefund)


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_select_slider_option_runs():
    at = _fresh_app()
    for widget_key, options in (("ex4_sigma", C.SIGMAS), ("ex5_epochen", C.EPOCH_CHOICES)):
        for opt in (options[0], options[-1]):
            at = _fresh_app()
            at.sidebar.select_slider(key=widget_key).set_value(opt).run(timeout=TIMEOUT)
            assert not at.exception, (widget_key, opt, [str(e) for e in at.exception])


# ------------------------------------------------------------------ je Szenario die erwartete Zeile
EXPECTED = {
    "XOR von Hand (vier Punkte)": ["Polynom 2, Reihenfolge a, b, c, d", "α = (1, 1, 1, 1)", "f(x) = 8 x₁x₂"],
    "Ringe: Innenstadt gegen Umland": ["Gauß σ = 1, Ringe mit Lücke, n = 100:** konvergiert nach"],
    "Liefergebiet ohne Lücke": ["Polynom 2, Liefergebiet (Kreis), n = 100:** konvergiert nach"],
    "Monde": ["Gauß σ = 0,5, Monde, n = 100:** konvergiert nach"],
    "Zufalls-Labels": ["Labels zufällig", "Dieselbe Meldung „konvergiert“ wie bei wahren Labels"],
    "Schmale Bandbreite (σ = 0,05)": ["σ = 0,05 (γ = 200,000):**"],
    "Breite Bandbreite (σ = 8)": ["σ = 8 (γ = 0,008):**"],
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_scenario_shows_its_expected_verdict_line(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    text = "\n".join(_verdicts(at))
    for part in EXPECTED[name]:
        assert part in text, (name, part)


def test_the_xor_scenario_is_the_default_and_other_scenarios_change_something():
    base = "\n".join(_verdicts(_fresh_app()))
    at = _fresh_app()
    at.sidebar.button(key="preset_Monde").click().run(timeout=TIMEOUT)
    assert "\n".join(_verdicts(at)) != base


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n"] = "9999"
    at.query_params["sg"] = "0.3"
    at.query_params["k3"] = "Quatsch"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["ex3_n"] == C.N_MAX and at.session_state["ex4_sigma"] == 0.25 and at.session_state["ex3_kern"] == C.DEFAULT_EX3_KERNEL


def test_expanders_are_found_by_title_not_by_index():
    labels = [e.label for e in _fresh_app().expander]
    assert "📐 Mathematische Formulierung" in labels and "⚠️ Annahmen, Grenzen und bekannte Probleme" in labels and "So funktioniert der Kerntrick im Perceptron" in labels


def test_no_streamlit_error_element_in_any_scenario():
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.error, (name, [e.value for e in at.error])
