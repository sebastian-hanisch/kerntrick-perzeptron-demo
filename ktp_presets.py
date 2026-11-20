"""Permalink (Begrenzen/Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler steht in SETTING_SPECS (Schlüssel = Session-State-Schlüssel = Name in `C.PRESETS`). Ein Preset nennt nur die Regler, die es ändert;
beim Anklicken werden alle übrigen Regler auf ihren Standardwert gesetzt (so ist genau ein Szenario aktiv und hervorgehoben)."""

import random
from dataclasses import dataclass

import streamlit as st

import ktp_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None   # geschlossene Auswahl (selectbox, select_slider): andere Werte aus der Adresszeile werden verworfen


SETTING_SPECS = {
    "xor_kern": SettingSpec("xk", str, C.DEFAULT_XOR_KERNEL, options=tuple(C.XOR_KERNELS)),
    "xor_order": SettingSpec("xo", str, C.DEFAULT_XOR_ORDER, options=tuple(C.XOR_ORDERS)),
    "ex2_grad": SettingSpec("d", int, C.DEFAULT_DEGREE, options=C.DEGREES),
    "ex2_p": SettingSpec("p", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "ex3_daten": SettingSpec("dt", str, C.DEFAULT_EX3_DATEN, options=C.EX3_DATEN),
    "ex3_kern": SettingSpec("k3", str, C.DEFAULT_EX3_KERNEL, options=tuple(C.EX3_KERNELS)),
    "ex3_n": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "ex3_mischen": SettingSpec("mx", bool, False),
    "ex4_sigma": SettingSpec("sg", float, C.DEFAULT_SIGMA, options=C.SIGMAS),
    "ex5_daten": SettingSpec("d5", str, C.DEFAULT_EX5_DATEN, options=C.EX5_DATEN),
    "ex5_etiketten": SettingSpec("et", str, C.DEFAULT_EX5_ETIKETTEN, options=C.EX5_ETIKETTEN),
    "ex5_kern": SettingSpec("k5", str, C.DEFAULT_EX5_KERNEL, options=tuple(C.EX5_KERNELS)),
    "ex5_epochen": SettingSpec("ep", int, C.DEFAULT_EPOCHS, options=C.EPOCH_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _cast(spec: SettingSpec, raw: str):
    """Wert aus der Adresszeile: auf Grenzen begrenzt bzw. auf die Auswahl eingerastet; None, wenn unbrauchbar."""
    if spec.caster is bool:
        return raw in ("1", "true", "True")
    value = spec.caster(raw)
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if state_key_is_even(spec):
        value -= value % C.N_STEP
        value = max(value, int(spec.lo))
    return value


def state_key_is_even(spec: SettingSpec) -> bool:
    """Die Punktzahl n muss gerade sein (halb und halb auf die Klassen verteilt)."""
    return spec.url_param == "n"


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _cast(spec, qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value) if isinstance(value, bool) else value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler: der Wert des Presets, sonst der Standardwert."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt alle Regler; die Hilfe nennt das Experiment, in dem es wirkt):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
