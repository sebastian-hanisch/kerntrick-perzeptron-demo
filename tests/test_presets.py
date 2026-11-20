"""Presets: Vollständigkeit, gültige Werte, jedes Szenario unterscheidet sich vom Standard, Permalink begrenzt und rastet ein."""

import pytest

import ktp_constants as C
import ktp_evaluation as ev
import ktp_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.SETTING_SPECS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_or_the_options():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
    assert C.N_MIN <= C.DEFAULT_N <= C.N_MAX and C.DEFAULT_N % C.N_STEP == 0


def test_the_first_preset_equals_the_defaults_and_every_other_differs_from_it_and_from_each_other():
    names = list(C.PRESETS)
    assert C.PRESETS[names[0]] == {}
    states = []
    for name in names:
        states.append({k: C.PRESETS[name].get(k, spec.default) for k, spec in P.SETTING_SPECS.items()})
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert states[i] != states[j], (names[i], names[j])


def test_settings_tuple_has_one_field_per_regler_in_the_same_order():
    assert list(ev.Settings._fields) == list(P.SETTING_SPECS)
    assert ev.default_settings() == ev.Settings(**{k: s.default for k, s in P.SETTING_SPECS.items()})


def test_defaults_are_valid_options_and_the_xor_defaults_are_the_hand_example():
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is not None:
            assert spec.default in spec.options, key
    assert C.DEFAULT_XOR_KERNEL == "Polynom 2" and C.DEFAULT_XOR_ORDER == "a, b, c, d" and len(C.XOR_ORDERS) == 24 and len(set(C.XOR_ORDERS)) == 24


@pytest.mark.parametrize("raw,key,expected", [("10000", "ex2_p", 1000), ("0", "ex2_p", 2), ("501", "ex3_n", 200), ("57", "ex3_n", 56), ("3", "ex3_n", 20),
                                              ("0.3", "ex4_sigma", 0.25), ("1000000", "ex5_epochen", 50000), ("Polynom 3", "ex3_kern", "Polynom 3"), ("Quatsch", "ex3_kern", None),
                                              ("1", "ex3_mischen", True), ("0", "ex3_mischen", False), ("9999999", "seed_input", C.SEED_MAX)])
def test_permalink_values_are_clamped_snapped_or_dropped(raw, key, expected):
    assert P._cast(P.SETTING_SPECS[key], raw) == expected


def test_url_parameters_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kernel_selection_lists_hold_valid_specs_without_a_dead_parameter():
    for table in (C.XOR_KERNELS, C.EX3_KERNELS, C.EX5_KERNELS):
        for name, spec in table.items():
            assert spec[0] in ("linear", "poly", "gauss", "sigmoid") and name  # Parameter stehen im Namen des Eintrags, kein zweiter Regler
    assert "Gauß σ = 0,25" in C.EX3_KERNELS and "Sigmoid a = 0,1, c = −1" in C.EX5_KERNELS
