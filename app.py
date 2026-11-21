"""Kerntrick im Perceptron: ein Skalarprodukt austauschen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Das Perceptron rechnet nur mit Skalarprodukten; ersetzt man sie durch eine Kernfunktion, trennt es ohne Änderung der Regel auch Mengen, die eine Gerade nicht trennt, sofern der Kern im Merkmalsraum trennt.
Fünf Experimente an ausgedachten 2D-Beispielen (XOR mit vier Punkten, Ringe, Liefergebiet, Monde); die eingefrorenen Messreihen stehen in ktp_sweep.json.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ktp_constants as C
import ktp_evaluation as ev
import ktp_kernels as KK
import ktp_presets as PR
import ktp_scenario as S
import ktp_visualization as V
from ktp_layout import dunkles_design, kopf_css
import modul_frisch

modul_frisch.aktualisieren(__file__)      # geänderte Hilfsmodule nach einem Update neu laden (Cloud behält sie sonst bis zum Neustart)

st.set_page_config(page_title="Kerntrick im Perceptron – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _xor(kern, order):
    return ev.analyse_xor(kern, order)


@st.cache_data(show_spinner=False)
def _cost(d, p, seed):
    return ev.analyse_cost(d, p, seed)


@st.cache_data(show_spinner=False)
def _dataset(daten, kern, n, mischen, seed):
    return ev.analyse_dataset(daten, kern, n, mischen, seed)


@st.cache_data(show_spinner=False)
def _bandwidth(sigma, seed):
    return ev.analyse_bandwidth(sigma, seed)


@st.cache_data(show_spinner=False)
def _trap(daten, etiketten, kern, epochen, seed):
    return ev.analyse_trap(daten, etiketten, kern, epochen, seed)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (Zahlen vorher mit ev.de formatieren: deutsches Dezimalkomma, keine Locale-Überraschung)."""
    head = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ktp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Kerntrick im Perceptron: ein Skalarprodukt austauschen")
st.markdown(
    """
Das **Perceptron** rechnet nur mit **Skalarprodukten** der Punkte. Ersetzt man jedes Skalarprodukt durch eine **Kernfunktion** (der **Kerntrick**), trennt dieselbe Regel auch Mengen,
die eine Gerade nicht trennt, sofern der Kern im Merkmalsraum trennt (Monde: Polynom 2 nicht, Polynom 3 und Gauß-Kern ja). Die Seite zeigt das an ausgedachten 2D-Beispielen (XOR mit vier Punkten, zwei Ringe, Kunden im Liefergebiet, zwei Monde) und fragt, woran man
merkt, dass es klappt: die Meldung „konvergiert“ sagt weniger, als man liest (bei kleiner Bandbreite fast nichts über Signal oder Rauschen der Labels). Gezählt werden **Korrekturen, Epochen, Zufalls-Seeds und Multiplikationen**.
Weiter unten: fünf **🔬 Experimente**, **🚧 Wo die Annahmen enden**, **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kerntrick-perzeptron-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Ein **Perceptron** trennt rote und blaue Punkte mit einer geraden Linie und schiebt sie nach jedem Fehler ein Stück. Bei zwei Ringen, einem roten in der Mitte und einem blauen darum, scheitert das, denn keine gerade Linie trennt sie. Der **Kerntrick** behebt das, ohne die Regel zu ändern: Man stellt sich vor, das Gelände unter den Punkten zu falten, sodass die Mitte höher liegt. Dort trennt eine flache Fläche beide Farben. Gerechnet wird dieses Gelände dabei nicht. Das Perceptron braucht von den Punkten nur ein Ähnlichkeitsmaß (das **Skalarprodukt**), und man ersetzt es durch eine **Kernfunktion**, die dieselbe Ähnlichkeit im gefalteten Gelände direkt liefert. Die Meldung „konvergiert“ heißt allerdings nur: ein Durchgang ohne Fehler. Ob die Farben ein Muster haben, sagt sie nicht von allein.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/kerntrick-perzeptron-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert der Kerntrick im Perceptron", expanded=True):
    st.markdown(
        """
1. **Die Regel:** Gehe die Punkte in fester Reihenfolge durch. Liegt ein Punkt auf der falschen Seite (oder genau auf der Grenze), **korrigiere** ihn: sein Zähler α steigt um 1.
   **Konvergiert** heißt: eine ganze Epoche (ein Durchlauf) ohne Korrektur.
2. **Nur Skalarprodukte:** Die Entscheidung f(x) = Σ α_j y_j (K(x_j, x) + 1) braucht die Punkte nur über die Kernfunktion K. Der Bias steckt als Zusatzkoordinate in „+ 1“.
3. **Der Trick:** Wer K = x·y durch (x·y + 1)² oder exp(−‖x − y‖²/(2σ²)) ersetzt, ändert **nur diese Zeile**: das Perceptron arbeitet dann in einem Merkmalsraum, den es nie ausrechnet.
   Das Beispiel XOR mit vier Punkten rechnet sich von Hand.
        """
    )

st.markdown("## ❓ Kernfragen")
st.markdown(
    """
1. [Warum reicht eine Gerade nicht, und was ändert der Kern?](#exp1)
2. [Was kostet der Kern, was spart er?](#exp2)
3. [Wie viele Korrekturen braucht es, und welche Trennung findet das Perceptron?](#exp3)
4. [Wie hängt die Zahl der Korrekturen an der Bandbreite σ?](#exp4)
5. [Was sagt „konvergiert“, und was sagt es nicht?](#exp5)
"""
)
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

PR.load_permalink_settings()
PR.init_session_state_defaults()

with st.sidebar:
    PR.sidebar_presets()  # Schnellstart oben, das aktive Szenario ist hervorgehoben
    st.header("⚙️ Experiment 1: XOR von Hand")
    xor_kern = st.selectbox("Kern", list(C.XOR_KERNELS), key="xor_kern", help="Der Kern ersetzt das Skalarprodukt in der Regel. Linear konvergiert auf XOR nie.")
    xor_order = st.selectbox("Reihenfolge der vier Punkte", C.XOR_ORDERS, key="xor_order", help="Eine der 24 festen Reihenfolgen, in der jede Epoche die Punkte durchläuft.")
    st.header("⚙️ Experiment 2: Aufwand")
    ex2_grad = st.radio("Grad d des Polynomkerns", C.DEGREES, key="ex2_grad", horizontal=True, help="Kern (x·y + 1)^d.")
    ex2_p = st.slider("Eingabedimension p (Rechenbeispiel)", *PR.bounds("ex2_p"), key="ex2_p", help="Nur für die Aufwandszahlen; die Punkte der Anzeige bleiben zweidimensional.")
    st.header("⚙️ Experiment 3: Datensätze")
    ex3_daten = st.selectbox("Datensatz", C.EX3_DATEN, key="ex3_daten", help="Ringe mit Lücke, Liefergebiet (Kreis, keine Lücke), Monde ohne oder mit Rauschen.")
    ex3_kern = st.selectbox("Kern", list(C.EX3_KERNELS), key="ex3_kern", help="Die Bandbreite σ steht im Namen des Gauß-Kerns.")
    ex3_n = st.slider("Zahl der Punkte n", C.N_MIN, C.N_MAX, step=C.N_STEP, key="ex3_n", help="Halb und halb auf die beiden Klassen verteilt (Liefergebiet: nach der Regel).")
    ex3_mischen = st.toggle("Reihenfolge in jeder Epoche neu mischen", key="ex3_mischen", help="Aus: eine feste Reihenfolge je Zufalls-Seed (die gemessene Regel). An: neue Zufallsreihenfolge je Epoche (nicht gemessen).")
    st.header("⚙️ Experiment 4: Bandbreite")
    ex4_sigma = st.select_slider("Bandbreite σ des Gauß-Kerns", C.SIGMAS, key="ex4_sigma", format_func=KK.fmt_num, help="Längenmaß in den Einheiten der Daten; γ = 1/(2σ²).")
    st.header("⚙️ Experiment 5: Die Falle")
    ex5_daten = st.selectbox("Daten", C.EX5_DATEN, key="ex5_daten", help="Ringe mit Lücke (n = 100) oder Liefergebiet ohne Lücke (n = 200).")
    ex5_etiketten = st.selectbox("Labels", C.EX5_ETIKETTEN, key="ex5_etiketten", help="Wahr, 10 % der Labels vertauscht, oder alle zufällig vertauscht (gleiche Punkte).")
    ex5_kern = st.selectbox("Kern", list(C.EX5_KERNELS), key="ex5_kern", help="Das Sigmoid tanh(a x·y + c) ist im Allgemeinen kein Skalarprodukt: nur als Falle.")
    ex5_epochen = st.select_slider("Obergrenze der Epochen", C.EPOCH_CHOICES, key="ex5_epochen", format_func=lambda v: ev.de(v),
                                   help="Voreinstellung 3 000. Bis 50 000 rechnet die Seite mit Spinner; die Korrekturgrenze ist 300 000.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Instanz", *PR.bounds("seed_input"), key="seed_input", step=1, help="Punkte, Reihenfolge und Labelsvertauschung folgen aus diesem Zufalls-Seed.")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=PR.randomize_seed)

PR.sync_query_params({k: st.session_state[k] for k in PR.SETTING_SPECS})
seed = int(seed)

# =====================================================================================================================================================
# Experiment 1
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: XOR von Hand, ein Skalarprodukt austauschen", anchor="exp1")
st.caption("Vier Punkte a = (1, 1), b = (−1, −1) mit y = +1 und c = (1, −1), d = (−1, 1) mit y = −1. Im Eingaberaum nicht durch eine Gerade trennbar.")
ax = _xor(xor_kern, xor_order)
c1, c2 = st.columns([1, 1])
with c1:
    inst = S.xor_instance()
    grid = ev.decision_grid(inst, ax.spec, ax.run.alpha, 80) if ax.run.converged else None
    st.plotly_chart(V.fig_points(inst.X, inst.y, alpha=ax.run.alpha if ax.run.converged else None, grid=grid, point_labels=list("abcd"), height=360), width="stretch", key="xor_chart")
    st.caption("Farbe: Klasse; Radius ∝ Zahl der Korrekturen des Punktes; Linie: f(x) = 0 (nur bei Konvergenz).")
with c2:
    st.markdown("**Gram-Matrix K̃ = K + 1** (Bias als Zusatzkoordinate)")
    st.markdown(md_table([""] + list("abcd"), [[n] + [ev.de(v, 4 if ax.spec[0] == "gauss" else 0) for v in ax.Kt[i]] for i, n in enumerate("abcd")]))
    m1, m2, m3 = st.columns(3)
    m1.metric("Korrekturen", ev.de(ax.run.updates) if ax.run.converged else f"{ev.de(ax.run.updates)} je Zyklus" if ax.run.stop == "Zyklus" else ev.de(ax.run.updates))
    m2.metric("Marge 2/‖w*‖", ev.de(ax.qp.margin, 3) if ax.qp is not None else "nicht trennbar")
    m3.metric("Schranke R̃²‖w*‖²", ev.de(ax.bound, 2) if ax.bound is not None else "–")
st.markdown("**Protokoll** (jede Prüfung: Punkt, f(x_i), Fehler, Zähler α danach)")
rows = []
shown = ax.trace[: 4 * 12] if ax.spec[0] == "gauss" else ax.trace[:60]
for step in shown:
    rows.append([ev.de(step.epoch), "abcd"[step.point], ev.de(step.f, 4 if ax.spec[0] == "gauss" else 0), ev.de(float(inst.y[step.point]) * step.f, 4 if ax.spec[0] == "gauss" else 0),
                 "ja: Korrektur" if step.error else "nein", "(" + ", ".join(ev.de(v) for v in step.alpha) + ")"])
st.markdown(md_table(["Epoche", "Punkt", "f(x_i)", "y_i f(x_i)", "Fehler (y f ≤ 0)", "α danach"], rows))
if len(ax.trace) > len(shown):
    st.caption(f"Gezeigt sind die ersten {ev.de(len(shown))} von {ev.de(len(ax.trace))} Prüfungen.")
if ax.w_after_epoch:
    st.markdown("**Normale w = (w₁, w₂, b) nach jeder Epoche:** " + "; ".join(f"({', '.join(ev.de(v) for v in w)})" for w in ax.w_after_epoch))
kind, text = ev.verdict_xor(ax)
getattr(st, kind)(text)

# =====================================================================================================================================================
# Experiment 2
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: Dasselbe im Merkmalsraum, und was es kostet", anchor="exp2")
st.caption("Zähleinheit: Multiplikationen je Skalarprodukt (kein Zeitmaß). Der Polynomkern (x·y + 1)^d hat den expliziten Merkmalsraum der Dimension D = C(p + d, d).")
cost = _cost(int(ex2_grad), int(ex2_p), seed)
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(md_table(["p", "Merkmale D", "Multiplikationen Kern", "billiger bis … Punkte", "mit Bau von φ(x)"], [[ev.de(p), ev.de(D), ev.de(k), ev.de(n0), ev.de(n1)] for p, D, k, n0, n1 in cost.rows]))
    st.caption("„billiger bis“: größte Zahl verschiedener korrigierter Punkte, bei der die duale Auswertung f(x) an einem neuen Punkt weniger Multiplikationen braucht als der explizite Weg; "
               "Gleichstand zählt als nicht billiger. Die Spalte mit Bau rechnet 2D Multiplikationen für den expliziten Weg (Annahme: mindestens eine je Merkmal).")
with c2:
    st.plotly_chart(V.fig_cost([r[0] for r in cost.rows], [r[1] for r in cost.rows], [r[2] for r in cost.rows]), width="stretch", key="cost_chart")
m1, m2, m3 = st.columns(3)
m1.metric("Korrekturen dual (Ringe, n = 60)", ev.de(cost.updates_dual))
m2.metric("Korrekturen primal im Merkmalsraum", ev.de(cost.updates_primal))
m3.metric("Normalen dual gegen primal, relativ", ev.sci(cost.rel_w))
kind, text = ev.verdict_cost(cost)
getattr(st, kind)(text)

# =====================================================================================================================================================
# Experiment 3
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: Ringe, Liefergebiet, Monde", anchor="exp3")
st.caption(f"Eine Instanz zum Zufalls-Seed der Seitenleiste; Obergrenze {ev.de(C.DEFAULT_EPOCHS)} Epochen und {ev.de(C.MAX_UPDATES)} Korrekturen. Die Zahlen der Messung sind eingefroren (Tabelle unten).")
da = _dataset(ex3_daten, ex3_kern, int(ex3_n), bool(ex3_mischen), seed)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Korrekturen", ev.de(da.run.updates))
m2.metric("Epochen", ev.de(da.run.epochs), delta="konvergiert" if da.run.converged else "Grenze erreicht", delta_color="off")
m3.metric("Schranke R̃²‖w*‖²", ev.de(da.bound, 1) if da.bound is not None else "–")
m4.metric("Marge in % der größten", ev.pct(da.ratio_free) if da.ratio_free is not None else "–", help="Maß mit freiem b: Abstand der gefundenen Trennung zu den nächsten Punkten, geteilt durch den der größten Trennung.")
st.plotly_chart(V.fig_points(da.inst.X, da.inst.y, alpha=da.run.alpha, grid=da.grid, height=440), width="stretch", key="dataset_chart")
st.caption("Farbe: Klasse; Radius ∝ Zahl der Korrekturen des Punktes; Fläche: Vorzeichen von f; Linie: f(x) = 0. Bei „Grenze erreicht“ zeigt die Fläche den letzten Stand.")
kind, text = ev.verdict_dataset(da)
getattr(st, kind)(text)
sw = ev.load_sweep()
with st.expander("Eingefrorene Messung: Korrekturen je Datensatz und Kern (Satz A und Satz B, Median [Minimum; Maximum])"):
    def _cell(r):
        u = r["updates"]
        return f"{ev.de(r['converged'])}/{ev.de(r['runs'])}; {ev.de(u['median'], 0 if float(u['median']).is_integer() else 1)} [{ev.de(u['min'])}; {ev.de(u['max'])}]"

    tab = [[f"Ringe mit Lücke, n = 100 ({ev.de(sw['ringe'][k]['A']['runs'])} Zufalls-Seeds je Satz, Grenze 3 000)", k, _cell(sw["ringe"][k]["A"]), _cell(sw["ringe"][k]["B"])] for k in sw["ringe"]]
    tab += [[f"Liefergebiet, n = 100 (30 Zufalls-Seeds je Satz, Grenze 3 000)", k[5:], _cell(v["A"]), _cell(v["B"])] for k, v in sw["liefergebiet"].items() if k.startswith("n100")]
    tab += [[f"Monde, {k.split(' ')[0]} {k.split(' ')[1]} (20 Zufalls-Seeds je Satz, Grenze 2 000)", " ".join(k.split(" ")[2:]), _cell(v["A"]), _cell(v["B"])] for k, v in sw["monde"].items()]
    st.markdown(md_table(["Datensatz", "Kern", "Satz A: konvergiert; Korrekturen", "Satz B: konvergiert; Korrekturen"], tab))
    st.caption("„konvergiert“ heißt hier: Zahl der Läufe mit einer Epoche ohne Korrektur von allen Läufen. Die Monde haben Rauschen mit Halbbreite 0 bzw. 0,1 (eigener Erzeuger, beschränkt).")

# =====================================================================================================================================================
# Experiment 4
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 4: Bandbreite", anchor="exp4")
st.caption("Ringe mit Lücke, n = 100, Gauß-Kern. Die Kurve ist eingefroren (je 30 Zufalls-Seeds in zwei Sätzen); der Punkt ist die Instanz der Seitenleiste, live gerechnet.")
ba = _bandwidth(float(ex4_sigma), seed)
cv = ev.sweep_curve()
c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(V.fig_sweep(cv["sigmas"], cv["median"], cv["min"], cv["max"], live_sigma=float(ex4_sigma), live_value=ba.run.updates), width="stretch", key="sweep_chart")
    st.caption("Median der beiden Satz-Mediane, Band: Minimum bis Maximum über alle 60 Läufe je σ.")
with c2:
    st.plotly_chart(V.fig_gram(ba.gram, title=f"Gram-Matrix K für σ = {KK.fmt_num(ba.sigma)} (nach Klasse und Winkel geordnet)"), width="stretch", key="gram_chart")
    st.caption("Kleines σ: die Matrix gleicht der Einheitsmatrix (jeder Punkt kennt nur sich selbst). Großes σ: sie gleicht der Einsmatrix. Das ist hier nur die Anzeige der Matrix; die Kondition ist nicht Gegenstand.")
kind, text = ev.verdict_bandwidth(ba)
getattr(st, kind)(text)

# =====================================================================================================================================================
# Experiment 5
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 5: Die Falle, „konvergiert“ meldet nichts", anchor="exp5")
st.caption("Wer „konvergiert“ liest, schließt auf ein gutes Modell. Die Meldung sagt nur, dass eine Epoche ohne Korrektur blieb. Die Tabellen sind eingefroren; der Lauf oben ist eine Instanz.")
if int(ex5_epochen) > C.SPINNER_EPOCHS:
    with st.spinner(f"Rechnet bis zu {ev.de(int(ex5_epochen))} Epochen und {ev.de(C.MAX_UPDATES)} Korrekturen, das kann einige Sekunden dauern ..."):
        ta = _trap(ex5_daten, ex5_etiketten, ex5_kern, int(ex5_epochen), seed)
else:
    ta = _trap(ex5_daten, ex5_etiketten, ex5_kern, int(ex5_epochen), seed)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Meldung", "konvergiert" if ta.run.converged else "Grenze erreicht", delta=f"nach {ev.de(ta.run.epochs)} Epochen", delta_color="off")
m2.metric("Korrekturen", ev.de(ta.run.updates))
m3.metric("Trainingsfehler / Testfehler", f"{ev.pct(ta.train_error, 1)} / {ev.pct(ta.test_error, 1)}", help="Testfehler gegen die wahre Klasse an frischen Punkten desselben Erzeugers.")
m4.metric("Kleinster Eigenwert der Gram-Matrix", ev.sci(ta.min_eigenvalue))
kind, text = ev.verdict_trap(ta)
getattr(st, kind)(text)

st.markdown("**Zufalls-Labels, eingefroren** (Ringe mit Lücke, n = 100, Gauß-Kern, 60 Zufalls-Seeds in zwei Sätzen zu je 30; Messgrenze 50 000 Epochen, Korrekturgrenze 300 000)")
zrows = []
for s, row in sw["zufall"].items():
    a_, b_ = row["A"], row["B"]
    zrows.append([s, f"{ev.de(a_['converged'] + b_['converged'])} von 60", f"{ev.de(a_['within_3000'] + b_['within_3000'])} von 60",
                  ev._band(row, ("updates", "median")), ev._band(row, ("true_updates_median",)), f"{ev.de(a_['stops']['Korrekturgrenze'] + b_['stops']['Korrekturgrenze'])} von 60"])
st.markdown(md_table(["σ", "konvergiert (Messgrenze)", "konvergiert innerhalb 3 000 Epochen", "Korrekturen, Median (beide Sätze)", "Korrekturen mit wahren Labels, Median", "Korrekturgrenze erreicht"], zrows))
st.markdown("**Liefergebiet n = 200 mit 10 % vertauschten Labels, eingefroren** (40 Zufalls-Seeds in zwei Sätzen zu je 20, Grenze 3 000 Epochen, 4 000 frische Testpunkte)")
drows = []
for k, row in sw["d2"].items():
    al = row["alle"]
    drows.append([k, f"{ev.de(al['converged'])} von 40", ev._band(row, ("updates", "median")), f"{ev.de((row['A']['train_error_mean'] + row['B']['train_error_mean']) / 2, 3)}",
                  f"{ev.de(al['test_error_mean'], 3)} ({ev.de(al['test_error_se'], 3)})"])
st.markdown(md_table(["Kern", "konvergiert", "Korrekturen, Median", "Trainingsfehler (Mittel)", "Testfehler gegen die wahre Klasse, Mittel (Standardfehler)"], drows))
kind, text = ev.verdict_d2()
getattr(st, kind)(text)

# =====================================================================================================================================================
# Grenzen
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(ev.assumptions_table_markdown())

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(ev.limits_markdown())

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/kerntrick-perzeptron-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Punkte $x_i \in \mathbb{R}^2$ mit Etiketten $y_i \in \{-1, +1\}$ und eine Kernfunktion $K$. Gesucht sind Zähler $\alpha_i$, mit denen $f(x) = \sum_j \alpha_j y_j \tilde K(x_j, x)$ jeden Punkt richtig einordnet. Gemessen wird die Zahl der Korrekturen und der Epochen.

| Symbol | Bedeutung |
|---|---|
| $x_i$, $y_i$ | Punkt $i$ (zwei Koordinaten) und sein Etikett, $-1$ oder $+1$ |
| $\varphi$ | Merkmalsabbildung: das „gefaltete Gelände“; wird nie ausgerechnet, nur über $K(x, x') = \langle \varphi(x), \varphi(x') \rangle$ benutzt |
| $K$, $\tilde K$ | Kernfunktion; $\tilde K = K + 1$ legt den Bias als Zusatzkoordinate hinein |
| $\alpha_i$ | Zahl der Korrekturen des Punktes $i$ |
| $f$ | Entscheidungsfunktion $f(x) = \sum_j \alpha_j y_j \tilde K(x_j, x)$; das Vorzeichen gibt die Klasse |
| $\sigma$ | Bandbreite des Gauß-Kerns: Längenmaß in den Einheiten der Daten |
| $\tilde R^2$, $\lVert w^* \rVert$ | $\tilde R^2 = \max_i \tilde K(x_i, x_i)$ und Norm der größten Trennung; die Schranke für die Korrekturen ist $\tilde R^2 \lVert w^* \rVert^2$ |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/kerntrick-perzeptron-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(ev.MATH_LESEHILFE)
    st.markdown(ev.MATH_SYMBOLE)
    st.markdown(ev.MATH_RULE_INTRO)
    for eq in ev.MATH_EQUATIONS:
        st.latex(eq)
    st.markdown(ev.MATH_AFTER)
    for eq in ev.MATH_BOUND_EQUATIONS:
        st.latex(eq)
    st.markdown(ev.MATH_LITERATUR)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
