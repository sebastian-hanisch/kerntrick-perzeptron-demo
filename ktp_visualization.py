"""Plotly-Figuren: alle Achsen fest (`fixedrange`), damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt."""

import numpy as np
import plotly.graph_objects as go

import ktp_constants as C

_FONT = dict(size=13)


def _base_layout(fig: go.Figure, height: int = 420, title: str | None = None, xtitle: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=40, r=10, t=40 if title else 10, b=40), title=title, font=_FONT,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), template="plotly")
    fig.update_xaxes(fixedrange=True, title=xtitle)
    fig.update_yaxes(fixedrange=True, title=ytitle)
    return fig


def fig_points(X, y, alpha=None, grid=None, point_labels=None, height: int = 420, title: str | None = None) -> go.Figure:
    """Punkte (Klasse +1 und −1), Radius ∝ α_i (Zahl der Korrekturen), Entscheidungsgrenze f = 0 aus dem Gitter `grid = (xs, ys, F)`."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    fig = go.Figure()
    if grid is not None:
        xs, ys, F = grid
        lim = float(np.max(np.abs(F))) or 1.0
        fig.add_trace(go.Contour(x=xs, y=ys, z=F, showscale=False, hoverinfo="skip", contours=dict(coloring="heatmap", showlines=False, start=-lim, end=lim, size=lim / 6.0),
                                 colorscale=[[0, "rgba(181,84,59,0.32)"], [0.5, "rgba(255,255,255,0.0)"], [1, "rgba(47,107,101,0.32)"]], name="f(x)"))
        fig.add_trace(go.Contour(x=xs, y=ys, z=F, showscale=False, hoverinfo="skip", contours=dict(coloring="none", start=0, end=0, size=1, showlines=True),
                                 line=dict(color="#222222", width=2), name="f(x) = 0"))
    for label, color, name in ((1.0, C.COLOR_POS, "Klasse +1"), (-1.0, C.COLOR_NEG, "Klasse −1")):
        m = y == label
        size = 9.0 if alpha is None else 7.0 + 14.0 * np.sqrt(np.asarray(alpha, float)[m] / max(float(np.max(alpha)), 1.0))
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers" if point_labels is None else "markers+text", name=name,
                                 text=None if point_labels is None else [point_labels[i] for i in np.flatnonzero(m)], textposition="top center",
                                 marker=dict(size=size, color=color, line=dict(color="white", width=1)), hoverinfo="skip" if point_labels is None else "text"))
    pad = 0.15 * max(float(np.ptp(X[:, 0])), float(np.ptp(X[:, 1])), 1.0)
    fig.update_xaxes(range=[float(X[:, 0].min()) - pad, float(X[:, 0].max()) + pad])
    fig.update_yaxes(range=[float(X[:, 1].min()) - pad, float(X[:, 1].max()) + pad])
    return _base_layout(fig, height, title, "x₁", "x₂")


def fig_gram(K: np.ndarray, title: str | None = None) -> go.Figure:
    """Gram-Matrix K als Heatmap (Punkte nach Klasse und Winkel geordnet; Farbskala fest 0 bis 1)."""
    fig = go.Figure(go.Heatmap(z=K, zmin=0.0, zmax=1.0, colorscale="Viridis", colorbar=dict(title="K", thickness=12), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, 380, title, "Punkt j", "Punkt i")


def fig_sweep(sigmas, med, lo, hi, live_sigma=None, live_value=None) -> go.Figure:
    """Korrekturen gegen die Bandbreite σ (logarithmisch, beide Achsen): Median mit Band [Minimum, Maximum] aus den eingefrorenen Zufalls-Seeds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sigmas + sigmas[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor=C.COLOR_BAND, line=dict(width=0), hoverinfo="skip", name="Minimum bis Maximum"))
    fig.add_trace(go.Scatter(x=sigmas, y=med, mode="lines+markers", line=dict(color=C.COLOR_LINE), name="Median"))
    if live_sigma is not None:
        fig.add_trace(go.Scatter(x=[live_sigma], y=[max(live_value, 1)], mode="markers", marker=dict(size=14, color=C.COLOR_NEG, symbol="diamond"), name="diese Instanz (live)"))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _base_layout(fig, 380, None, "Bandbreite σ", "Korrekturen")


def fig_cost(ps, explicit, kernel) -> go.Figure:
    """Multiplikationen je Skalarprodukt: expliziter Merkmalsraum gegen Kernauswertung, logarithmische Achse."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(p) for p in ps], y=explicit, name="explizit (D Multiplikationen)", marker_color=C.COLOR_NEG))
    fig.add_trace(go.Bar(x=[str(p) for p in ps], y=kernel, name="Kern (p + d − 1 Multiplikationen)", marker_color=C.COLOR_POS))
    fig.update_layout(barmode="group")
    fig.update_yaxes(type="log")
    return _base_layout(fig, 340, None, "Eingabedimension p", "Multiplikationen je Skalarprodukt")
