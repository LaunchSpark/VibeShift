"""Altair chart builders shared by ``eval-report`` (static PNGs) and the notebook.

Each builder takes a polars DataFrame from ``frames.py`` and returns a finished
chart: wrap it in ``mo.ui.altair_chart`` in marimo, or ``.save(path, ppi=200)``
headlessly (vl-convert-python does the rendering).

Colors are the validated reference palette (light and dark steps). Models keep a
fixed slot by name in ``MODEL_ORDER``, so a model's color never changes when
other models are added or missing; add a new model there to give it a slot. The
confusion matrix encodes counts on a single-hue blue ramp.
"""

from __future__ import annotations

import altair as alt
import polars as pl

from evaluation.common.labels import LABELS

# Fixed model -> categorical slot. Color follows the model, never its rank.
MODEL_ORDER = (
    "laya",
    "stacking",
    "transformers_distilroberta",
    "transformers_goemotions",
    "llm_zeroshot",
    "nrc_wordcount",
    "nrc_macro",
)

_FONT = "system-ui, -apple-system, Segoe UI, sans-serif"

_PALETTES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"],
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "on_dark": "#ffffff", "on_light": "#0b0b0b",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
        # Near-zero recedes toward the dark surface; high counts get bright.
        "ramp": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "on_dark": "#0b0b0b", "on_light": "#f0efec",
    },
}


def _palette(dark: bool) -> dict:
    return _PALETTES["dark" if dark else "light"]


def _style(chart: alt.TopLevelMixin, p: dict) -> alt.TopLevelMixin:
    axis = {"labelColor": p["ink2"], "titleColor": p["ink2"], "labelFont": _FONT,
            "titleFont": _FONT, "gridColor": p["grid"], "domainColor": p["axis"],
            "tickColor": p["axis"]}
    text = {"labelColor": p["ink2"], "titleColor": p["ink2"], "labelFont": _FONT,
            "titleFont": _FONT}
    return (
        chart.configure(background=p["surface"], font=_FONT)
        .configure_view(stroke=None)
        .configure_axis(**axis)
        .configure_legend(**text)
        .configure_header(**text)
        .configure_title(color=p["ink"], font=_FONT, anchor="start", fontSize=14)
    )


def _model_scale(models: list[str], p: dict) -> alt.Scale:
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(set(models) - set(MODEL_ORDER))  # no slot yet -> muted gray
    return alt.Scale(
        domain=known + unknown,
        range=[p["series"][MODEL_ORDER.index(m)] for m in known] + [p["muted"]] * len(unknown),
    )


def plot_model_comparison(df: pl.DataFrame, *, dark: bool = False) -> alt.FacetChart:
    """Macro-F1, Top-2 accuracy, and Brier per model, one panel per metric.

    Panels keep independent y-axes: Brier runs 0-2 and lower is better, so it
    must not share an axis with the accuracy metrics. Click a bar (or legend
    entry) to highlight that model across all panels.
    """
    p = _palette(dark)
    pick = alt.selection_point(fields=["model"], bind="legend")
    scale = _model_scale(df["model"].unique().to_list(), p)
    base = alt.Chart().encode(
        x=alt.X("model:N", sort=scale.domain, axis=None),
        y=alt.Y("value:Q", title=None),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.8}).encode(
        color=alt.Color("model:N", scale=scale, title="Model"),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.3)),
        tooltip=["model:N", "metric:N", alt.Tooltip("value:Q", format=".3f")],
    ).add_params(pick)
    labels = base.mark_text(dy=-6, fontSize=10, color=p["ink2"]).encode(
        text=alt.Text("value:Q", format=".2f"),
    )
    return _style(
        alt.layer(bars, labels, data=df)
        .properties(width=alt.Step(26), height=200)
        .facet(column=alt.Column("metric:N", title=None,
                                 sort=df["metric"].unique(maintain_order=True).to_list()))
        .resolve_scale(y="independent")
        .properties(title="Model comparison"),
        p,
    )


def plot_confusion_matrix(
    df: pl.DataFrame, *, title: str = "Confusion matrix", dark: bool = False
) -> alt.LayerChart:
    """7x7 gold-vs-predicted heatmap from one row per message (``message_frame``).

    Clicking a cell selects its (true_label, pred_label) pair. In marimo, filter
    the same DataFrame with ``mo.ui.altair_chart(chart).apply_selection(df)`` to
    list the messages behind the cell.
    """
    p = _palette(dark)
    cell = alt.selection_point(fields=["true_label", "pred_label"])
    base = (
        alt.Chart(df)
        .transform_aggregate(n="count()", groupby=["true_label", "pred_label"])
        .transform_joinaggregate(max_n="max(n)")
        .encode(
            x=alt.X("pred_label:N", title="Predicted", sort=list(LABELS),
                    scale=alt.Scale(domain=list(LABELS)), axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("true_label:N", title="Gold", sort=list(LABELS),
                    scale=alt.Scale(domain=list(LABELS))),
        )
    )
    rects = base.mark_rect(cornerRadius=2, stroke=p["surface"], strokeWidth=2).encode(
        color=alt.Color("n:Q", title="Messages",
                        scale=alt.Scale(range=p["ramp"], domainMin=0),
                        legend=alt.Legend(format="d", tickMinStep=1)),
        opacity=alt.condition(cell, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("true_label:N", title="Gold"),
                 alt.Tooltip("pred_label:N", title="Predicted"),
                 alt.Tooltip("n:Q", title="Messages")],
    ).add_params(cell)
    counts = base.mark_text(fontSize=11).encode(
        text="n:Q",
        color=alt.condition("datum.n / datum.max_n > 0.5",
                            alt.value(p["on_dark"]), alt.value(p["on_light"])),
    )
    return _style(
        alt.layer(rects, counts).properties(width=300, height=300, title=title), p
    )


def plot_tau_sweep(
    df: pl.DataFrame, *, tau: float | None = None, title: str = "τ sweep", dark: bool = False
) -> alt.LayerChart:
    """Macro-F1 and JSD(τ-gated vs gold) across τ (``sweep_table``), with an
    optional rule marking the current τ. Both metrics live on [0, 1]."""
    p = _palette(dark)
    names = {"macro_f1": "Macro-F1", "jsd_gated": "JSD (τ-gated vs gold)"}
    long = (
        df.select("tau", *names)
        .unpivot(index="tau", variable_name="metric", value_name="value")
        .with_columns(pl.col("metric").replace(names))
    )
    color = alt.Color("metric:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=list(names.values()), range=p["series"][:2]))
    hover = alt.selection_point(fields=["tau"], on="pointerover", clear="pointerout",
                                empty=False)
    lo, hi = float(df["tau"].min()), float(df["tau"].max())
    # One full-height hover strip per τ (midpoint to midpoint), so hovering snaps
    # along τ only and the tooltip lists both metrics. A 2-D "nearest" hover would
    # jump between the two lines and read as the data going up and down.
    taus = df["tau"].to_list()
    edges = [lo] + [(a + b) / 2 for a, b in zip(taus, taus[1:])] + [hi]
    strips = df.select("tau", *names).with_columns(
        pl.Series("x0", edges[:-1]), pl.Series("x1", edges[1:])
    )
    ticks = [round(t / 10, 1) for t in range(int(lo * 10 + 0.999), int(hi * 10) + 1)]
    base = alt.Chart(long).encode(
        x=alt.X("tau:Q", title="τ", scale=alt.Scale(zero=False, nice=False),
                axis=alt.Axis(values=ticks, format=".1f")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=color,
    )
    layers = [
        base.mark_line(strokeWidth=2),
        alt.Chart(strips).mark_rule(color=p["axis"]).encode(
            x="tau:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        ),
        base.mark_point(size=64, filled=True).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        ),
        base.mark_text(align="left", dx=6, fontSize=11, color=p["ink2"])
        .encode(text="metric:N", color=alt.value(p["ink2"]))
        .transform_filter(alt.datum.tau == float(df["tau"].max())),
        alt.Chart(strips).mark_rect(opacity=0).encode(
            x="x0:Q", x2="x1:Q",
            tooltip=[alt.Tooltip("tau:Q", format=".2f", title="τ")]
            + [alt.Tooltip(f"{k}:Q", format=".3f", title=v) for k, v in names.items()],
        ).add_params(hover),
    ]
    if tau is not None:
        marker = alt.Chart(pl.DataFrame({"tau": [float(tau)]}))
        layers += [
            marker.mark_rule(strokeDash=[4, 4], color=p["muted"]).encode(x="tau:Q"),
            marker.mark_text(align="left", dx=4, y=8, fontSize=11, color=p["ink2"])
            .encode(x="tau:Q", text=alt.value(f"τ = {tau:.2f}")),
        ]
    return _style(alt.layer(*layers).properties(width=420, height=240, title=title), p)


def plot_distribution_comparison(
    df: pl.DataFrame, *, title: str = "Label distribution vs gold", dark: bool = False
) -> alt.Chart:
    """Gold label mix vs mean predicted probabilities vs τ-gated top-label counts
    (``distribution_frame``); the legend carries each source's JSD to gold."""
    p = _palette(dark)
    sources = df["source"].unique(maintain_order=True).to_list()
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("label:N", title=None, sort=list(LABELS), axis=alt.Axis(labelAngle=0),
                scale=alt.Scale(paddingInner=0.25)),
        xOffset=alt.XOffset("source:N", sort=sources, scale=alt.Scale(paddingInner=0.1)),
        y=alt.Y("share:Q", title="Share of messages", axis=alt.Axis(format="%")),
        color=alt.Color("source:N", sort=sources, title=None,
                        scale=alt.Scale(domain=sources, range=p["series"][:3]),
                        legend=alt.Legend(orient="top", direction="vertical")),
        tooltip=["label:N", "source:N", alt.Tooltip("share:Q", format=".1%")],
    )
    return _style(chart.properties(width=420, height=220, title=title), p)
