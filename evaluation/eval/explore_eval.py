import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from evaluation.common import io
    from evaluation.eval import charts, frames
    from evaluation.eval.evaluate import evaluate_model

    return charts, evaluate_model, frames, io, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # VibeShift evaluation explorer

    Drag **τ** to re-gate every model's stored probabilities live, pick a model,
    and click a confusion-matrix cell to list the Discord messages behind it.
    To add a model, drop its `{model}.jsonl` into `evaluation/predictions/` and
    re-run the notebook.
    """)
    return


@app.cell
def _(io, mo):
    models = io.list_prediction_models()
    mo.stop(
        not models,
        mo.md("No prediction files yet — run a model first, e.g. `pixi run eval-nrc`."),
    )
    messages = io.load_messages()
    consensus = io.load_consensus()
    preds = {name: io.load_predictions(name) for name in models}
    mo.callout(
        mo.md("Running on the **10-row sample fixture**: the real 300-message dataset "
              "hasn't been exported yet."),
        kind="warn",
    ) if io.using_sample() else None
    return consensus, messages, models, preds


@app.cell
def _(mo, models):
    tau_slider = mo.ui.slider(
        0.10, 0.85, step=0.05, value=0.35, label="Neutral cutoff (τ)", show_value=True
    )
    model_picker = mo.ui.dropdown(
        models, value="laya" if "laya" in models else models[0], label="Model"
    )
    split_picker = mo.ui.dropdown(["test", "dev", "all"], value="test", label="Split")
    mo.hstack([tau_slider, model_picker, split_picker], justify="start", gap=2)
    return model_picker, split_picker, tau_slider


@app.cell
def _(consensus, frames, messages, mo, split_picker):
    gold = frames.filter_split(messages, consensus, split_picker.value)
    dark = mo.app_meta().theme == "dark"
    return dark, gold


@app.cell
def _(charts, dark, evaluate_model, frames, gold, mo, pl, preds, tau_slider):
    results = {name: evaluate_model(rows, gold, tau_slider.value) for name, rows in preds.items()}
    metrics_table = pl.DataFrame(
        [{"model": name, **{k: round(v, 3) for k, v in r.items()}} for name, r in results.items()]
    )
    mo.vstack([
        charts.plot_model_comparison(frames.metrics_frame(results), dark=dark),
        mo.ui.table(metrics_table, selection=None),
    ])
    return


@app.cell
def _(charts, dark, frames, gold, messages, mo, model_picker, preds, tau_slider):
    model_rows = preds[model_picker.value]
    message_df = frames.message_frame(model_rows, gold, messages, tau_slider.value)
    cm_chart = mo.ui.altair_chart(
        charts.plot_confusion_matrix(
            message_df, title=f"{model_picker.value} — τ = {tau_slider.value:.2f}", dark=dark
        ),
        legend_selection=False,
    )
    distribution = charts.plot_distribution_comparison(
        frames.distribution_frame(model_rows, gold, tau_slider.value), dark=dark
    )
    mo.hstack([cm_chart, distribution], justify="start", wrap=True)
    return cm_chart, message_df, model_rows


@app.cell
def _(cm_chart, message_df, mo, pl):
    # A cell click selects a (true_label, pred_label) pair; with nothing selected,
    # default to every misclassified message.
    has_selection = any(
        values
        for fields in cm_chart.selections.values()
        for key, values in fields.items()
        if key in ("true_label", "pred_label")
    )
    shown = (
        cm_chart.apply_selection(message_df)
        if has_selection
        else message_df.filter(~pl.col("correct"))
    )
    heading = (
        "Messages in the selected cell"
        if has_selection
        else "All misclassified messages (click a confusion-matrix cell to filter)"
    )
    mo.vstack([
        mo.md(f"**{heading}** · {shown.height} message(s)"),
        mo.ui.table(shown.drop("correct"), selection=None, page_size=10),
    ])
    return


@app.cell
def _(charts, dark, frames, gold, model_picker, model_rows, tau_slider):
    charts.plot_tau_sweep(
        frames.sweep_table(model_rows, gold),
        tau=tau_slider.value,
        title=f"{model_picker.value} — τ sweep",
        dark=dark,
    )
    return


if __name__ == "__main__":
    app.run()
