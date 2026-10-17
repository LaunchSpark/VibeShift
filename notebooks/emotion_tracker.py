import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from vibeshift.emotions import batch_predict_emotions, emotion_labels, load_model

    return batch_predict_emotions, emotion_labels, load_model, mo


@app.cell
def _(emotion_labels, load_model, mo):
    with mo.status.spinner("Loading LaunchSpark/laya-emotion-tracker..."):
        load_model()
    mo.md("**Labels:** " + ", ".join(emotion_labels()))
    return


@app.cell
def _(mo):
    messages = mo.ui.text_area(
        value="I can't believe we actually won!\nugh, great, another meeting\nI'm so sorry I forgot",
        label="One message per line",
        full_width=True,
    )
    messages
    return (messages,)


@app.cell
def _(batch_predict_emotions, messages):
    test_data = {
        i: line for i, line in enumerate(messages.value.splitlines()) if line.strip()
    }
    predictions = batch_predict_emotions(test_data)
    return predictions, test_data


@app.cell
def _(mo, predictions, test_data):
    mo.ui.table(
        [{"message": test_data[k], "emotion": predictions[k]} for k in test_data],
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
