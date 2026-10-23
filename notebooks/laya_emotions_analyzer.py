# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "laya @ git+https://github.com/NandhaKishorM/laya.git",
#     "huggingface_hub",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import laya
    import marimo as mo
    from huggingface_hub import snapshot_download

    return Path, json, laya, mo, snapshot_download


@app.cell
def _(mo):
    mo.md(r"""
    # Laya emotions analyzer

    Batch-predicts the primary emotion of each message with the fine-tuned
    [`LaunchSpark/laya-emotion-tracker`](https://huggingface.co/LaunchSpark/laya-emotion-tracker)
    model.
    """)
    return


@app.cell
def _(laya, mo, snapshot_download):
    # Same checkpoint as the bot (src/vibeshift/emotions.py): the 7-label model.
    with mo.status.spinner("Loading LaunchSpark/laya-emotion-tracker..."):
        model_dir = snapshot_download("LaunchSpark/laya-emotion-tracker", revision="v2-7-labels")
        agent = laya.Agent(model_dir)
    return agent, model_dir


@app.cell
def _(Path, json, model_dir):
    # The checkpoint ships the schema it was trained on, so the labels always match it.
    questions = json.loads((Path(model_dir) / "questions.json").read_text())
    return (questions,)


@app.cell
def _(agent, laya, mo, questions):
    def batch_predict_emotions(input_dict):
        """
        Takes a dictionary of {key: text_message} and returns a new dictionary
        with the same keys containing the predicted emotion for each message.
        """
        output_dict = {}
        for key, message in mo.status.progress_bar(
            input_dict.items(), title="Predicting emotions", total=len(input_dict)
        ):
            result = laya.decide(agent, message, questions=questions)
            output_dict[key] = result["emotion_tracker"]["choice"]
        return output_dict

    return (batch_predict_emotions,)


@app.cell
def _(mo):
    # Matthew 6, one verse per line.
    verses = [
        "Take heed that you do not do your charitable deeds before men, to be seen by them. Otherwise you have no reward from your Father in heaven.",
        "Therefore, when you do a charitable deed, do not sound a trumpet before you as the hypocrites do in the synagogues and in the streets, that they may have glory from men. Assuredly, I say to you, they have their reward.",
        "But when you do a charitable deed, do not let your left hand know what your right hand is doing,",
        "that your charitable deed may be in secret; and your Father who sees in secret will Himself reward you openly.",
        "And when you pray, you shall not be like the hypocrites. For they love to pray standing in the synagogues and on the corners of the streets, that they may be seen by men. Assuredly, I say to you, they have their reward.",
        "But you, when you pray, go into your room, and when you have shut your door, pray to your Father who is in the secret place; and your Father who sees in secret will reward you openly.",
        "And when you pray, do not use vain repetitions as the heathen do. For they think that they will be heard for their many words.",
        "Therefore do not be like them. For your Father knows the things you have need of before you ask Him.",
        "In this manner, therefore, pray: Our Father in heaven, Hallowed be Your name.",
        "Your kingdom come. Your will be done On earth as it is in heaven.",
        "Give us this day our daily bread.",
        "And forgive us our debts, As we forgive our debtors.",
        "And do not lead us into temptation, But deliver us from the evil one. For Yours is the kingdom and the power and the glory forever. Amen.",
        "For if you forgive men their trespasses, your heavenly Father will also forgive you.",
        "But if you do not forgive men their trespasses, neither will your Father forgive your trespasses.",
        "Moreover, when you fast, do not be like the hypocrites, with a sad countenance. For they disfigure their faces that they may appear to men to be fasting. Assuredly, I say to you, they have their reward.",
        "But you, when you fast, anoint your head and wash your face,",
        "so that you do not appear to men to be fasting, but to your Father who is in the secret place; and your Father who sees in secret will reward you openly.",
        "Do not lay up for yourselves treasures on earth, where moth and rust destroy and where thieves break in and steal;",
        "but lay up for yourselves treasures in heaven, where neither moth nor rust destroys and where thieves do not break in and steal.",
        "For where your treasure is, there your heart will be also.",
        "The lamp of the body is the eye. If therefore your eye is good, your whole body will be full of light.",
        "But if your eye is bad, your whole body will be full of darkness. If therefore the light that is in you is darkness, how great is that darkness!",
        "No one can serve two masters; for either he will hate the one and love the other, or else he will be loyal to the one and despise the other. You cannot serve God and mammon.",
        "Therefore I say to you, do not worry about your life, what you will eat or what you will drink; nor about your body, what you will put on. Is not life more than food and the body more than clothing?",
        "Look at the birds of the air, for they neither sow nor reap nor gather into barns; yet your heavenly Father feeds them. Are you not of more value than they?",
        "Which of you by worrying can add one cubit to his stature?",
        "So why do you worry about clothing? Consider the lilies of the field, how they grow: they neither toil nor spin;",
        "and yet I say to you that even Solomon in all his glory was not arrayed like one of these.",
        "Now if God so clothes the grass of the field, which today is, and tomorrow is thrown into the oven, will He not much more clothe you, O you of little faith?",
        "Therefore do not worry, saying, ‘What shall we eat?’ or ‘What shall we drink?’ or ‘What shall we wear?’",
        "For after all these things the Gentiles seek. For your heavenly Father knows that you need all these things.",
        "But seek first the kingdom of God and His righteousness, and all these things shall be added to you.",
        "Therefore do not worry about tomorrow, for tomorrow will worry about its own things. Sufficient for the day is its own trouble.",
    ]
    messages = mo.ui.text_area(
        value="\n".join(verses),
        label="One message per line",
        rows=12,
        full_width=True,
    )
    messages
    return (messages,)


@app.cell
def _(batch_predict_emotions, messages):
    lines = [line for line in messages.value.splitlines() if line.strip()]
    test_data = dict(enumerate(lines, 1))
    predictions = batch_predict_emotions(test_data)
    return predictions, test_data


@app.cell
def _(mo, predictions, test_data):
    mo.ui.table(
        [
            {"#": key, "message": message, "emotion": predictions[key]}
            for key, message in test_data.items()
        ],
        selection=None,
    )
    return


if __name__ == "__main__":
    app.run()
