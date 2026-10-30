# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=1.20",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from huggingface_hub import login, whoami

    return login, mo, whoami


@app.cell
def _(mo):
    mo.md(r"""
    # Log in to Hugging Face

    Press **Log in with browser**. A link and a short code appear under the cell: open the
    link, check that the code matches, and approve. The cell finishes by itself once you
    have approved (the code is valid for 15 minutes).

    The login is saved on this machine (`~/.cache/huggingface/token`), the same place
    `hf auth login` puts it, and renews itself, so scripts and other notebooks can push.
    """)
    return


@app.cell
def _(mo):
    login_button = mo.ui.run_button(label="Log in with browser")
    login_button
    return (login_button,)


@app.cell
def _(login, login_button, mo, whoami):
    mo.stop(not login_button.value, mo.md("Not logged in from this notebook yet."))

    # No token passed and no terminal attached: huggingface_hub uses its browser (device
    # code) flow, printing the link and code to this cell's output while it waits.
    login(skip_if_logged_in=False)
    account = whoami()
    orgs = ", ".join(org["name"] for org in account.get("orgs", [])) or "none"
    mo.md(f"Logged in as **{account['name']}** (organizations: {orgs}).")
    return


if __name__ == "__main__":
    app.run()
