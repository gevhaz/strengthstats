from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    """Render the homepage of the app."""
    return render_template("index.html")
