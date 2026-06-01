from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
# DISCLAIMER , some of the comments has been added by Ai as my code didnt have much comments and i told the Ai to explain the code , also remove dead commented code
 
# a web app that guesses if text is positive or negative. it trains a little
# model when the server starts (tf-idf + logistic regression) and then you can
# type anything into the page and it scores it live.
# pip install flask scikit-learn

app = Flask(__name__)

# my tiny training set. labels: 1 = positive, 0 = negative.
TRAIN = [
    ("i love this so much", 1),
    ("this is amazing and great", 1),
    ("absolutely wonderful experience", 1),
    ("best thing ever, so happy", 1),
    ("really enjoyed it, fantastic", 1),
    ("what a beautiful day", 1),
    ("i am so pleased with this", 1),
    ("it works perfectly, thank you", 1),
    ("i hate this", 0),
    ("this is terrible and awful", 0),
    ("worst experience of my life", 0),
    ("so disappointed and sad", 0),
    ("it broke and i am angry", 0),
    ("really bad, do not buy", 0),
    ("this makes me upset", 0),
    ("horrible, waste of money", 0),
]

# build and train the model once at startup
texts = [t for t, _ in TRAIN]
labels = [y for _, y in TRAIN]
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(texts, labels)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"label": "neutral", "score": 0.5})

    # predict_proba gives the probability of positive
    prob = model.predict_proba([text])[0][1]
    label = "positive" if prob >= 0.5 else "negative"
    return jsonify({"label": label, "score": round(float(prob), 3)})


if __name__ == "__main__":
    app.run(debug=True)
