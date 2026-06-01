# Sentiment Analysis (Web)

A web app that guesses if text is positive or negative. It trains a small
machine-learning model (TF-IDF + logistic regression) when the server starts,
then you type into the page and it scores the sentiment live as you type.

This one mixes AI + backend + frontend: the model runs on a Flask server and
the page calls it with fetch.

## features

- trains a real ML model at startup (scikit-learn)
- live prediction as you type (debounced fetch)
- shows positive/negative + a confidence score
- result colored green/red

## run

```bash
pip install flask scikit-learn
python app.py
```

open http://127.0.0.1:5000

tags: python, ai, ml, flask, backend, nlp

wanted to actually deploy a model behind a web page, not just print to terminal.
