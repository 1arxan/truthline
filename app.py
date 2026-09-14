from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load the trained model and vectorizer once, when the server starts
model = joblib.load("text_model.pkl")
vectorizer = joblib.load("text_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def home():
    return "Truthline backend is running!"


@app.route("/check-text", methods=["POST"])
def check_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in the request body"}), 400

    raw_text = data["text"]
    cleaned = clean_text(raw_text)

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label = "real" if prediction == 1 else "fake"
    confidence = round(max(probability) * 100, 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "input_preview": raw_text[:100]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)