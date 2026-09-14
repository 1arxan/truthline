from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
from PIL import Image, ImageChops, ImageEnhance, ExifTags

app = Flask(__name__)

# Load the trained text model and vectorizer once, when the server starts
model = joblib.load("text_model.pkl")
vectorizer = joblib.load("text_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_ela_score(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    resaved_path = "temp_resaved.jpg"
    original.save(resaved_path, "JPEG", quality=quality)
    resaved = Image.open(resaved_path)

    diff = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    os.remove(resaved_path)
    return max_diff


def check_exif(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()

    if exif_data is None:
        return {"has_exif": False, "software": None}

    readable_tags = {}
    for tag_id, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        readable_tags[tag_name] = value

    return {
        "has_exif": True,
        "software": readable_tags.get("Software", None)
    }


@app.route("/")
def home():
    return render_template("index.html")


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


@app.route("/check-image", methods=["POST"])
def check_image():
    if "image" not in request.files:
        return jsonify({"error": "Please upload a file with key 'image'"}), 400

    uploaded_file = request.files["image"]
    temp_path = "temp_upload.jpg"
    uploaded_file.save(temp_path)

    try:
        max_diff = generate_ela_score(temp_path)
        exif_result = check_exif(temp_path)

        suspicion_score = 0
        if max_diff > 50:
            suspicion_score += 50
        if not exif_result["has_exif"]:
            suspicion_score += 30
        if exif_result["software"] and "photoshop" in exif_result["software"].lower():
            suspicion_score += 20

        trust_score = 100 - suspicion_score

        return jsonify({
            "ela_max_difference": max_diff,
            "has_exif": exif_result["has_exif"],
            "software_tag": exif_result["software"],
            "trust_score": trust_score
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)