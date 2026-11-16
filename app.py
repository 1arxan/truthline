from flask import Flask, request, jsonify, render_template
import joblib
import re
import os
import requests
from dotenv import load_dotenv
from PIL import Image, ImageChops, ImageEnhance, ExifTags

load_dotenv()

app = Flask(__name__)

model = joblib.load("text_model.pkl")
vectorizer = joblib.load("text_vectorizer.pkl")

FACTCHECK_API_KEY = os.getenv("FACTCHECK_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_MODEL_URL = "https://api-inference.huggingface.co/models/Organika/sdxl-detector"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def search_fact_checks(claim_text):
    if not FACTCHECK_API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": claim_text[:200], "key": FACTCHECK_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=8)
        data = response.json()
        claims = data.get("claims", [])
        results = []
        for claim in claims[:3]:
            for review in claim.get("claimReview", []):
                results.append({
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "rating": review.get("textualRating", "Not rated"),
                    "url": review.get("url", "")
                })
        return results
    except Exception as e:
        print("Fact Check API error:", e)
        return []


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
    return {"has_exif": True, "software": readable_tags.get("Software", None)}


def upload_to_imgbb(image_path):
    if not IMGBB_API_KEY:
        return None
    url = "https://api.imgbb.com/1/upload"
    try:
        with open(image_path, "rb") as f:
            response = requests.post(url, params={"key": IMGBB_API_KEY}, files={"image": f}, timeout=15)
        data = response.json()
        return data.get("data", {}).get("url")
    except Exception as e:
        print("imgbb upload error:", e)
        return None


def reverse_image_search(image_url):
    if not SERPAPI_KEY or not image_url:
        return []
    url = "https://serpapi.com/search"
    params = {"engine": "google_reverse_image", "image_url": image_url, "api_key": SERPAPI_KEY}
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        matches = data.get("image_results", [])
        results = []
        for match in matches[:5]:
            results.append({
                "title": match.get("title", "Untitled"),
                "source": match.get("source", "Unknown"),
                "link": match.get("link", "")
            })
        return results
    except Exception as e:
        print("Reverse image search error:", e)
        return []


def detect_ai_generated(image_path):
    if not HUGGINGFACE_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    try:
        with open(image_path, "rb") as f:
            response = requests.post(HF_MODEL_URL, headers=headers, data=f, timeout=30)
        if response.status_code == 503:
            print("Hugging Face model is loading, try again shortly.")
            return None
        result = response.json()
        if isinstance(result, list):
            scores = {item["label"].lower(): item["score"] for item in result}
            ai_score = scores.get("artificial", scores.get("fake", scores.get("ai", 0)))
            return round(ai_score * 100, 2)
        print("Unexpected Hugging Face response:", result)
        return None
    except Exception as e:
        print("Hugging Face AI detection error:", e)
        return None


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

    word_count = len(raw_text.split())
    reliability = "low" if word_count < 15 else "medium"
    note = (
        "This model only looks at news-writing style. "
        "It cannot verify whether a claim is true."
    )

    fact_checks = search_fact_checks(raw_text)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "input_preview": raw_text[:100],
        "fact_checks": fact_checks,
        "reliability": reliability,
        "word_count": word_count,
        "note": note,
        "fact_check_found": len(fact_checks) > 0
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
        ai_score = detect_ai_generated(temp_path)

        public_url = upload_to_imgbb(temp_path)
        similar_images = reverse_image_search(public_url) if public_url else []

        suspicion_score = 0
        if max_diff > 65:
            suspicion_score += 45
        elif max_diff > 40:
            suspicion_score += 20
        if not exif_result["has_exif"]:
            suspicion_score += 10
        if exif_result["software"] and any(tool in exif_result["software"].lower() for tool in ["photoshop", "gimp", "lightroom"]):
            suspicion_score += 35
        if len(similar_images) > 0:
            suspicion_score += 15
        if ai_score is not None:
            if ai_score > 70:
                suspicion_score += 60
            elif ai_score > 40:
                suspicion_score += 25

        trust_score = max(0, 100 - suspicion_score)

        return jsonify({
            "ela_max_difference": max_diff,
            "has_exif": exif_result["has_exif"],
            "software_tag": exif_result["software"],
            "ai_generated_score": ai_score,
            "trust_score": trust_score,
            "similar_images": similar_images,
            "found_elsewhere": len(similar_images) > 0
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)