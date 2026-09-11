import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("SCRIPT STARTED")

# ---- Load data ----
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# ---- Label and combine ----
fake_df["label"] = 0
real_df["label"] = 1

combined_df = pd.concat([fake_df, real_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined shape:", combined_df.shape)
print(combined_df["label"].value_counts())

# ---- Clean text ----
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

combined_df["clean_text"] = combined_df["text"].apply(clean_text)

# ---- Split train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    combined_df["clean_text"],
    combined_df["label"],
    test_size=0.2,
    random_state=42
)

# ---- TF-IDF vectorize ----
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining samples:", X_train_vec.shape)
print("Testing samples:", X_test_vec.shape)

# ---- Train model ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---- Evaluate ----
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ---- Save model + vectorizer for Flask to use later ----
joblib.dump(model, "text_model.pkl")
joblib.dump(vectorizer, "text_vectorizer.pkl")
print("\nSaved text_model.pkl and text_vectorizer.pkl")