print("SCRIPT STARTED")

import pandas as pd

# Load the two CSV files into pandas DataFrames (think: two spreadsheets in memory)
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

# Peek at what's inside each one
print("FAKE sample:")
print(fake_df.head())

print("\nREAL sample:")
print(real_df.head())

print("\nFake shape:", fake_df.shape)
print("Real shape:", real_df.shape)
print("\nColumns:", fake_df.columns.tolist())
# Label each dataset: 0 = fake, 1 = real
fake_df["label"] = 0
real_df["label"] = 1

# Combine into a single DataFrame
combined_df = pd.concat([fake_df, real_df], ignore_index=True)

# Shuffle the rows so fake/real aren't in two separate blocks
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined shape:", combined_df.shape)
print(combined_df["label"].value_counts())
import re

def clean_text(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r"http\S+|www\S+", "", text)    # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)          # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()      # collapse extra whitespace
    return text

# Apply cleaning to the text column
combined_df["clean_text"] = combined_df["text"].apply(clean_text)

print("\nBefore cleaning:")
print(combined_df["text"].iloc[0][:200])
print("\nAfter cleaning:")
print(combined_df["clean_text"].iloc[0][:200])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split data: 80% to train the model, 20% held back to test it honestly
X_train, X_test, y_train, y_test = train_test_split(
    combined_df["clean_text"],
    combined_df["label"],
    test_size=0.2,
    random_state=42
)

# Convert text into numbers TF-IDF can understand
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nTraining samples:", X_train_vec.shape)
print("Testing samples:", X_test_vec.shape)