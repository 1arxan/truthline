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