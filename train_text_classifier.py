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