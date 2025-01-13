import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv("extracted_sentences.csv")

# Create a mapping dictionary for labels
label_mapping = {
    "neutral": 0,
    "pro-israeli": 1,
    "pro-palestinian": 2,
    "anti-israeli": 3,
    "anti-palestinian": 4,
}

# Add numerical labels
df["label"] = df["type"].map(label_mapping)

# Separate neutral and non-neutral samples
neutral_samples = df[df["type"] == "neutral"]
non_neutral_samples = df[df["type"] != "neutral"]

# Randomly sample from neutral class (let's take 1500 samples)
neutral_balanced = neutral_samples.sample(n=1500, random_state=42)

# Combine balanced neutral samples with non-neutral samples
balanced_df = pd.concat([neutral_balanced, non_neutral_samples])

# Shuffle the final dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset
balanced_df.to_csv("sentences.csv", index=False)

# Print distribution of classes in the balanced dataset
print("\nBalanced Dataset Distribution:")
print("-" * 30)
class_counts = balanced_df["type"].value_counts()
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

print("\nPercentages:")
print("-" * 30)
total = len(balanced_df)
for class_name, count in class_counts.items():
    percentage = (count / total) * 100
    print(f"{class_name}: {percentage:.2f}%")
