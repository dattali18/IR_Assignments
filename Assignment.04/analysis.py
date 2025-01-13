import pandas as pd

# Read the CSV file
df = pd.read_csv("extracted_sentences.csv")

# Count occurrences of each type
class_counts = df["type"].value_counts()

# Print the results
print("\nSentiment Class Distribution:")
print("-" * 30)
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Optional: Calculate percentages
total = len(df)
print("\nPercentages:")
print("-" * 30)
for class_name, count in class_counts.items():
    percentage = (count / total) * 100
    print(f"{class_name}: {percentage:.2f}%")
