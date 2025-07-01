import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load both files
train_df = pd.read_csv('twiiter.csv')
test_df = pd.read_csv('facebook.csv')

# Combine both datasets
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# First split: separate training set (70%) and remaining (30%)
train_data, temp_data = train_test_split(combined_df, train_size=0.7, random_state=42)

# Split remaining 30% into 50% validation and 50% test (i.e., 15% each of total)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save to new CSV files

train_data.to_csv('train.csv', index=False)
val_data.to_csv('validation.csv', index=False)
test_data.to_csv('test.csv', index=False)

print("✅ Files saved under 'data_split/' folder:")
print(f"Train: {len(train_data)} rows")
print(f"Validation: {len(val_data)} rows")
print(f"Test: {len(test_data)} rows")