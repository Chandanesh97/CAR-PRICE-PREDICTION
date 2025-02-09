import pandas as pd

# Loading the dataset
df = pd.read_csv("dataset/Car details.csv")

# Converting numerical text fields to proper float
df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
df['max_power'] = df['max_power'].str.replace(' bhp', '', regex=False)
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

# Droping the 'torque' column because not required for the car price prediction
df = df.drop(columns=['torque'])

# Handling missing values by filling with median values
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert categorical columns to numerical using One-Hot Encoding
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)

# Save cleaned dataset
df.to_csv("dataset/cleaned car prices.csv", index=False)
print("✅ Data Preprocessing Completed. Cleaned dataset saved!")