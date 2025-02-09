import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading cleaned dataset after the data preprocessing
df = pd.read_csv("dataset/cleaned car prices.csv")

# Histogram of Selling Prices
plt.figure(figsize=(10,5))
sns.histplot(df['selling_price'], bins=50, kde=True)
plt.title("Distribution of Selling Prices")
plt.xlabel("Price (INR)")
plt.ylabel("Count")
plt.show()

# Selecting Only Numeric Columns for Correlation heatmap
numeric_df = df.select_dtypes(include=['number'])

# Plot the Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter Plot: Year vs Selling Price
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['year'], y=df['selling_price'])
plt.title("Car Year vs Selling Price")
plt.xlabel("Year")
plt.ylabel("Price (INR)")
plt.show()

print("✅ EDA Completed. Check the visualizations.")