import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the cleaned dataset after Data Preprocessing
df = pd.read_csv("dataset/cleaned car prices.csv")

# Selecting the features and target variable
features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
X = df[features]
y = df['selling_price']

# Handling missing values
X.fillna(X.median(), inplace=True)

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Trained Successfully!")
print(f"📈 Model Performance:\n- MAE: {mae:.2f}\n- MSE: {mse:.2f}\n- R² Score: {r2:.4f}")

# Save trained model
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model Saved as 'linear_regression_model.pkl'")  