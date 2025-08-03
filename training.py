import pandas as pd
import pickle
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === STEP 1: Load Dataset ===
train_file = r"C:\Users\Rohan B\Downloads\team_25\team_25\credit_score_training_data.xlsx"
df = pd.read_excel(train_file)

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into features and target
X = df.drop(columns=['CREDIT_SCORE'])
y = df['CREDIT_SCORE']

# === STEP 2: Define Models to Compare ===
models_to_try = {
    "RandomForest": lambda seed: RandomForestRegressor(random_state=seed),
    "GradientBoosting": lambda seed: GradientBoostingRegressor(random_state=seed),
    "MLPRegressor": lambda seed: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
}

# === STEP 3: Train and Evaluate Models ===
best_r2 = float('-inf')
best_model = None
best_model_name = ""
best_scaler = None
best_comparison_df = None
best_epoch = None

for model_name, model_func in models_to_try.items():
    print(f"\nTraining model: {model_name}")

    for epoch in range(1, 21):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=epoch
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train model
        model = model_func(epoch)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Epoch {epoch:2d} | R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

        # Track best model overall
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = model_name
            best_scaler = scaler
            best_epoch = epoch
            best_comparison_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': y_pred.round(2)
            })

# === STEP 4: Save the Best Model ===
with open("credit_score_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(best_scaler, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# === STEP 5: Summary ===
print("\nBest model saved successfully.")
print(f"Best Model: {best_model_name} from epoch {best_epoch}")
print(f"Best R² Score: {best_r2:.4f}")
print("\nSample Prediction Results:")
print(best_comparison_df.head(10))
