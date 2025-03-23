import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
def load_data(file_path="dataset.csv"):
    df = pd.read_csv(file_path)
    return df


# Data preprocessing
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["description", "interior_color"], errors='ignore')
    df = df.dropna(subset=["price"])  # Remove rows where price is missing

    # Fill missing values
    for col in ["cylinders", "mileage", "doors"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["engine", "fuel", "transmission", "trim", "body", "exterior_color"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical variables
    label_encoders = {}
    for col in ["make", "model", "engine", "fuel", "transmission", "trim", "body", "exterior_color", "drivetrain"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders


# Train model
def train_model(df):
    X = df.drop(columns=["price", "name"], errors='ignore')
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance: MAE={mae}, RMSE={rmse}, R2={r2}")

    # Save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as model.pkl")

    return model


# Load trained model
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# Predict car price
def predict_price(input_data):
    model = load_model()

    # Ensure the input follows the correct feature order
    feature_order = ['make', 'model', 'year', 'engine', 'cylinders', 'fuel', 'mileage',
                     'transmission', 'trim', 'body', 'doors', 'exterior_color', 'drivetrain']

    input_df = pd.DataFrame([input_data])[feature_order]  # Reorder correctly

    predicted_price = model.predict(input_df)[0]
    return round(predicted_price, 2)


if __name__ == "__main__":
    df = load_data()
    df, encoders = preprocess_data(df)
    model = train_model(df)

    # Example prediction
    sample_input = {"make": 15, "model": 137, "year": 2022, "engine": 23, "cylinders": 6.0, "fuel": 4, "mileage": 30000,
                    "transmission": 19, "trim": 148, "body": 6, "doors": 4.0, "exterior_color": 256, "drivetrain": 1}
    predicted_price = predict_price(sample_input)
    print(f"Predicted Price: ${predicted_price}")

