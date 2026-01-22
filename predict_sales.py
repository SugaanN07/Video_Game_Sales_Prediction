"""
Purpose:
---------
This script trains a machine learning model to predict global video game sales
and allows making predictions for new (hypothetical) games.

It supports two modes:
1) train   -> trains models and saves the best one
2) predict -> loads a trained model and predicts sales for a new game

"""


# 1. Standard library imports
# ===========================
import argparse   # Allows us to pass arguments via the command line
import json       # Used to parse prediction examples passed as JSON
import os         # Used for file and folder handling
import warnings   # Used to hide unnecessary warnings for cleaner output

warnings.filterwarnings("ignore")

# 2. Data science libraries
# =========================

import pandas as pd     # For reading CSV files and manipulating data
import numpy as np      # For numerical operations

# 3. Scikit-learn imports
# =======================

from sklearn.model_selection import train_test_split   # Splits data into train/test
from sklearn.pipeline import Pipeline                  # Ensures preprocessing + model stay together
from sklearn.compose import ColumnTransformer           # Applies different preprocessing to different columns
from sklearn.impute import SimpleImputer                # Handles missing values by providing basic strategies for imputing them
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OneHotEncoder turns one column into many binary columns (making categories yes/no) and StandardScaler rescales numbers so that Mean = 0 & Standard deviation = 1
from sklearn.linear_model import LinearRegression       
from sklearn.ensemble import RandomForestRegressor      
from sklearn.metrics import mean_squared_error, r2_score

# 4. Model persistence
# ====================

import joblib   # Saves and loads trained models

# 5. Command-line arguments
# =========================

# Creates CLI (Command-Line Interface)
parser = argparse.ArgumentParser(description="Train or predict video game sales") 

parser.add_argument(
    "--mode",
    choices=["train", "predict"],
    required=True,
    help="Choose whether to train a model or make a prediction"
)

parser.add_argument("--data", help="Path to training CSV file")
parser.add_argument("--out", help="Path to save trained model")
parser.add_argument("--model", help="Path to trained model for prediction")
parser.add_argument("--example", help="JSON string describing a game to predict")

# Reads everything we typed into the terminal and stores it in args
args = parser.parse_args()

# 6. Feature engineering function
# ===============================

def prepare_features(df):
    """
    This function takes a raw dataframe and:
    - Cleans messy values
    - Creates new meaningful features
    - Ensures consistency between training and prediction
    """

    # Convert User_Score to numeric (some values are 'tbd')
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

    # Create binary features indicating whether review scores exist
    df["Critic_Score_present"] = df["Critic_Score"].notna().astype(int)
    df["User_Score_present"] = df["User_Score"].notna().astype(int)

    # Convert Year_of_Release into Game_Age (more meaningful for models)
    CURRENT_YEAR = 2016
    df["Game_Age"] = CURRENT_YEAR - df["Year_of_Release"]

    # Group smaller publishers into "Other" to reduce noise
    top_publishers = df["Publisher"].value_counts().head(25).index
    df["Publisher_top"] = df["Publisher"].where(
        df["Publisher"].isin(top_publishers),
        "Other"
    )

    return df

# 7. TRAINING MODE
# ================

if args.mode == "train":

    if not args.data or not args.out:
        raise ValueError("Training requires --data and --out arguments")

    # Load dataset
    df = pd.read_csv(args.data)

    # Remove rows without a known target value
    df = df.dropna(subset=["Global_Sales"])

    # Apply feature engineering
    df = prepare_features(df)

    # Define model inputs and target
    FEATURES = [
        "Platform",
        "Genre",
        "Publisher_top",
        "Game_Age",
        "Critic_Score_present",
        "User_Score_present",
        "Rating"
    ]

    TARGET = "Global_Sales"

    X = df[FEATURES]
    y = df[TARGET]

    # Separate categorical and numerical features
    categorical_features = ["Platform", "Genre", "Publisher_top", "Rating"]
    numeric_features = ["Game_Age", "Critic_Score_present", "User_Score_present"]

    """
    Preprocessing for numerical features
    
    Why we use SimpleImputer's median strategy:
    - Numerical columns often have missing values
    - Median is robust to outliers
    - Keeps dataset size intact
    
    Why we use StandardScaler:
    - Centers the data
    - Normalises the scale
    - Is especially important for Linear Regression and other distance-based models
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    """
    Preprocessing for categorical features
    
    Why we use SimpleImputer's most frequent strategy:
    - Some rows have missing publishers or genres
    - We fill missing values with the most common value
    - This avoids dropping rows
    
    Why we use OneHotEncoder:
    - It converts categories into binary columns
    - If a new platform appears later, the model doesn’t crash
    - Unknown categories become all zeros
    """
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Define models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    best_model = None
    best_rmse = float("inf")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate each model
    for name, model in models.items():

        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        # Initially wanted to use below code but can't due to my version of python not supporting squared =False (I'm using python 3.13)
        # Really does not matter in the grand scheme of things but would've made code slightly nicer to look at
        # rmse = mean_squared_error(y_test, predictions, squared=False)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print(f"{name} -> RMSE: {rmse:.3f}, R²: {r2:.3f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline

    # Save the best-performing model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best_model, args.out)

    print(f"\nBest model saved to: {args.out}")

# 8. PREDICTION MODE
# ==================

elif args.mode == "predict":

    if not args.model or not args.example:
        raise ValueError("Prediction requires --model and --example arguments")

    # Load trained model
    model = joblib.load(args.model)

    # Convert JSON string to DataFrame
    example_dict = json.loads(args.example)
    example_df = pd.DataFrame([example_dict])

    # Make prediction
    prediction = model.predict(example_df)[0]

    print(f"Predicted Global Sales: {prediction:.2f} million units")
