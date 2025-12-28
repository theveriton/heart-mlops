import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]

def load_data():
    df = pd.read_csv(URL, names=COLS)
    df = df.replace("?", pd.NA)
    df = df.dropna()
    df["target"] = (df["target"] > 0).astype(int)
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = load_data()
    df.to_csv("data/raw/heart.csv", index=False)
    print("Saved to data/raw/heart.csv")
