import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def main():
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv("data/raw/heart.csv")

    X = df.drop("target", axis=1)
    y = df["target"]

    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes("object").columns.tolist()

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat)
    ])

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_score = 0
    best_name = None
    best_model = None

    # Compare models using cross-validated metrics
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("model", m)])
        scores = cross_validate(pipe, X, y, cv=5,
                                scoring=["accuracy", "precision", "recall", "roc_auc"],
                                return_train_score=False)
        mean_roc = scores["test_roc_auc"].mean()
        print(f"{name}: ROC-AUC={mean_roc:.4f}")

        if mean_roc > best_score:
            best_score = mean_roc
            best_model = pipe
            best_name = name

    # Fit best model on full data
    best_model.fit(X, y)

    # Save model locally and log to MLflow
    model_path = "model.pkl"
    joblib.dump(best_model, model_path)
    print("Saved model to", model_path)

    # MLflow logging (local)
    try:
        mlflow.set_experiment("heart-disease-experiments")
        with mlflow.start_run(run_name=f"train_{best_name}"):
            mlflow.log_param("model", best_name)
            mlflow.log_metric("roc_auc", float(best_score))
            mlflow.sklearn.log_model(best_model, "model")
    except Exception as e:
        print("MLflow logging failed:", e)

    print(f"Best model: {best_name}, ROC-AUC={best_score:.4f}")


if __name__ == "__main__":
    main()
