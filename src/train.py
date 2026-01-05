import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
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

    # Compatible OneHotEncoder for different sklearn versions
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", encoder, cat)
    ])

    models = {
        "rf": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 10, 20]}
        }
    }

    best_score = 0
    best_name = None
    best_model = None

    # Tune and compare models
    for name, config in models.items():
        pipe = Pipeline([("pre", pre), ("model", config["model"])])
        grid = GridSearchCV(pipe, config["params"], cv=5, scoring="roc_auc", n_jobs=-1)
        grid.fit(X, y)
        
        mean_roc = grid.best_score_
        print(f"{name}: Best ROC-AUC={mean_roc:.4f}, Params={grid.best_params_}")

        if mean_roc > best_score:
            best_score = mean_roc
            best_model = grid.best_estimator_
            best_name = name

    # Fit best model on full data
    best_model.fit(X, y)

    # Save model locally and log to MLflow
    model_path = "model.pkl"
    joblib.dump(best_model, model_path)
    print("Saved model to", model_path)

    # MLflow logging
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
