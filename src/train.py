import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/raw/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes("object").columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

models = {
    "log_reg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier()
}

best_score = 0
best_model = None

for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("model", m)])
    score = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc").mean()
    print(name, score)
    
    if score > best_score:
        best_score = score
        best_model = pipe

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Saved best model with ROC-AUC:", best_score)
