import os
import sys
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression

DATA = Path("data/processed/match_training.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "home_points_last_5", "home_gf_last_5", "home_ga_last_5",
    "away_points_last_5", "away_gf_last_5", "away_ga_last_5",
]

def main() -> int:
    if not DATA.exists():
        print("Missing training dataset:", DATA)
        return 1
    
    df = pd.read_parquet(DATA).dropna(subset=["label"]).copy()

    X = df[FEATURE_COLS]
    y = df["label"].astype(int)

    # time-ish split: last 20% as test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("soccer_outcome")

    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000, multi_class="multinomial")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        pred = proba.argmax(axis=1)

        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, proba, labels=[0,1,2])

        mlflow.log_param("model", "logreg_multinomial")
        mlflow.log_param("features", ",".join(FEATURE_COLS))
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("log_loss", float(ll))

        mlflow.sklearn.log_model(model, artifact_path="model")

        out_path = MODEL_DIR / "logreg_model.joblib"
        
        joblib.dump(model, out_path)
        mlflow.log_artifact(str(out_path))

        print(f"Trained. accuracy={acc:.3f} log_loss={ll:.3f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())