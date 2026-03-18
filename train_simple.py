#!/usr/bin/env python3
"""
Baseline訓練スクリプト - シンプル版
Usage: python train_simple.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from scipy.special import expit
import json

# == Data Loading ==
train_df = pd.read_csv("data/train.csv", encoding="utf-8")
test_df = pd.read_csv("data/test.csv", encoding="utf-8")

print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

# == Preprocessing ==
y = (train_df["Churn"] == "Yes").astype(int).values
X = train_df.drop(["Churn", "id"], axis=1, errors="ignore")
X_test = test_df.drop("id", axis=1, errors="ignore")

# Encode categoricals
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Fill NaN
X = X.fillna(X.mean(numeric_only=True))
X_test = X_test.fillna(X.mean(numeric_only=True))

print(f"✓ Data processed: X={X.shape}, y dist={np.bincount(y).tolist()}")

# == Training ==
oof_preds = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold}...", end="")

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # LightGBM params
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 100,
        "learning_rate": 0.05,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    # Predict with sigmoid
    val_raw = model.predict(X_val)
    val_prob = expit(val_raw)

    auc = roc_auc_score(y_val, val_prob)
    logloss = log_loss(y_val, val_prob)
    print(f" AUC={auc:.4f}, Logloss={logloss:.4f}")

    for idx, prob in zip(val_idx, val_prob):
        oof_preds.append(
            {"index": idx, "fold": fold, "target": y[idx], "prediction": prob}
        )

# == Evaluate ==
oof_df = pd.DataFrame(oof_preds).sort_values("index").reset_index(drop=True)
cv_auc = roc_auc_score(oof_df["target"], oof_df["prediction"])
cv_logloss = log_loss(oof_df["target"], oof_df["prediction"])

print(f"\n{'='*50}")
print(f"CV AUC: {cv_auc:.4f}")
print(f"CV Logloss: {cv_logloss:.4f}")
print(f"{'='*50}")

# == Save ==
output_dir = Path("EXP/EXP001/outputs/child-exp000")
output_dir.mkdir(parents=True, exist_ok=True)

oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

results = {"cv_auc": float(cv_auc), "cv_logloss": float(cv_logloss), "n_folds": 5}
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")
