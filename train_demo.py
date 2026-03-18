#!/usr/bin/env python3
"""
高速デモ版 - サンプリングして訓練
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
print("Loading data...")
train_df = pd.read_csv("data/train.csv", encoding="utf-8")
test_df = pd.read_csv("data/test.csv", encoding="utf-8")

# サンプリング（デモ用）
np.random.seed(42)
sample_idx = np.random.choice(len(train_df), size=50000, replace=False)
train_df_sample = train_df.iloc[sample_idx].reset_index(drop=True)

print(
    f"✓ Full Train: {train_df.shape}, Sample: {train_df_sample.shape}, Test: {test_df.shape}"
)

# == Preprocessing ==
y = (train_df_sample["Churn"] == "Yes").astype(int).values
X = train_df_sample.drop(["Churn", "id"], axis=1, errors="ignore")
X_test = test_df.drop("id", axis=1, errors="ignore")

# Encode categoricals
le_dict = {}
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
    X_test[col] = le.transform(X_test[col].astype(str))

# Fill NaN
X = X.fillna(X.mean(numeric_only=True))
mean_vals = train_df_sample[X.columns].mean(numeric_only=True)
X_test = X_test.fillna(mean_vals)

print(f"✓ Data processed: X={X.shape}, y dist={np.bincount(y).tolist()}")

# == Training (3-fold for speed) ==
oof_preds = []
test_preds = []
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\nFold {fold}...", end=" ", flush=True)

    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Simplified LightGBM params
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 50,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": 4,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(100),
        ],
    )

    # Predict with sigmoid
    val_raw = model.predict(X_val)
    val_prob = expit(val_raw)

    auc = roc_auc_score(y_val, val_prob)
    logloss = log_loss(y_val, val_prob)
    print(f"AUC={auc:.4f}, Logloss={logloss:.4f}")

    for idx, prob in zip(val_idx, val_prob):
        oof_preds.append(
            {"index": idx, "fold": fold, "target": y[idx], "prediction": prob}
        )

    # Test predictions
    test_raw = model.predict(X_test)
    test_prob = expit(test_raw)
    test_preds.append(test_prob)

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

# Test predictions (average)
test_pred_avg = np.mean(test_preds, axis=0)
test_df_sub = pd.DataFrame({"prediction": test_pred_avg})
test_df_sub.to_csv(output_dir / "test_predictions.csv", index=False)

results = {
    "cv_auc": float(cv_auc),
    "cv_logloss": float(cv_logloss),
    "n_folds": 3,
    "sample_size": len(train_df_sample),
}
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_dir}")
print(f"✓ OOF predictions: {(output_dir / 'oof_predictions.csv')}")
print(f"✓ Test predictions: {(output_dir / 'test_predictions.csv')}")
