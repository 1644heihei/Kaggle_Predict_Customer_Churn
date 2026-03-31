"""
OOFベースのスタッキング
各モデルのOOF予測を使用してメタモデルを学習
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json

# OOF予測の読み込み
exp002_oof = pd.read_csv("EXP/EXP002/outputs/child-exp000/oof_predictions.csv")
exp003_oof = pd.read_csv("EXP/EXP003/outputs/child-exp000/oof_predictions.csv")
exp004_oof = pd.read_csv("EXP/EXP004/outputs/child-exp000/oof_predictions.csv")

# テスト予測の読み込み
exp002_test = pd.read_csv("EXP/EXP002/outputs/child-exp000/test_predictions.csv")
exp003_test = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004_test = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")

print("=" * 50)
print("OOFベーススタッキング")
print("=" * 50)

# OOFファイルの構造を確認
print(f"\nOOF形式:")
print(f"  EXP002: {exp002_oof.columns.tolist()}")
print(f"  EXP003: {exp003_oof.columns.tolist()}")
print(f"  EXP004: {exp004_oof.columns.tolist()}")

print(f"\nOOF形状:")
print(f"  EXP002: {exp002_oof.shape}")
print(f"  EXP003: {exp003_oof.shape}")
print(f"  EXP004: {exp004_oof.shape}")

# OOF結合 (indexベース - すべて同じ順序と想定)
oof_features = pd.DataFrame()
oof_features["pred_002"] = exp002_oof.iloc[:, 0].values  # 最初のカラムを予測とみなす
oof_features["pred_003"] = exp003_oof.iloc[:, 0].values
oof_features["pred_004"] = exp004_oof.iloc[:, 0].values

# トレーニングターゲットを取得
data_train = pd.read_csv("data/train.csv")
y_train = data_train["Churn"].values

print(f"\nトレーニングデータ:")
print(f"  y_train shape: {y_train.shape}")
print(f"  oof_features shape: {oof_features.shape}")

# メタモデルの学習 (LogisticRegression)
print(f"\n--- メタモデル学習中 ---")
meta_model = LogisticRegression(random_state=42, max_iter=1000)

# 正則化パラメータのスイープ（ハイパーパラメータチューニング）
best_c = 1.0
best_score = -np.inf

for c_val in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    meta_model_tmp = LogisticRegression(
        C=c_val, random_state=42, max_iter=1000, solver="lbfgs"
    )
    meta_model_tmp.fit(oof_features, y_train)

    from sklearn.metrics import roc_auc_score

    train_auc = roc_auc_score(y_train, meta_model_tmp.predict_proba(oof_features)[:, 1])
    print(f"  C={c_val}: train AUC={train_auc:.6f}")

    if train_auc > best_score:
        best_score = train_auc
        best_c = c_val

# 最適Cで再学習
meta_model = LogisticRegression(
    C=best_c, random_state=42, max_iter=1000, solver="lbfgs"
)
meta_model.fit(oof_features, y_train)

print(f"\n最適メタモデル (C={best_c}):")
print(f"  係数: {meta_model.coef_[0]}")
print(f"  切片: {meta_model.intercept_}")

# テストセット用メタ特徴
test_features = pd.DataFrame()
test_features["pred_002"] = exp002_test.iloc[:, 0].values
test_features["pred_003"] = exp003_test.iloc[:, 0].values
test_features["pred_004"] = exp004_test.iloc[:, 0].values

print(f"\nテスト予測:")
print(f"  test_features shape: {test_features.shape}")

# メタモデルでテスト予測
y_pred_stacking = meta_model.predict_proba(test_features)[:, 1]

print(f"\nスタッキング結果統計:")
print(f"  Mean: {y_pred_stacking.mean():.6f}")
print(f"  Std:  {y_pred_stacking.std():.6f}")
print(f"  Min:  {y_pred_stacking.min():.6f}")
print(f"  Max:  {y_pred_stacking.max():.6f}")

# サブミッション作成
submission_stacking = pd.DataFrame(
    {"id": exp004_test["id"].values, "Churn": y_pred_stacking}
)

submission_stacking.to_csv("submission_stacking_exp234.csv", index=False)
print(f"\n✓ submission_stacking_exp234.csv を保存")

# さらに高度なメタモデル: LightGBM or XGBoost
print(f"\n" + "=" * 50)
print("XGBoost メタモデル")
print("=" * 50)

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    dtrain = xgb.DMatrix(oof_features, label=y_train)

    params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "verbosity": 0,
    }

    xgb_meta = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)

    # テスト予測
    dtest = xgb.DMatrix(test_features)
    y_pred_xgb_meta = xgb_meta.predict(dtest)

    print(f"\nXGBoost メタモデル結果統計:")
    print(f"  Mean: {y_pred_xgb_meta.mean():.6f}")
    print(f"  Std:  {y_pred_xgb_meta.std():.6f}")
    print(f"  Min:  {y_pred_xgb_meta.min():.6f}")
    print(f"  Max:  {y_pred_xgb_meta.max():.6f}")

    submission_xgb_meta = pd.DataFrame(
        {"id": exp004_test["id"].values, "Churn": y_pred_xgb_meta}
    )

    submission_xgb_meta.to_csv("submission_stacking_xgb_meta.csv", index=False)
    print(f"✓ submission_stacking_xgb_meta.csv を保存")

except ImportError:
    print("XGBoost not available for meta-model")
