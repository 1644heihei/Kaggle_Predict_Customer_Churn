"""
EXP008: クラス不均衡対応 XGBoost
scale_pos_weight パラメータで正・負クラスのバランスを調整
Geminiから: 「クラス不均衡への対応」が重要指摘事項
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import json
import os

print("=" * 70)
print("EXP008: クラス不均衡対応 XGBoost")
print("=" * 70)

# =====================================================
# データ読み込み
# =====================================================
print("\nデータ読み込み中...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# クラス分布の確認
if "Churn" in train.columns:
    churn_counts = train["Churn"].value_counts()
    print(f"\nChurn分布:")
    print(
        f"  否: {churn_counts.get('No', 0):,} ({churn_counts.get('No', 0)/len(train)*100:.1f}%)"
    )
    print(
        f"  是: {churn_counts.get('Yes', 0):,} ({churn_counts.get('Yes', 0)/len(train)*100:.1f}%)"
    )

    # scale_pos_weight の計算
    pos_count = (train["Churn"] == "Yes").sum()
    neg_count = (train["Churn"] == "No").sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\n計算される scale_pos_weight: {scale_pos_weight:.4f}")
else:
    scale_pos_weight = 1.0

# =====================================================
# 前処理とターゲットエンコード
# =====================================================


def preprocess_and_features(df, is_train=True):
    """前処理と特徴エンジニアリング"""
    df = df.copy()

    # ターゲットのエンコード（訓練時）
    if is_train and "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # カテゴリ変数のエンコード
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")
    if "id" in categorical_cols:
        categorical_cols.remove("id")

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # =====================================================
    # 特徴エンジニアリング
    # =====================================================

    # 相互作用項
    df["MonthlyCharges_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges_x_tenure"] = df["TotalCharges"] * df["tenure"]

    # サービスカウント
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["ServiceCount"] = 0
    for col in service_cols:
        if col in df.columns:
            df["ServiceCount"] += (df[col] > 0).astype(int)

    # グループ集計
    if is_train:
        contract_avg = df.groupby("Contract")["MonthlyCharges"].transform("mean")
        internet_avg = df.groupby("InternetService")["tenure"].transform("mean")
        df["Contract_AvgCharge"] = contract_avg
        df["InternetService_AvgTenure"] = internet_avg
    else:
        df["Contract_AvgCharge"] = df["MonthlyCharges"].mean()
        df["InternetService_AvgTenure"] = df["tenure"].mean()

    return df


train_proc = preprocess_and_features(train, is_train=True)
test_proc = preprocess_and_features(test, is_train=False)

print(f"\n処理後のシェイプ:")
print(f"  Train: {train_proc.shape}")
print(f"  Test: {test_proc.shape}")

# =====================================================
# データの準備
# =====================================================

y = train_proc["Churn"].values
X = train_proc.drop(["Churn", "id"], axis=1, errors="ignore")
X_test = test_proc.drop(["id"], axis=1, errors="ignore")

print(f"\n特徴数: {X.shape[1]}")
print(f"訓練データ: {X.shape[0]}, テストデータ: {X_test.shape[0]}")

# =====================================================
# クロスバリデーション + XGBoost（クラス不均衡対応）
# =====================================================

print("\n交差検証開始...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(X_test), 5))

results = {}
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/5 ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    # XGBoostパラメータ（クラス不均衡対応）
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,  # ✓ クラス不均衡対応
        "verbosity": 0,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # OOF予測
    oof_preds[val_idx] = model.predict(dval)

    # テスト予測
    test_preds[:, fold] = model.predict(dtest)

    # スコア
    auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.6f}")

# =====================================================
# 結果の集計
# =====================================================

mean_auc = np.mean(fold_scores)
std_auc = np.std(fold_scores)

print(f"\n{'=' * 70}")
print(f"CV結果:")
print(f"  Mean AUC: {mean_auc:.6f}")
print(f"  Std AUC:  {std_auc:.6f}")
print(f"  Fold Scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"\n  Scale Pos Weight: {scale_pos_weight:.4f}")
print(f"{'=' * 70}")

# テスト予測（5フォルド平均）
test_pred_mean = test_preds.mean(axis=1)

# =====================================================
# 結果の保存
# =====================================================

os.makedirs("EXP/EXP008/outputs/child-exp000", exist_ok=True)

# OOF予測
oof_df = pd.DataFrame(
    {"index": range(len(oof_preds)), "fold": -1, "target": y, "prediction": oof_preds}
)
oof_df.to_csv("EXP/EXP008/outputs/child-exp000/oof_predictions.csv", index=False)

# テスト予測
test_df = pd.DataFrame({"id": test["id"].values, "prediction": test_pred_mean})
test_df.to_csv("EXP/EXP008/outputs/child-exp000/test_predictions.csv", index=False)

# 結果JSON
results_dict = {
    "cv_auc_mean": float(mean_auc),
    "cv_auc_std": float(std_auc),
    "cv_auc_scores": [float(s) for s in fold_scores],
    "scale_pos_weight": float(scale_pos_weight),
    "test_pred_mean": float(test_pred_mean.mean()),
    "test_pred_std": float(test_pred_mean.std()),
}

with open("EXP/EXP008/outputs/child-exp000/results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✓ EXP008 (クラス不均衡対応XGBoost) 完了!")
print(f"  CV AUC: {mean_auc:.6f}")
print(f"  OOF: EXP/EXP008/outputs/child-exp000/oof_predictions.csv")
print(f"  Test: EXP/EXP008/outputs/child-exp000/test_predictions.csv")

if mean_auc >= 0.916:
    print(f"\n🎯 CV AUC 0.916達成! OOFスタッキングに追加推奨")
elif mean_auc >= 0.915:
    print(f"\n✓ CV AUC 0.915達成. OOFスタッキング対象")
else:
    print(f"\n⚠️ CV AUC {mean_auc:.6f}. 多様性確保として活用")
