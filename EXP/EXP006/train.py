"""
EXP006: 拡張特徴エンジニアリング + XGBoost
- ログ変換
- 多項式特徴
- 正規化
- より多くの相互作用項
- バイナリ特徴の組み合わせ
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import json
import os

# =====================================================
# データ読み込み
# =====================================================
print("データ読み込み中...")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# =====================================================
# 前処理（EXP003/004と同じ）
# =====================================================


def load_and_preprocess(df, is_train=True):
    """基本的な前処理"""
    df = df.copy()

    # ターゲットのエンコード（訓練時のみ）
    if is_train and "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # カテゴリ変数の特定
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Churn" in categorical_cols:
        categorical_cols.remove("Churn")
    if "id" in categorical_cols:
        categorical_cols.remove("id")

    # カテゴリ変数をエンコード
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, categorical_cols


# データの前処理
train_prep, cat_cols = load_and_preprocess(train, is_train=True)
test_prep, _ = load_and_preprocess(test, is_train=False)

# =====================================================
# 拡張特徴エンジニアリング
# =====================================================


def create_advanced_features(df, is_train=False):
    """拡張特徴エンジニアリング"""
    df = df.copy()

    # 1. ログ変換（連続特徴）
    # TotalChargesにログ変換
    df["TotalCharges_log"] = np.log1p(df["TotalCharges"])
    df["MonthlyCharges_log"] = np.log1p(df["MonthlyCharges"])

    # 2. 多項式特徴
    df["tenure_squared"] = df["tenure"] ** 2
    df["tenure_cubed"] = df["tenure"] ** 3
    df["MonthlyCharges_squared"] = df["MonthlyCharges"] ** 2

    # 3. 相互作用項（前回と同じ + 追加）
    df["MonthlyCharges_x_tenure"] = df["MonthlyCharges"] * df["tenure"]
    df["TotalCharges_x_tenure"] = df["TotalCharges"] * df["tenure"]
    df["MonthlyCharges_x_log_tenure"] = df["MonthlyCharges"] * np.log1p(df["tenure"])

    # 4. 比率特徴
    df["Charges_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    df["Avg_monthly_from_total"] = df["TotalCharges"] / (df["tenure"] + 1)

    # 5. サービスカウント（元のEXPと同じ）
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
    # これらはすでにエンコードされている
    df["ServiceCount"] = 0
    for col in service_cols:
        if col in df.columns:
            df["ServiceCount"] += (df[col] > 0).astype(int)

    # 6. グループ集計特徴
    if is_train:
        contract_avg = df.groupby("Contract")["MonthlyCharges"].mean()
        internet_avg = df.groupby("InternetService")["tenure"].mean()
        df["Contract_AvgCharge"] = df["Contract"].map(contract_avg)
        df["InternetService_AvgTenure"] = df["InternetService"].map(internet_avg)
    else:
        # テスト時は訓練セットの統計を使用する必要があるため、ダミー値
        df["Contract_AvgCharge"] = df["Contract"].mean()
        df["InternetService_AvgTenure"] = df["tenure"].mean()

    return df


train_feat = create_advanced_features(train_prep, is_train=True)
test_feat = create_advanced_features(test_prep, is_train=False)

print(f"拡張後の特徴数: {train_feat.shape[1]}")
print(
    f"新しい特徴: TotalCharges_log, MonthlyCharges_log, tenure_squared, tenure_cubed, etc."
)

# =====================================================
# クロスバリデーション + 学習
# =====================================================

print("\n交差検証開始...")

# ターゲット
y = train_feat["Churn"].values
X = train_feat.drop(["Churn", "id"], axis=1, errors="ignore")
X_test = test_feat.drop(["id"], axis=1, errors="ignore")

print(f"特徴数: {X.shape[1]}")
print(f"訓練データ: {X.shape[0]}, テストデータ: {X_test.shape[0]}")

# スプリッター
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 予測の保存
oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(X_test), 5))  # 5フォルド

results = {}
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold + 1}/5 ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # XGBoostハイパーパラメータ（EXP004と同じ）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # OOF予測
    oof_preds[val_idx] = model.predict(dval)

    # テスト予測
    test_preds[:, fold] = model.predict(dtest)

    # スコア
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_val, oof_preds[val_idx])
    fold_scores.append(auc)
    print(f"Fold {fold + 1} AUC: {auc:.6f}")

# =====================================================
# 結果の保存
# =====================================================

mean_auc = np.mean(fold_scores)
std_auc = np.std(fold_scores)

print(f"\n{'=' * 60}")
print(f"CV結果:")
print(f"  Mean AUC: {mean_auc:.6f}")
print(f"  Std AUC:  {std_auc:.6f}")
print(f"  Fold Scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"{'=' * 60}")

# テスト予測（5フォルド平均）
test_pred_mean = test_preds.mean(axis=1)

# ディレクトリ作成
os.makedirs("EXP/EXP006/outputs/child-exp000", exist_ok=True)

# OOF予測を保存
oof_df = pd.DataFrame(
    {
        "index": range(len(oof_preds)),
        "fold": -1,  # 持持
        "target": y,
        "prediction": oof_preds,
    }
)
oof_df.to_csv("EXP/EXP006/outputs/child-exp000/oof_predictions.csv", index=False)

# テスト予測を保存
test_df = pd.DataFrame({"id": test["id"].values, "prediction": test_pred_mean})
test_df.to_csv("EXP/EXP006/outputs/child-exp000/test_predictions.csv", index=False)

# 結果を保存
results_dict = {
    "cv_auc_mean": float(mean_auc),
    "cv_auc_std": float(std_auc),
    "cv_auc_scores": [float(s) for s in fold_scores],
    "test_pred_mean": float(test_pred_mean.mean()),
    "test_pred_std": float(test_pred_mean.std()),
}

with open("EXP/EXP006/outputs/child-exp000/results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\n✓ EXP006 completed successfully!")
print(f"  OOF: EXP/EXP006/outputs/child-exp000/oof_predictions.csv")
print(f"  Test: EXP/EXP006/outputs/child-exp000/test_predictions.csv")
print(f"  Results: EXP/EXP006/outputs/child-exp000/results.json")
