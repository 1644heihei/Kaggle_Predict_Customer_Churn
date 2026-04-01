"""
OOFスタッキング実装: 3モデル統合
EXP004, EXP007, EXP008 のOOF予測を使用してメタモデルを学習

3本柱戦略:
  - EXP004: XGBoost Optuna (最高精度 0.9164)
  - EXP007: CatBoost (多様性)
  - EXP008: XGBoost 不均衡対応 (不均衡に強い)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("OOFスタッキング: 3モデル統合（EXP004, EXP007, EXP008）")
print("=" * 80)

# =====================================================
# Step 1: OOF予測の読み込み
# =====================================================
print("\n[STEP 1] OOF予測の読み込み")
print("-" * 80)

models_info = {
    "EXP004": "EXP/EXP004/outputs/child-exp000/oof_predictions.csv",
    "EXP007": "EXP/EXP007/outputs/child-exp000/oof_predictions.csv",
    "EXP008": "EXP/EXP008/outputs/child-exp000/oof_predictions.csv",
}

oof_data = {}
for model_name, path in models_info.items():
    try:
        df = pd.read_csv(path)
        oof_data[model_name] = df
        print(f"✓ {model_name}: {df.shape}")
    except FileNotFoundError:
        print(f"❌ {model_name}: ファイルが見つかりません")
        exit()

# =====================================================
# Step 2: テスト予測の読み込み
# =====================================================
print("\n[STEP 2] テスト予測の読み込み")
print("-" * 80)

test_data = {}
for model_name in models_info.keys():
    path = models_info[model_name].replace("oof_", "test_")
    try:
        df = pd.read_csv(path)
        test_data[model_name] = df
        print(f"✓ {model_name}: {df.shape}")
    except FileNotFoundError:
        print(f"❌ {model_name}: テストファイルが見つかりません")
        exit()

# =====================================================
# Step 3: OOF特徴の統合
# =====================================================
print("\n[STEP 3] OOF特徴の統合とターゲット抽出")
print("-" * 80)

oof_features = pd.DataFrame()

for i, (model_name, df) in enumerate(oof_data.items(), 1):
    if "prediction" in df.columns:
        pred_col = df["prediction"].values
    else:
        pred_col = df.iloc[:, -1].values

    oof_features[f"pred_{model_name}"] = pred_col
    print(f"  {i}. {model_name}: {pred_col.shape}")

# ターゲットを抽出
if "target" in oof_data["EXP004"].columns:
    y_train = oof_data["EXP004"]["target"].values
else:
    y_train = oof_data["EXP004"].iloc[:, -2].values

print(f"\n✓ OOF特徴統合完了: {oof_features.shape}")
print(f"  ターゲット: {y_train.shape}")

# =====================================================
# Step 4: OOF特徴の統計
# =====================================================
print("\n[STEP 4] OOF特徴の統計")
print("-" * 80)

print("\nOOF相関マトリックス:")
corr_matrix = oof_features.corr()
print(corr_matrix.to_string())

print("\n各モデルの基本統計:")
for col in oof_features.columns:
    mean = oof_features[col].mean()
    std = oof_features[col].std()
    print(f"  {col}: Mean={mean:.6f}, Std={std:.6f}")

# =====================================================
# Step 5: テスト特徴の統合
# =====================================================
print("\n[STEP 5] テスト特徴の統合")
print("-" * 80)

test_features = pd.DataFrame()
test_ids = None

for model_name, df in test_data.items():
    if "prediction" in df.columns:
        pred_col = df["prediction"].values
    else:
        pred_col = df.iloc[:, -1].values

    test_features[f"pred_{model_name}"] = pred_col

    if test_ids is None and "id" in df.columns:
        test_ids = df["id"].values

print(f"✓ テスト特徴統合完了: {test_features.shape}")

# =====================================================
# Step 6: データの正規化
# =====================================================
print("\n[STEP 6] データの正規化")
print("-" * 80)

scaler = StandardScaler()
oof_features_scaled = scaler.fit_transform(oof_features)
test_features_scaled = scaler.transform(test_features)

print(f"✓ 正規化完了")

# =====================================================
# Step 7: メタモデルの学習
# =====================================================
print("\n[STEP 7] メタモデルのトレーニング")
print("-" * 80)

# Ridge回帰
print("\nメタモデル A: Ridge 回帰")
ridge_meta = Ridge(alpha=1.0, random_state=42)
ridge_meta.fit(oof_features_scaled, y_train)

ridge_coef = ridge_meta.coef_
ridge_pred_train = ridge_meta.predict(oof_features_scaled)
ridge_auc = roc_auc_score(y_train, ridge_pred_train)

print(f"  係数: {ridge_coef}")
print(f"  合計: {ridge_coef.sum():.6f}")
print(f"  Train AUC: {ridge_auc:.6f}")

# ロジスティック回帰
print("\nメタモデル B: ロジスティック回帰")
lr_meta = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
lr_meta.fit(oof_features_scaled, y_train)

lr_coef = lr_meta.coef_[0]
lr_pred_train = lr_meta.predict_proba(oof_features_scaled)[:, 1]
lr_auc = roc_auc_score(y_train, lr_pred_train)

print(f"  係数: {lr_coef}")
print(f"  合計: {lr_coef.sum():.6f}")
print(f"  Train AUC: {lr_auc:.6f}")

# =====================================================
# Step 8: テスト予測の生成
# =====================================================
print("\n[STEP 8] テスト予測の生成とアンサンブル")
print("-" * 80)

ridge_test_pred = ridge_meta.predict(test_features_scaled)
lr_test_pred = lr_meta.predict_proba(test_features_scaled)[:, 1]

# 2つのメタモデルを平均
y_pred_final = (ridge_test_pred + lr_test_pred) / 2
y_pred_final = np.clip(y_pred_final, 0, 1)

print(f"✓ テスト予測生成完了")

print(f"\nRidge メタモデル予測統計:")
print(f"  Mean: {ridge_test_pred.mean():.6f}")
print(f"  Std:  {ridge_test_pred.std():.6f}")

print(f"\nLR メタモデル予測統計:")
print(f"  Mean: {lr_test_pred.mean():.6f}")
print(f"  Std:  {lr_test_pred.std():.6f}")

print(f"\n最終統合予測統計:")
print(f"  Mean: {y_pred_final.mean():.6f}")
print(f"  Std:  {y_pred_final.std():.6f}")

# =====================================================
# Step 9: サブミッション作成
# =====================================================
print("\n[STEP 9] サブミッション作成")
print("-" * 80)

submission_final = pd.DataFrame({"id": test_ids, "Churn": y_pred_final})
submission_final.to_csv("submission_stacking_3models_final.csv", index=False)
print(f"✓ submission_stacking_3models_final.csv を保存 ⭐")

# 単純版（比較用）
y_simple = test_features.mean(axis=1).values
submission_simple = pd.DataFrame({"id": test_ids, "Churn": y_simple})
submission_simple.to_csv("submission_simple_avg_3models.csv", index=False)
print(f"✓ submission_simple_avg_3models.csv を保存（比較用）")

# =====================================================
# Step 10: 比較分析
# =====================================================
print("\n[STEP 10] 比較分析")
print("-" * 80)

print(f"\n比較: 単純平均 vs メタモデルスタッキング")
print(f"  単純平均（3モデル）Mean: {y_simple.mean():.6f}")
print(f"  Stacking最終 Mean: {y_pred_final.mean():.6f}")
print(f"  差分: {(y_pred_final.mean() - y_simple.mean())*1000:.3f}ppt")

print(f"\n各モデルの係数比較:")
print(f"\nRidge係数（正規化前）:")
for i, (model, coef) in enumerate(zip(["EXP004", "EXP007", "EXP008"], ridge_coef)):
    print(f"  {model}: {coef:.6f}")

print(f"\nLR係数（正規化前）:")
for i, (model, coef) in enumerate(zip(["EXP004", "EXP007", "EXP008"], lr_coef)):
    print(f"  {model}: {coef:.6f}")

# =====================================================
# 最終推奨
# =====================================================
print("\n" + "=" * 80)
print("📊 最終推奨")
print("=" * 80)
print(
    f"""
【推奨提出ファイル】

✅ submission_stacking_3models_final.csv ⭐（推奨）
   - EXP004, EXP007, EXP008 の3モデルメタモデル統合
   - Ridge + LR 平均版
   - 期待LB: 0.9165～0.9175（0.917突破見込み）

【成功指標】
 Train AUC (Ridge): {ridge_auc:.6f}
 Train AUC (LR):    {lr_auc:.6f}
 
【次のステップ】
1. submission_stacking_3models_final.csv をKaggleに提出
2. LB スコアを確認
3. 0.917超え達成か確認

【モデル構成の理由】
- EXP004: 最高精度（0.9164）をベース
- EXP007: CatBoost で異なるアルゴリズム多様性
- EXP008: クラス不均衡に強い独自視点

相関が高い3モデルを最適重みで統合することで
+0.2～0.3% の改善を期待
""".format(
        ridge_auc, lr_auc
    )
)

print("=" * 80)
