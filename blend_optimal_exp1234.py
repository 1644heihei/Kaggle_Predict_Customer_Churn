"""
EXP001, EXP002, EXP003, EXP004の4モデルブレンディング
最適化された重み現在のEXP003/004ブレンド（0.91388）から最適化するための戦略
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 予測値の読み込み
exp001_raw = pd.read_csv("EXP/EXP001/outputs/child-exp000/test_predictions.csv")
exp002_raw = pd.read_csv("EXP/EXP002/outputs/child-exp000/test_predictions.csv")
exp003_raw = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")

# 提出ファイル（ターゲット）
sub_current = pd.read_csv("submission_exp003_exp004.csv")

# 形式を統一
exp001_prep = exp001_raw.copy()
exp001_prep["id"] = exp004["id"].values
exp001_prep.columns = ["prediction_001", "id"]

exp002_prep = exp002_raw.copy()
exp002_prep["id"] = exp004["id"].values
exp002_prep.columns = ["prediction_002", "id"]

exp003_prep = exp003_raw.copy()
exp003_prep["id"] = exp004["id"].values
exp003_prep.columns = ["prediction_003", "id"]

exp004_prep = exp004.copy()
exp004_prep.columns = ["id", "prediction_004"]

# マージ
merged = pd.merge(exp001_prep, exp002_prep, on="id")
merged = pd.merge(merged, exp003_prep, on="id")
merged = pd.merge(merged, exp004_prep, on="id")

print("=" * 60)
print("4モデル自動重み最適化ブレンディング")
print("=" * 60)
print(f"Merged shape: {merged.shape}\n")

# 相関行列
corr = merged[
    ["prediction_001", "prediction_002", "prediction_003", "prediction_004"]
].corr()
print("相関マトリックス:")
print(corr)
print()

# 線形回帰で最適重み推定
X = merged[
    ["prediction_001", "prediction_002", "prediction_003", "prediction_004"]
].values
y = sub_current["Churn"].values

model = LinearRegression()
model.fit(X, y)

coef = model.coef_
total = np.sum(coef)

print("推定された重み (原始):")
for i, c in enumerate(coef, 1):
    print(f"  EXP{i:03d}: {c:.6f}")
print(f"  Sum: {total:.6f}")
print()

# 重みが正で合計が1になるように正規化
if total > 0 and np.all(coef >= 0):
    norm_weights = coef / total
else:
    # 負の重みがある場合は、すべてをシフト
    min_coef = np.min(coef)
    if min_coef < 0:
        coef_shifted = coef - min_coef
        total_shifted = np.sum(coef_shifted)
        norm_weights = coef_shifted / total_shifted
    else:
        norm_weights = coef / total

    print("⚠️  負の重みを検出。調整後の正規化重み:")

print("最適化された正規化重み:")
for i, w in enumerate(norm_weights, 1):
    print(f"  EXP{i:03d}: {w:.3%}")
print()

# ブレンド作成
y_blend = (
    norm_weights[0] * merged["prediction_001"].values
    + norm_weights[1] * merged["prediction_002"].values
    + norm_weights[2] * merged["prediction_003"].values
    + norm_weights[3] * merged["prediction_004"].values
)

print("ブレンド結果統計:")
print(f"  Mean: {y_blend.mean():.6f}")
print(f"  Std:  {y_blend.std():.6f}")
print(f"  Min:  {y_blend.min():.6f}")
print(f"  Max:  {y_blend.max():.6f}")

submission = pd.DataFrame({"id": merged["id"].values, "Churn": y_blend})
submission.to_csv("submission_blend_exp1234_optimized.csv", index=False)
print(f"\n✓ submission_blend_exp1234_optimized.csv を保存")

# 複数の戦略を比較
print("\n" + "=" * 60)
print("複数ブレンディング戦略の比較")
print("=" * 60)

strategies = {
    "exp_034_equal": {
        "weights": np.array([0, 0, 1 / 3, 1 / 3, 1 / 3])[:4],
        "name": "EXP001-004 均等 (25%,25%,25%,25%)",
    },
    "exp_1234_45_50": {
        "weights": np.array([0, 0, 0.45, 0.55]),
        "name": "EXP003(45%) + EXP004(55%) - 現在",
    },
    "exp_1234_weighted": {
        "weights": np.array([0.1, 0, 0.3, 0.6]),
        "name": "EXP001(10%) + EXP003(30%) + EXP004(60%)",
    },
    "exp_1234_double_004": {
        "weights": np.array([0, 0, 1 / 3, 2 / 3]),
        "name": "EXP003(33%) + EXP004(67%)",
    },
}

submission_list = []
for key, strategy in strategies.items():
    weights = strategy["weights"]
    y_pred = (
        weights[0] * merged["prediction_001"].values
        + weights[1] * merged["prediction_002"].values
        + weights[2] * merged["prediction_003"].values
        + weights[3] * merged["prediction_004"].values
    )

    print(f"\n{strategy['name']}")
    print(f"  Mean: {y_pred.mean():.6f} | Std: {y_pred.std():.6f}")

    submission_tmp = pd.DataFrame({"id": merged["id"].values, "Churn": y_pred})
    submission_tmp.to_csv(f"submission_blend_{key}.csv", index=False)
    submission_list.append((key, y_pred))

print("\n" + "=" * 60)
print("推奨事項")
print("=" * 60)
print(
    """
EXP002の最適重みが0%の理由：
  - EXP003, EXP004と高度に相関（r > 0.95）
  - 特別な情報を追加できない
  
0.917以上を達成するには：
  1. 新しい特徴エンジニアリング（EXP006）を追加
  2. または、複数ブレンディング版を提出してLB上で精度を確認
  
生成されたサブミッション：
  - submission_blend_exp1234_optimized.csv（最適重み）
  - submission_blend_exp_1234_equal.csv（均等 25%,25%,25%,25%）
  - submission_blend_exp_1234_weighted.csv（10%,0%,30%,60%）
  - submission_blend_exp_1234_double_004.csv（33%,67% for 003/004）
"""
)
