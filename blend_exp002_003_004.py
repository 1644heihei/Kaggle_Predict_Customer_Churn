"""
EXP002, EXP003, EXP004の3モデルブレンディング
重み最適化版
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# 予測値の読み込み
exp002_raw = pd.read_csv("EXP/EXP002/outputs/child-exp000/test_predictions.csv")
exp003_raw = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")
sub_current = pd.read_csv("submission_exp003_exp004.csv")

# idの調整
exp002_prep = exp002_raw.copy()
exp002_prep["id"] = exp004["id"].values
exp002_prep.columns = ["prediction_002", "id"]

exp003_prep = exp003_raw.copy()
exp003_prep["id"] = exp004["id"].values
exp003_prep.columns = ["prediction_003", "id"]

exp004_prep = exp004.copy()
exp004_prep.columns = ["id", "prediction_004"]

# マージ
merged = pd.merge(exp002_prep, exp003_prep, on="id")
merged = pd.merge(merged, exp004_prep, on="id")

print("=" * 50)
print("3モデルブレンディング分析")
print("=" * 50)
print(f"Merged shape: {merged.shape}\n")

# 相関行列
corr = merged[["prediction_002", "prediction_003", "prediction_004"]].corr()
print("相関マトリックス:")
print(corr)
print()

# 線形回帰で最適重み推定
X = merged[["prediction_002", "prediction_003", "prediction_004"]].values
y = sub_current["Churn"].values

model = LinearRegression()
model.fit(X, y)

coef_002, coef_003, coef_004 = model.coef_
total = coef_002 + coef_003 + coef_004

print("推定された重み (正規化前):")
print(f"  EXP002: {coef_002:.6f}")
print(f"  EXP003: {coef_003:.6f}")
print(f"  EXP004: {coef_004:.6f}")
print(f"  Sum: {total:.6f}")
print()

if total > 0:
    norm_002 = coef_002 / total
    norm_003 = coef_003 / total
    norm_004 = coef_004 / total

    print("正規化された重み:")
    print(f"  EXP002: {norm_002:.1%}")
    print(f"  EXP003: {norm_003:.1%}")
    print(f"  EXP004: {norm_004:.1%}")
    print()

    # ブレンド作成
    y_blend = (
        norm_002 * merged["prediction_002"].values
        + norm_003 * merged["prediction_003"].values
        + norm_004 * merged["prediction_004"].values
    )

    print("ブレンド予測の統計:")
    print(f"  Mean: {y_blend.mean():.6f}")
    print(f"  Std:  {y_blend.std():.6f}")
    print(f"  Min:  {y_blend.min():.6f}")
    print(f"  Max:  {y_blend.max():.6f}")
    print()

    # サブミッション作成
    submission = pd.DataFrame({"id": merged["id"].values, "Churn": y_blend})

    submission.to_csv("submission_blend_exp234_v1.csv", index=False)
    print(
        f"✓ submission_blend_exp234_v1.csv を保存 (重み: EXP002 {norm_002:.1%}, EXP003 {norm_003:.1%}, EXP004 {norm_004:.1%})"
    )

    # より単純な重み付けも試す
    print("\n" + "=" * 50)
    print("単純な均等重み付け（1/3 each）")
    print("=" * 50)
    y_simple = (
        merged["prediction_002"].values
        + merged["prediction_003"].values
        + merged["prediction_004"].values
    ) / 3

    submission_simple = pd.DataFrame({"id": merged["id"].values, "Churn": y_simple})

    print("ブレンド予測の統計:")
    print(f"  Mean: {y_simple.mean():.6f}")
    print(f"  Std:  {y_simple.std():.6f}")
    print(f"  Min:  {y_simple.min():.6f}")
    print(f"  Max:  {y_simple.max():.6f}")

    submission_simple.to_csv("submission_blend_exp234_equal.csv", index=False)
    print(f"✓ submission_blend_exp234_equal.csv を保存")

    # 加重平均も試す (25%, 25%, 50%)
    print("\n" + "=" * 50)
    print("試行: 25% 25% 50% 重み付け")
    print("=" * 50)
    y_weighted = (
        0.25 * merged["prediction_002"].values
        + 0.25 * merged["prediction_003"].values
        + 0.50 * merged["prediction_004"].values
    )

    submission_weighted = pd.DataFrame({"id": merged["id"].values, "Churn": y_weighted})

    print("ブレンド予測の統計:")
    print(f"  Mean: {y_weighted.mean():.6f}")
    print(f"  Std:  {y_weighted.std():.6f}")
    print(f"  Min:  {y_weighted.min():.6f}")
    print(f"  Max:  {y_weighted.max():.6f}")

    submission_weighted.to_csv("submission_blend_exp234_252550.csv", index=False)
    print(f"✓ submission_blend_exp234_252550.csv を保存")
