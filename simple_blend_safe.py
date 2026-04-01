"""
シンプルなブレンディング（EXP003 + EXP004）に戻す
メタモデルのスケーリング問題を回避
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("シンプルブレンディング: EXP003(45%) + EXP004(55%)")
print("=" * 70)

# EXP003とEXP004から読込
exp003 = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")

print(f"\nEXP003 shape: {exp003.shape}")
print(f"EXP004 shape: {exp004.shape}")

# 予測値抽出
if "prediction" in exp003.columns:
    pred_003 = exp003["prediction"].values
else:
    pred_003 = exp003.iloc[:, -1].values

if "prediction" in exp004.columns:
    pred_004 = exp004["prediction"].values
else:
    pred_004 = exp004.iloc[:, 1].values

print(f"\nEXP003 予測: Mean={pred_003.mean():.6f}, Std={pred_003.std():.6f}")
print(f"EXP004 予測: Mean={pred_004.mean():.6f}, Std={pred_004.std():.6f}")

# ブレンディング: 45% + 55%
y_blend = 0.45 * pred_003 + 0.55 * pred_004

print(f"\nブレンド結果: Mean={y_blend.mean():.6f}, Std={y_blend.std():.6f}")

# ID抽出
test_ids = exp004["id"].values

# サブミッション作成
submission = pd.DataFrame({"id": test_ids, "Churn": y_blend})

submission.to_csv("submission_blend_34_45_55.csv", index=False)

print(f"\n✓ submission_blend_34_45_55.csv を作成")
print(f"  (EXP003: 45%, EXP004: 55%)")
print(f"  このファイルを提出してください")
