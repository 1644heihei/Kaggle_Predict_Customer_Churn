#!/usr/bin/env python3
"""提出用CSV生成スクリプト"""

import pandas as pd
import numpy as np

# テストデータのIDを読込
test_df = pd.read_csv("data/test.csv", encoding="utf-8")

# EXP002 の予測を読込
test_pred_exp002 = pd.read_csv("outputs/child-exp000/test_predictions.csv")

# EXP001 の予測を読込（ある場合）
try:
    test_pred_exp001 = pd.read_csv(
        "EXP/EXP001/outputs/child-exp000/test_predictions.csv"
    )
    print("✓ EXP001 predictions loaded")
    has_exp001 = True
except FileNotFoundError:
    print("⚠ EXP001 predictions not found (will use EXP002 only)")
    has_exp001 = False

print(f"✓ EXP002 predictions loaded: {test_pred_exp002.shape}")
print(f"✓ Test IDs loaded: {len(test_df)}")

# アンサンブル（平均）
if has_exp001 and len(test_pred_exp001) == len(test_pred_exp002):
    # EXP001 と EXP002 の平均
    ensemble_pred = (
        test_pred_exp001["prediction"].values + test_pred_exp002["prediction"].values
    ) / 2
    print(f"✓ Ensemble: (EXP001 + EXP002) / 2")
else:
    # EXP002 のみ使用
    ensemble_pred = test_pred_exp002["prediction"].values
    print(f"✓ Using EXP002 predictions only")

# 提出用 CSV を作成
submission_df = pd.DataFrame({"id": test_df["id"].values, "Churn": ensemble_pred})

# 保存
submission_df.to_csv("submission.csv", index=False)
print(f"\n✓ Submission saved: submission.csv ({len(submission_df)} rows)")
print(f"  Min prediction: {ensemble_pred.min():.6f}")
print(f"  Max prediction: {ensemble_pred.max():.6f}")
print(f"  Mean prediction: {ensemble_pred.mean():.6f}")

# サンプル表示
print("\nFirst 10 rows:")
print(submission_df.head(10))
