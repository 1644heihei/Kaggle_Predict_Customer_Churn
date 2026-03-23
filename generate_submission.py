#!/usr/bin/env python3
"""提出用CSV生成スクリプト（EXP003のみ）"""

import pandas as pd
import numpy as np

# テストデータのIDを読込
test_df = pd.read_csv("data/test.csv", encoding="utf-8")

# EXP003（XGBoost + Feature Engineering）の予測を読込
try:
    test_pred_exp003_path1 = "EXP/EXP003/outputs/child-exp000/test_predictions.csv"
    test_pred_exp003_path2 = "outputs/child-exp000/test_predictions.csv"
    try:
        test_pred_exp003 = pd.read_csv(test_pred_exp003_path1)
        print("✓ EXP003 predictions loaded from EXP/EXP003/outputs/")
    except FileNotFoundError:
        test_pred_exp003 = pd.read_csv(test_pred_exp003_path2)
        print("✓ EXP003 predictions loaded from outputs/")
except FileNotFoundError:
    print("✗ EXP003 predictions not found!")
    exit(1)

print(f"✓ Test IDs loaded: {len(test_df)}")

# EXP003 の予測を使用
ensemble_pred = test_pred_exp003["prediction"].values

print(f"✓ Using EXP003 predictions only (CV AUC: 0.9159)")

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
