#!/usr/bin/env python3
"""提出用CSV生成スクリプト（EXP004最適化版を含む）

複数のアンサンブル戦略オプション：
1. EXP004のみ（最高スコア AUC 0.9164）
2. EXP003 + EXP004（異なるハイパラ）
3. 4モデル加重平均（EXP001, 002, 003, 004）
"""

import pandas as pd
import numpy as np
from pathlib import Path

# テストデータのIDを読込
test_df = pd.read_csv("data/test.csv", encoding="utf-8")
test_ids = test_df["id"].values

print("=" * 60)
print("📊 提出用ファイル生成")
print("=" * 60)

# ===== 各EXPの予測を読込 =====
predictions = {}

# EXP001
try:
    exp001_pred = pd.read_csv("EXP/EXP001/outputs/child-exp000/test_predictions.csv")
    predictions["EXP001"] = exp001_pred["prediction"].values
    print("✓ EXP001 loaded (CV AUC: 0.9134)")
except FileNotFoundError:
    print("✗ EXP001 not found")

# EXP002
try:
    exp002_pred = pd.read_csv("EXP/EXP002/outputs/child-exp000/test_predictions.csv")
    predictions["EXP002"] = exp002_pred["prediction"].values
    print("✓ EXP002 loaded (CV AUC: 0.9126)")
except FileNotFoundError:
    print("✗ EXP002 not found")

# EXP003
try:
    exp003_pred = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
    predictions["EXP003"] = exp003_pred["prediction"].values
    print("✓ EXP003 loaded (CV AUC: 0.9159)")
except FileNotFoundError:
    print("✗ EXP003 not found")

# EXP004（新）
try:
    exp004_pred = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")
    predictions["EXP004"] = exp004_pred["prediction"].values
    print("✓ EXP004 loaded (CV AUC: 0.9164) ⭐ NEW")
except FileNotFoundError:
    print("✗ EXP004 not found")

print(f"\n✓ Total models loaded: {len(predictions)}")

# ===== 複数の提出ファイルを生成 =====

# 1. EXP004のみ（最高スコア）
if "EXP004" in predictions:
    print("\n" + "=" * 60)
    print("📝 Option 1: EXP004 Only (Best Single Model)")
    print("=" * 60)

    ensemble_pred = predictions["EXP004"]
    submission_df = pd.DataFrame({"id": test_ids, "Churn": ensemble_pred})
    submission_df.to_csv("submission_exp004_only.csv", index=False)

    print(f"✓ Saved: submission_exp004_only.csv")
    print(f"  CV AUC: 0.9164")
    print(f"  Rows: {len(submission_df)}")
    print(f"  Pred range: {ensemble_pred.min():.6f} ~ {ensemble_pred.max():.6f}")
    print(f"  Mean: {ensemble_pred.mean():.6f}")

# 2. EXP003 + EXP004（2モデルアンサンブル）
if "EXP003" in predictions and "EXP004" in predictions:
    print("\n" + "=" * 60)
    print("📝 Option 2: EXP003 + EXP004 (2-Model Ensemble)")
    print("=" * 60)

    # スコアベースの重み付け
    weights = {"EXP003": 0.45, "EXP004": 0.55}  # EXP004（0.9164）を重視

    ensemble_pred = (
        predictions["EXP003"] * weights["EXP003"]
        + predictions["EXP004"] * weights["EXP004"]
    )

    submission_df = pd.DataFrame({"id": test_ids, "Churn": ensemble_pred})
    submission_df.to_csv("submission_exp003_exp004.csv", index=False)

    print(f"✓ Saved: submission_exp003_exp004.csv")
    print(f"  Weights: EXP003={weights['EXP003']}, EXP004={weights['EXP004']}")
    print(f"  Rows: {len(submission_df)}")
    print(f"  Pred range: {ensemble_pred.min():.6f} ~ {ensemble_pred.max():.6f}")
    print(f"  Mean: {ensemble_pred.mean():.6f}")

# 3. 4モデル加重アンサンブル
if len(predictions) == 4:
    print("\n" + "=" * 60)
    print("📝 Option 3: 4-Model Weighted Ensemble")
    print("=" * 60)

    # スコアベースの重み付け
    weights = {
        "EXP001": 0.18,  # 0.9134
        "EXP002": 0.18,  # 0.9126
        "EXP003": 0.32,  # 0.9159
        "EXP004": 0.32,  # 0.9164 ⭐
    }

    ensemble_pred = (
        predictions["EXP001"] * weights["EXP001"]
        + predictions["EXP002"] * weights["EXP002"]
        + predictions["EXP003"] * weights["EXP003"]
        + predictions["EXP004"] * weights["EXP004"]
    )

    submission_df = pd.DataFrame({"id": test_ids, "Churn": ensemble_pred})
    submission_df.to_csv("submission_4model_ensemble.csv", index=False)

    print(f"✓ Saved: submission_4model_ensemble.csv")
    print(f"  Weights:")
    for model, weight in weights.items():
        print(f"    - {model}: {weight:.2f}")
    print(f"  Rows: {len(submission_df)}")
    print(f"  Pred range: {ensemble_pred.min():.6f} ~ {ensemble_pred.max():.6f}")
    print(f"  Mean: {ensemble_pred.mean():.6f}")

print("\n" + "=" * 60)
print("✅ All submission files generated!")
print("=" * 60)
print("\n🎯 推奨提出ファイル:")
print("   → submission_exp004_only.csv (最高スコア)")
print("   または")
print("   → submission_4model_ensemble.csv (安定性)")
