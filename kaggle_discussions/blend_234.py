#!/usr/bin/env python3
"""Blending with EXP002, EXP003, EXP004"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load predictions
base_path = Path("../EXP")
print("Loading predictions from EXP folders...")
exp002 = pd.read_csv(base_path / "EXP002/outputs/child-exp000/test_predictions.csv")
exp003 = pd.read_csv(base_path / "EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv(base_path / "EXP004/outputs/child-exp000/test_predictions.csv")

print(f"✓ EXP002: {exp002.shape}, CV AUC: 0.9126")
print(f"✓ EXP003: {exp003.shape}, CV AUC: 0.9159")
print(f"✓ EXP004: {exp004.shape}, CV AUC: 0.9164 ⭐")

# Weighted blending (EXP004最高性能なので50%配分)
weights = {"EXP002": 0.25, "EXP003": 0.25, "EXP004": 0.50}
blended = (
    exp002["prediction"].values * weights["EXP002"]
    + exp003["prediction"].values * weights["EXP003"]
    + exp004["prediction"].values * weights["EXP004"]
)

print(f"\n【Blending Results】")
print(f"Weights: {weights}")
print(f"Prediction stats:")
print(f"  Mean: {blended.mean():.6f}")
print(f"  Min:  {blended.min():.6f}")
print(f"  Max:  {blended.max():.6f}")
print(f"  Std:  {blended.std():.6f}")

# Save
submission = pd.DataFrame({"id": exp004["id"].values, "Churn": blended})
submission.to_csv("submission_blend_234.csv", index=False)
print(f"\n✅ Saved: submission_blend_234.csv ({submission.shape})")
print(f"Sample predictions: {submission['Churn'].head(3).values}")
