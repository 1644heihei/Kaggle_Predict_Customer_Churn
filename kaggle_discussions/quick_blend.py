#!/usr/bin/env python3
"""
Quick Ensemble Blending from EXP001-004
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load predictions from EXP folders
base_path = Path("../EXP")

print("Loading predictions from EXP folders...")
exp001 = pd.read_csv(base_path / "EXP001/outputs/child-exp000/test_predictions.csv")
exp003 = pd.read_csv(base_path / "EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv(base_path / "EXP004/outputs/child-exp000/test_predictions.csv")

print(f"EXP001 shape: {exp001.shape}, AUC: 0.9134")
print(f"EXP003 shape: {exp003.shape}, AUC: 0.9159")
print(f"EXP004 shape: {exp004.shape}, AUC: 0.9164 ⭐")

# Simple weighted average blending
print("\n【Blending Results】")
weights = {
    "EXP001": 0.25,
    "EXP003": 0.25,
    "EXP004": 0.50,  # Highest performing
}

blended = (
    exp001["Churn"] * weights["EXP001"]
    + exp003["Churn"] * weights["EXP003"]
    + exp004["Churn"] * weights["EXP004"]
)

print(f"Weights: {weights}")
print(f"Blended prediction stats:")
print(f"  Mean: {blended.mean():.6f}")
print(f"  Min:  {blended.min():.6f}")
print(f"  Max:  {blended.max():.6f}")

# Save submission
submission = pd.DataFrame({"id": exp004["id"], "Churn": blended})

submission.to_csv("submission_blended.csv", index=False)
print(f"\n✅ Submission saved: submission_blended.csv")
print(f"   Shape: {submission.shape}")
print(f"   Predictions: {submission['Churn'].values[:10]}")
