#!/usr/bin/env python3
"""Quick Ensemble Blending from EXP001-004"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load predictions
base_path = Path("../EXP")
print("Loading predictions from EXP folders...")
exp001 = pd.read_csv(base_path / "EXP001/outputs/child-exp000/test_predictions.csv")
exp003 = pd.read_csv(base_path / "EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv(base_path / "EXP004/outputs/child-exp000/test_predictions.csv")

print(f"✓ EXP001: {exp001.shape}, CV AUC: 0.9134")
print(f"✓ EXP003: {exp003.shape}, CV AUC: 0.9159")
print(f"✓ EXP004: {exp004.shape}, CV AUC: 0.9164 ⭐")

# Weighted blending
weights = {"EXP001": 0.25, "EXP003": 0.25, "EXP004": 0.50}
blended = (
    exp001["prediction"].values * weights["EXP001"]
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
submission.to_csv("submission_blended.csv", index=False)
print(f"\n✅ Saved: submission_blended.csv ({submission.shape})")
print(f"Sample: {submission.head(3).values}")
