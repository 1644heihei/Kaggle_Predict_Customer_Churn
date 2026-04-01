import sys
import time
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch

print("[TEST] Loading config...")
with open("EXP/EXP009/config/child-exp000.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print("[OK] Config loaded")

print("[TEST] Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(f"[OK] Loaded: train {train_df.shape}, test {test_df.shape}")

print("[TEST] Preprocessing...")
y_train = (train_df["Churn"] == "Yes").astype(int).values
X_train = train_df.drop(columns=["Churn", "id"], errors="ignore")
X_test = test_df.drop(columns=["id"], errors="ignore")

print("[TEST] Encoding...")
categorical_cols = X_train.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    try:
        X_test[col] = le.transform(X_test[col].astype(str))
    except ValueError:
        X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_train.mean(numeric_only=True))
print(f"[OK] Encoded: {X_train.shape}")

print("[TEST] Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"[OK] Scaled")

print("[TEST] Creating tensors (sample 1000 rows)...")
X_tr_tensor = torch.FloatTensor(X_train_scaled[:1000])
y_tr_tensor = torch.FloatTensor(y_train[:1000])
print(f"[OK] Tensor shapes: {X_tr_tensor.shape}, {y_tr_tensor.shape}")

print("[TEST] Creating DataLoader...")
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"[OK] DataLoader created")

print("[TEST] Iterating batches...")
for i, (X_batch, y_batch) in enumerate(loader):
    print(f"  Batch {i+1}: X{X_batch.shape}, y{y_batch.shape}")
    if i >= 2:
        break

print("\n[SUCCESS] All tests passed!")
