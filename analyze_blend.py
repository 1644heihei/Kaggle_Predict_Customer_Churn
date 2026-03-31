import pandas as pd
import numpy as np

# ファイル読み込み
sub = pd.read_csv("submission_exp003_exp004.csv")
exp003 = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")

print("=== 現在のブレンディング分析 ===")
print(f"Submission shape: {sub.shape}")
print(f"\n統計:")
print(f'  Mean:  {sub["Churn"].mean():.6f}')
print(f'  Std:   {sub["Churn"].std():.6f}')
print(f'  Min:   {sub["Churn"].min():.6f}')
print(f'  Max:   {sub["Churn"].max():.6f}')

# ブレンディング比率推測 - idが異なる形式なので調整
exp003_prep = exp003.copy()
exp003_prep["id"] = exp004["id"].values
exp003_prep.columns = ["prediction_003", "id"]

exp004_prep = exp004.copy()
exp004_prep.columns = ["id", "prediction_004"]

merged = pd.merge(exp003_prep, exp004_prep, on="id", suffixes=("_003", "_004"))
merged["blend"] = sub["Churn"].values

# 相関
corr = merged[["prediction_003", "prediction_004", "blend"]].corr()
print(f"\n相関マトリックス:")
print(corr)

# 重み推測: blend = w*pred003 + (1-w)*pred004
# 線形回帰で推測
from sklearn.linear_model import LinearRegression

X = merged[["prediction_003", "prediction_004"]].values
y = merged["blend"].values
model = LinearRegression()
model.fit(X, y)
w003, w004 = model.coef_

print(f"\n推測された重み付け:")
print(f"  EXP003: {w003:.4f}")
print(f"  EXP004: {w004:.4f}")
print(f"  Sum: {w003+w004:.4f}")
