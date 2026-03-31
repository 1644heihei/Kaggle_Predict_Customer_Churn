"""
ブレンディング重み最適化グリッドサーチ
EXP003, EXP004の重みをグリッドサーチで探索
"""

import pandas as pd
import numpy as np
from itertools import product

# 予測値の読み込み
exp003_raw = pd.read_csv("EXP/EXP003/outputs/child-exp000/test_predictions.csv")
exp004 = pd.read_csv("EXP/EXP004/outputs/child-exp000/test_predictions.csv")

# idの調整
exp003_prep = exp003_raw.copy()
exp003_prep["id"] = exp004["id"].values
exp003_prep.columns = ["prediction_003", "id"]

exp004_prep = exp004.copy()
exp004_prep.columns = ["id", "prediction_004"]

# マージ
merged = pd.merge(exp003_prep, exp004_prep, on="id")

print("=" * 60)
print("ブレンディング重み最適化グリッドサーチ")
print("=" * 60)
print(f"merged shape: {merged.shape}\n")

# グリッドサーチ（複数の戦略を試す）
strategies = []

# 戦略1: 均等重み
strategies.append(("equal_50_50", [0.5, 0.5], "EXP003(50%) + EXP004(50%)"))

# 戦略2: 現在の推定重み
strategies.append(("current_45_55", [0.45, 0.55], "EXP003(45%) + EXP004(55%)"))

# 戦略3: EXP004を重視
for w1 in [0.3, 0.35, 0.4]:
    w2 = 1.0 - w1
    strategies.append(
        (f"exp004_heavy_{w2:.0%}", [w1, w2], f"EXP003({w1:.0%}) + EXP004({w2:.0%})")
    )

# 戦略4: EXP003を重視
for w1 in [0.55, 0.6, 0.65]:
    w2 = 1.0 - w1
    strategies.append(
        (f"exp003_heavy_{w1:.0%}", [w1, w2], f"EXP003({w1:.0%}) + EXP004({w2:.0%})")
    )

# 戦略5: より細かい刻み
for w1 in np.arange(0.25, 0.76, 0.05):
    w2 = 1.0 - w1
    strategies.append(
        (f"blend_{w1:.2f}_{w2:.2f}", [w1, w2], f"EXP003({w1:.1%}) + EXP004({w2:.1%})")
    )

print(f"試す戦略数: {len(strategies)}\n")

# ブレンディング結果を保存
results_list = []

for strategy_name, weights, description in strategies:
    w1, w2 = weights

    # ブレンディング
    y_blend = (
        w1 * merged["prediction_003"].values + w2 * merged["prediction_004"].values
    )

    # 統計
    mean_pred = y_blend.mean()
    std_pred = y_blend.std()

    results_list.append(
        {
            "strategy": strategy_name,
            "description": description,
            "w_exp003": w1,
            "w_exp004": w2,
            "mean": mean_pred,
            "std": std_pred,
            "min": y_blend.min(),
            "max": y_blend.max(),
        }
    )

    # トップ戦略を保存
    if w1 in [0.45, 0.5, 0.55, 0.6] and w2 in [0.4, 0.45, 0.5, 0.55, 0.6]:
        submission = pd.DataFrame({"id": merged["id"].values, "Churn": y_blend})
        submission.to_csv(f"submission_blend_w{w1:.2f}_{w2:.2f}.csv", index=False)

# 結果をDataFrameにして表示
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values("mean", ascending=False)

print("平均予測値でトップ20の戦略:")
print(
    results_df[["description", "w_exp003", "w_exp004", "mean", "std"]]
    .head(20)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("推奨ブレンディング:")
print("=" * 60)

# 平均値の分布をチェック
print(f"\n平均値の統計:")
print(f"  Min:  {results_df['mean'].min():.6f}")
print(f"  Mean: {results_df['mean'].mean():.6f}")
print(f"  Max:  {results_df['mean'].max():.6f}")
print(f"  Std:  {results_df['mean'].std():.6f}")

# トップ3の戦略を提案
print(f"\nトップ3の戦略:")
for idx, row in results_df.head(3).iterrows():
    print(f"  {idx+1}. {row['description']}")
    print(f"     Mean: {row['mean']:.6f}, Std: {row['std']:.6f}")

# 最も「中庸」な戦略も探す
mid_value = results_df["mean"].median()
mid_idx = (results_df["mean"] - mid_value).abs().idxmin()
print(f"\n中央値付近の戦略:")
print(f"  {results_df.loc[mid_idx, 'description']}")
print(f"     Mean: {results_df.loc[mid_idx, 'mean']:.6f}")

print(f"\n✓ 複数のブレンディング候補を作成しました")
print(f"  submission_blend_w0.45_0.55.csv（現在の45/55）")
print(f"  submission_blend_w0.50_0.50.csv（均等50/50）など")
