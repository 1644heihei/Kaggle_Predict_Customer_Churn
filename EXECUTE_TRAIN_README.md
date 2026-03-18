# 訓練実行ガイド

## 🚀 概要

このプロジェクトの訓練は **Google Colab** 上で実行されることを想定しています。

**主要なノートブック**: [execute_train.ipynb](execute_train.ipynb)

---

## 📋 前準備

### 1. Kaggleデータセットのダウンロード
```bash
# ローカルマシン上で実行
kaggle competitions download -c predict-customer-churn
unzip predict-customer-churn.zip -d data/
```

結果：
```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 2. Google Drive へのアップロード
```
My Drive/
└── kaggle_churn/
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── sample_submission.csv
    ├── EXP/
    └── outputs/
```

---

## 🔧 execute_train.ipynb の使い方

### Cell 1: ライブラリインストール
```python
# PyTorch, scikit-learn, LightGBM等をインストール
pip install lightgbm xgboost scikit-learn pandas numpy
```

### Cell 2-5: 初期化
- Google Drive マウント
- 作業ディレクトリ設定
- 必要なモジュール import

### Cell 6: **実験指定（最重要）**
```python
# ここで EXP と child-exp を指定
EXP_NAME = "EXP001"
CHILD_EXP_NAME = "child-exp000"
```

これだけ変更すれば、以下が自動実行：
- config ファイル読込
- 訓練スクリプト実行
- OOF予測生成
- 結果を Google Drive に保存

### Cell 7: 訓練実行
```python
# 訓練スクリプト実行
os.system(f"python EXP/{EXP_NAME}/train.py \
    --config EXP/{EXP_NAME}/config/{CHILD_EXP_NAME}.yaml")
```

### Cell 8: 結果の確認と保存
```python
# results.json を読込
# CV スコアと結果サマリーを表示
```

---

## 📂 ファイル構成（訓練時）

```
EXP001/
├── train.py                    # ← これを execute_train.ipynb が実行
├── config/
│   └── child-exp000.yaml       # ← EXP_NAME/CHILD_EXP_NAME から参照
└── outputs/
    └── child-exp000/           # ← 自動生成
        ├── config.yaml         # （実行設定のコピー）
        ├── oof_predictions.csv # フォール毎の予測
        └── results.json        # {cv_auc: 0.856, ...}
```

---

## 🔄 複数 child-exp を連続実行

異なるハイパラで連続実行したい場合：

```python
# execute_train.ipynb の Cell 6 を複数回実行

# 1回目
EXP_NAME = "EXP001"
CHILD_EXP_NAME = "child-exp000"
# Cell 7-8 を実行

# 2回目
CHILD_EXP_NAME = "child-exp001"
# Cell 6 を変更して Cell 7-8 を実行

# 3回目
CHILD_EXP_NAME = "child-exp002"
# ...
```

---

## 📊 訓練スクリプト（train.py）の要件

新 EXP を作成する際、以下の要件を満たす `train.py` を実装してください：

### 入力
- **第1引数**: `--config` で設定ファイル path を指定
- **設定ファイル形式**: YAML
  ```yaml
  model:
    type: "lightgbm"
    params:
      num_leaves: 100
      learning_rate: 0.05
  cv:
    n_splits: 5
    stratify: true
  training:
    epochs: 500
  target:
    name: "Churn"
  ```

### 出力
必ず `outputs/{child_exp_name}/` ディレクトリを作成し、以下を保存：

1. **config.yaml** - 実行時の設定ファイル（コピー）
2. **oof_predictions.csv**
   ```
   index, fold, target, prediction
   0, 0, 1, 0.823
   1, 0, 0, 0.055
   2, 1, 1, 0.741
   ...
   ```
3. **results.json**
   ```json
   {
     "cv_auc": 0.856,
     "cv_logloss": 0.312,
     "gap_pubpriv": -0.012,
     "best_threshold": 0.5
   }
   ```

### 重要な実装ポイント

```python
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # 設定を読込
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 訓練
    train_df = pd.read_csv("data/train.csv")
    
    # CV fold を作成
    skf = StratifiedKFold(
        n_splits=config['cv']['n_splits'],
        shuffle=True,
        random_state=42
    )
    
    # フォール毎に訓練
    oof_predictions = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[config['target']['name']])):
        # 訓練
        # 検証
        # oof_predictions に追加
        pass
    
    # 結果を保存
    oof_df = pd.DataFrame(oof_predictions)
    oof_df.to_csv(f"outputs/{child_exp_name}/oof_predictions.csv", index=False)
    
    results = {
        "cv_auc": ...,
        "cv_logloss": ...,
        ...
    }
    with open(f"outputs/{child_exp_name}/results.json", "w") as f:
        json.dump(results, f)
    
    print(f"CV AUC: {results['cv_auc']:.4f}")

if __name__ == "__main__":
    main()
```

---

## 📝 config.yaml テンプレート

`EXP001/config/child-exp000.yaml`:

```yaml
# Model Configuration
model:
  type: "lightgbm"  # or "xgboost", "catboost", "neural_net"
  params:
    num_leaves: 100
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    verbose: -1

# Cross-Validation Configuration
cv:
  n_splits: 5
  stratify: true  # クラス分布を保持
  random_state: 42
  method: "stratified_kfold"  # "stratified_kfold", "group_kfold", "time_split"

# Training Configuration
training:
  num_iterations: 500  # LightGBM: num_leaves iterations
  early_stopping_rounds: 50  # 改善なし{N}回で停止
  eval_metric: "auc"  # "auc", "binary_logloss"
  verbose_eval: 20

# Class Weight (不均衡対策)
class_weight:
  enabled: true
  method: "balanced"  # "balanced" or "manual"
  # manual 場合：
  # weight: {0: 1.0, 1: 2.0}

# Target Configuration
target:
  name: "Churn"  # ターゲット列名
  type: "binary"  # "binary" or "multiclass"

# Data Configuration
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  random_seed: 42

# Feature Configuration (オプション)
features:
  use_categorical: true
  use_numerical: true
  encoding: "label"  # "label", "onehot"
```

---

## 🔍 トラブルシューティング

### Q: "config file not found"
A: execute_train.ipynb Cell 6 で EXP_NAME と CHILD_EXP_NAME が正しいか確認。ファイルパスが `EXP/{EXP_NAME}/config/{CHILD_EXP_NAME}.yaml` になっているか確認。

### Q: "Google Drive マウント失敗"
A: Cell 2 で認証が要求されます。Colab 画面に従って認可してください。

### Q: "out of memory"
A: LightGBM の `num_leaves` や `bagging_freq` を削減。または `num_iterations` を減らしてデバッグ。

### Q: OOF 予測がおかしい
A: `oof_predictions.csv` フォーマットを確認。columns は `[index, fold, target, prediction]` であるべき。

---

## 📊 実行結果の確認

execute_train.ipynb Cell 8 で、実行結果を確認：

```
CV AUC: 0.856
CV Logloss: 0.312
Gap (Public-Private estimate): TBD

Top 3 важные features:
1. monthly_charge: 0.234
2. tenure: 0.198
3. contract_length: 0.156
```

**このスコアを [EXP/EXP_SUMMARY.md](EXP/EXP_SUMMARY.md) に記入してください**。

---

## 🚀 新 EXP 作成時

新しいコード変更が必要な場合：

```bash
# ローカルで実行（または Colab では手動で以下を実行）

# 1. ディレクトリ作成
mkdir -p EXP/EXP002/config EXP/EXP002/outputs

# 2. テンプレート作成
cp EXP/EXP001/train.py EXP/EXP002/train.py
cp EXP/EXP001/infer.py EXP/EXP002/infer.py
cp EXP/EXP001/config/child-exp000.yaml EXP/EXP002/config/child-exp000.yaml

# 3. train.py, infer.py を編集
# 4. 新設定ファイルを作成
```

その後 execute_train.ipynb の Cell 6 で：
```python
EXP_NAME = "EXP002"
CHILD_EXP_NAME = "child-exp000"
```

---

**最終更新**: 2026-03-18
