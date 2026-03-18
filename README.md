# Kaggle: Predict Customer Churn

## プロジェクト概要
顧客離脱予測（Customer Churn Prediction）を目的としたKaggleコンペティション用リポジトリです。
このプロジェクトは、AIエージェント（Claude/Codex）を活用した体系的な開発パイプラインを採用しています。

## ディレクトリ構成

```
/
├── _CLAUDE.md                          # AIエージェント向け詳細指示書（430+行）
├── _AGENTS.md                          # コンパクト版指示書
├── README.md                           # このファイル
├── EXECUTE_TRAIN_README.md             # 訓練実行方法
├── execute_train.ipynb                 # Colab用訓練入口
├── palla-customer-churn-inference.ipynb # Kaggle提出用推論スクリプト
│
├── docs/
│   ├── DATASET.md                      # データセット詳細
│   ├── OVERVIEW.md                     # 手法概要
│   ├── papers/                         # 参考論文
│   └── Idea_Research/                  # 研究ドキュメント
│
├── EXP/                                # 実験管理フォルダ
│   ├── EXP_SUMMARY.md                  # 全実験の記録＆学習
│   ├── EXP001/                         # 主要実験001
│   │   ├── train.py                    # 訓練スクリプト
│   │   ├── infer.py                    # 推論スクリプト
│   │   ├── infer_fast.py               # 高速推論版（オプション）
│   │   ├── config/
│   │   │   ├── child-exp000.yaml       # ハイパラ設定-000
│   │   │   ├── child-exp001.yaml       # ハイパラ設定-001
│   │   │   └── child-exp002.yaml       # ハイパラ設定-002
│   │   └── outputs/                    # 結果出力
│   │       ├── child-exp000/
│   │       │   ├── config.yaml
│   │       │   ├── oof_predictions.csv
│   │       │   └── results.json
│   │       └── ...
│   ├── EXP002/
│   └── ...
│
├── outputs/
│   ├── Analysis/                       # CV/LB分析結果
│   └── submissions/                    # サブミッション結果
│
├── data/
│   ├── train.csv                       # 訓練データ
│   ├── test.csv                        # テストデータ
│   ├── sample_submission.csv           # サンプルサブミッション
│   └── submission.csv                  # 最終サブミッション
│
└── notebooks/
    ├── eda.ipynb                       # EDA用ノートブック
    └── model.ipynb                     # モデル開発用
```

## 開発プロセス

### 実験体系
このプロジェクトは**2段階の実験管理**を採用しています：

1. **EXP{N}**：コード変更が必要な場合に作成
   - モデルアーキテクチャの変更
   - 新しいアルゴリズムの試行
   - パイプライン構造の改善

2. **child-exp{N}**：ハイパーパラメータ調整のみ
   - LR、batch size、epochs等の調整
   - loss weights、regularization等の変更
   - 同じ`train.py`を複数の設定で実行

### 実験の進め方

1. [EXP/EXP_SUMMARY.md](EXP/EXP_SUMMARY.md) で過去の結果と失敗を確認
2. `_CLAUDE.md` のガードレールを参照して避けるべきパターンを把握
3. 新EXPまたはchild-expを立案
4. `execute_train.ipynb` で訓練実行
5. 結果を EXP_SUMMARY.md に記録

## AIエージェント向け指示

### エージェント向け重要文書
- **[_CLAUDE.md](_CLAUDE.md)**：詳細なガードレール、失敗パターン、評価関数
- **[_AGENTS.md](_AGENTS.md)**：簡潔版（Codex用）

エージェントは常に以下を参照：
- 「このパターンは LB に悪影響」という黒リスト
- 「成功した戦略」という白リスト
- 「正しい評価指標の計算方法」
- 「一般的なバグ」と対策

## ローカル開発フロー

```bash
# 環境構築
pip install -r requirements.txt

# (オプション) 新実験のテンプレート作成
python scripts/create_exp.py --exp_num 2 --name "New Experiment"

# (実際の訓練はColab)
# execute_train.ipynb で EXP + child-exp を指定して実行
```

## Kaggle提出フロー

1. `palla-customer-churn-inference.ipynb` を Kaggle Notebook 内で実行
2. 自動的に `submission.csv` を生成
3. 手動で提出

## 参考資料

- **開発思想**：[AI-driven Kaggle gold medal solution](https://zenn.dev/chiman/articles/...) （参考実装）
- **モデルアーキテクチャ**：[docs/OVERVIEW.md](docs/OVERVIEW.md)
- **データセット詳細**：[docs/DATASET.md](docs/DATASET.md)
- **実験履歴**：[EXP/EXP_SUMMARY.md](EXP/EXP_SUMMARY.md)

## ライセンス
Kaggle Competition用

## 最終更新
2026-03-18
