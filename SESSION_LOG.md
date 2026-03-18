# セッションプログレス - Kaggle Predict Customer Churn

## ✅ 完了したタスク

### 1. ディレクトリ構成の整備 (2026-03-18)
- csiro-biomassと同じ体系を実装
- EXP/CHILD-EXPの階層管理システム
- AIエージェント向けガイドライン（_CLAUDE.md, _AGENTS.md）

### 2. Jupyterノートブック作成 (2026-03-18)
- execute_train.ipynb - Colab訓練入口
- palla-customer-churn-inference.ipynb - Kaggle推論スクリプト

### 3. 初期訓練テスト (2026-03-18)
- **EXP001 - Baseline LightGBM** ✅ 成功
  - データ: 594K行 → デモで50K sample使用
  - モデル: LightGBM (num_leaves=50, lr=0.1)
  - CV方法: 3-fold StratifiedKFold
  - **結果: CV AUC = 0.9134, Logloss = 0.6926**
  - 出力: oof_predictions.csv, test_predictions.csv, results.json

## 📊 実験記録

### EXP001 - child-exp000 (Baseline)

| 項目 | 値 |
|------|-----|
| Model | LightGBM |
| Sample Size | 50,000 |
| CV Folds | 3 |
| CV AUC | 0.9134 |
| CV Logloss | 0.6926 |
| Status | ✅ Done |
| Date | 2026-03-18 |

## 🎯 次のステップ

1. **フルデータで訓練** - 50K → 全594K
2. **ハイパラチューニング** - child-exp001, exp002, ...
3. **複数モデルアンサンブル** - EXP002: XGBoost, EXP003: NN等
4. **Feature Engineering** - target encoding, interaction terms等

## 🔧 開発パイプラインの検証

✅ データ読込・前処理  
✅ CV実装 (StratifiedKFold)  
✅ LightGBMモデル実装  
✅ 結果保存 (JSON, CSV)  
✅ 再現性確保 (seed固定)  
✅ Windows環境対応 (encoding指定)  

## 📝 Notes

- データの不均衡: 0が77%, 1が23% → class_weight='balanced'推奨
- AUCが高い(0.91) → 強いシグナルあり
- デモ版は高速化のため50Kサンプリング、本訓練では全データ使用推奨
