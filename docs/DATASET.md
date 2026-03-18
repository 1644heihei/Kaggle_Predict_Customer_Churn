# Dataset Overview

## データセット概要

### タスク
顧客の離脱（Churn）を予測する二項分類問題。

### ファイル
- `train.csv` - 訓練データ（顧客情報 + 離脱ラベル）
- `test.csv` - テストデータ（顧客情報のみ）
- `sample_submission.csv` - 提出形式：`Id, Target（0 or 1）`

### サンプル数
- Train: TBD（確認時に更新）
- Test: TBD

### ターゲット分布
- Churn = 0: TBD%
- Churn = 1: TBD%

（**実データ確認時に更新してください**）

---

## 特徴カラム（推定）

一般的な顧客離脱データセットの特徴：
- **顧客ID**: customer_id (categorical)
- **契約情報**: contract_duration, contract_type (categorical)
- **使用量**: monthly_charge, total_charges (numerical)
- **サービス**: internet_type, phone_service, streaming (categorical)
- **地域**: state, region (categorical)
- **人口統計**: age, gender (categorical)

**実のカラム確認は `exploratory_data_analysis` ノートブックを参照**。

---

## 前処理の考慮事項

1. **欠損値**: 存在するかどうかを確認
2. **外れ値**: charge系の異常値チェック
3. **カテゴリカル特徴**: One-hot encoding or label encoding
4. **スケーリング**: 必要に応じて StandardScaler/MinMaxScaler
5. **クラス不均衡**: クラスウェイト調整

---

## 便利なリソース

- `notebooks/eda.ipynb` - EDA結果
- `EXP001/train.py` - データ読込・前処理の参考実装

---

**最終更新**: 2026-03-18（初期化）
