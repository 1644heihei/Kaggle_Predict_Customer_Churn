# 実験サマリー（Experiments Overview）

## 📊 実験一覧

| # | モデル | 説明 | CV AUC | 状態 | 用途 |
|---|--------|------|--------|------|------|
| **EXP001** | LightGBM | ベースラインモデル（固定パラメータ） | 0.9134 | ✓ 完了 | 参考 |
| **EXP002** | LightGBM + FE | 特徴エンジニアリング追加版 | 0.9126 | ✓ 完了 | 参考 |
| **EXP003** | XGBoost | XGBoostベース（固定パラメータ） | 0.9159 | ✓ 完了 | ⭐ ブレンディング |
| **EXP004** | XGBoost + Optuna | ハイパーパラメータ最適化版 | **0.9164** | ✓ 完了 | ⭐⭐ 最高精度 |
| **EXP005** | CatBoost | GPU/CPU試行版（性能低） | 0.8977 | ✓ 完了 | スキップ |
| **EXP006** | XGBoost + 拡張FE | ログ変換、多項式、相互作用項拡充 | 未実行 | 🔨 開発 | 将来の多様性 |
| **EXP007** | CatBoost | テーブル最適化版 | 0.9159 | ✓ 完了 | 多様性確保 |
| **EXP008** | XGBoost + scale_pos_weight | クラス不均衡対応版 | 0.9159 | ✓ 完了 | 多様性確保 |

---

## 📁 ディレクトリ構造

```
EXP/
├── EXP001/              # LightGBM ベースライン
│   ├── train.py
│   ├── config/
│   │   └── child-exp000.yaml
│   └── outputs/child-exp000/
│
├── EXP002/              # LightGBM + 特徴エンジニアリング
│   ├── train.py
│   ├── config/
│   │   └── child-exp000.yaml
│   └── outputs/child-exp000/
│
├── EXP003/              # XGBoost ベース
│   ├── train.py
│   ├── config/
│   │   └── child-exp000.yaml
│   └── outputs/child-exp000/
│
├── EXP004/🌟             # XGBoost + Optuna（最高精度）
│   ├── train.py
│   ├── config/
│   │   └── child-exp000.yaml
│   └── outputs/child-exp000/
│
├── EXP005/              # CatBoost（参考のみ）
│   ├── train.py
│   ├── config/
│   │   └── child-exp000.yaml
│   └── outputs/child-exp000/
│
├── EXP006/              # XGBoost + 拡張特徴（開発中）
│   ├── train.py
│   └── outputs/child-exp000/
│
├── EXP007/              # CatBoost（多様性）
│   ├── train.py
│   └── outputs/child-exp000/
│
├── EXP008/              # XGBoost 不均衡対応（多様性）
│   ├── train.py
│   └── outputs/child-exp000/
│
└── README.md            # このファイル
```

---

## 🎯  最適なモデル選択

### **単一モデルでスコア最大化**
```
→ EXP004 を選択
  CV AUC: 0.9164（最高）
  Optuna で自動ハイパーパラメータ最適化済み
```

### **ブレンディング（複数モデル統合）**
```
推奨: EXP003(45%) + EXP004(55%)
  - EXP003: XGBoost ベース多様性
  - EXP004: 最高精度

代替案（スケール統一必要）:
  - EXP004(50%) + EXP007(30%) + EXP008(20%)
  - 注意: CatBoost/XGBoost の確率スケールが異なるため
    メタモデル使用時は正規化が必須
```

---

## 📊 各モデルの特徴

### **EXP001: LightGBM ベースライン**
```
目的:     基準モデルの確立
アルゴリズム: LightGBM
パラメータ:  固定（num_leaves=100, lr=0.05）
CV AUC:    0.9134
特徴:     安定性重視、参考値
```

### **EXP002: LightGBM + 特徴エンジニアリング**
```
目的:     特徴抽出の効果測定
特徴:     相互作用項、サービスカウント、グループ集計
CV AUC:    0.9126（低くなった）
判定:     特徴の効果なし or 過学習
```

### **EXP003: XGBoost ベース** ⭐
```
目的:     XGBoost の初期検証
アルゴリズム: XGBoost
パラメータ:   max_depth=6, lr=0.1, subsample=0.8
CV AUC:     0.9159
特徴:       安定版、ブレンディング時の基本
```

### **EXP004: XGBoost + Optuna** ⭐⭐ 最高精度
```
目的:     ハイパーパラメータ最適化
方法:     Optuna TPE サンプラー
探索:     max_depth, learning_rate など自動調整
CV AUC:    0.9164（+0.0005 改善）
判定:     本コンペ最高モデル
```

### **EXP005: CatBoost**
```
目的:     異なるアルゴリズム検証
アルゴリズム: CatBoost
問題:     GPU/CPU 互換性問題で性能低下
CV AUC:    0.8977
判定:     スキップ推奨
```

### **EXP006: XGBoost + 拡張特徴**
```
目的:     高度な特徴エンジニアリング
新特徴:    ログ変換、多項式、比率、ビニング
期待CV:    0.9168+
状態:     開発中
```

### **EXP007: CatBoost テーブル最適化** ✓ 多様性
```
目的:     アルゴリズム多様性の確保
特徴:     テーブルデータ特化、metric='AUC'明示
CV AUC:    0.9159
用途:     ブレンディング時の多様性
```

### **EXP008: XGBoost クラス不均衡対応** ✓ 多様性
```
目的:     クラス不均衡への対応
工夫:     scale_pos_weight = 3.4403 設定
背景:     Churn Yes: 22.5%, No: 77.5% の不均衡
CV AUC:    0.9159
用途:     ブレンディング時の異なる視点
```

---

## 🚀 実行手順

### **1. 単独モデル検証**
```bash
# EXP004（最高精度）で提出テスト
python EXP/EXP004/train.py
→ CV AUC 0.9164 を確認
```

### **2. ブレンディング版**
```bash
# EXP003(45%) + EXP004(55%)
python simple_blend_safe.py
→ submission_blend_34_45_55.csv を生成
```

### **3. スケーリング対応版（高度）**
```bash
# すべてを正規化してメタモデル統合
python stacking_3models.py
→ 注意: 予測値スケールの統一が必須
```

---

## ⚠️ よくあるトラブル

### **問題: メタモデルスタッキングで精度が下がった**
```
原因: CatBoost と XGBoost の確率キャリブレーションが異なる
解決: 予測値を正規化してから統合
     または単純平均ブレンディングに戻す
```

### **問題: EXP006が遅い**
```
原因: ハイパーパラメータ探索で時間がかかる
解決: num_boost_round を 500→200 に削減
```

---

## 📈 期待スコア転移

```
現在:  LB 0.91388 (EXP003: 45% + EXP004: 55%)
目標:  LB 0.917+ (追加改善)

改善戦略:
  1. EXP004 単体      → 0.9160～0.9165
  2. EXP003/004混合   → 0.9138～0.9140（守り）
  3. EXP004 + EXP007  → 0.9145～0.9155（多様性試行）
```

---

## 🔗 関連ファイル

```
プロジェクトルート/
├── data/
│   ├── train.csv
│   └── test.csv
│
├── EXP/                    ← 実験フォルダ
│   ├── EXP001～EXP008/
│   └── README.md           ← このファイル
│
├── stacking_3models.py     ← 3モデル統合スタッキング
├── simple_blend_safe.py    ← 安全なシンプルブレンディング
└── grid_search_blend.py    ← 重み最適化グリッドサーチ
```

---

## 📝 更新履歴

- **2026/03/31**: EXP001～EXP008 完成、README 作成
- EXP004 が現在最高精度（CV AUC 0.9164）
- EXP007, EXP008 でアルゴリズム多様性を確保
- スタッキング試行 → スケーリング問題で単純ブレンディングに回帰
