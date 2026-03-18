# AIエージェント向けガイドライン - Predict Customer Churn

このドキュメントは Claude/Codex などの AI エージェントが、このプロジェクトでコード実装や実験設計を行う際のガイドラインです。このファイルを必ず参照してから実装してください。

---

## 🎯 プロジェクトの目的

Kaggleの「Predict Customer Churn」コンペにおいて、顧客離脱を高精度に予測するモデルを開発すること。
**評価指標**：通常は ROC-AUC または Logloss（コンペ要項確認必須）

---

## 📋 タスク概要

### データセット
- **訓練データ**：`data/train.csv` (顧客情報 + 離脱/残存ラベル)
- **テストデータ**：`data/test.csv` (顧客情報のみ)
- **ターゲット**：二項分類 (Churn: 1 or 0)

### ビジネス要件（重要）
- 通常、False Negativeコスト > False Positiveコスト
- つまり「実際に離脱する人を見逃す」が最悪

### 評価フロー
1. フォールド分割（層化K-Fold推奨）
2. OOF（Out-of-Fold）予測生成
3. CV スコア計算（正式な評価指標に合わせたカスタム関数）
4. 最終サブミッション

---

## ⚙️ 実験体系

### EXP 管理ルール

#### EXP を新規作成する条件（コード変更が必要）
- ✅ 新しいモデルアーキテクチャ
- ✅ 新しいアルゴリズム説 (Gradient Boosting → Neural Net など)
- ✅ 新しい特徴エンジニアリング手法
- ✅ 新しい解釈可能性手法の追加
- ✅ パイプライン構造の大きな変更

#### child-exp を作成する条件（ハイパラ調整のみ）
- ✅ learning rate 変更
- ✅ batch size 変更
- ✅ epochs 変更
- ✅ optimizer / scheduler 変更
- ✅ regularization 強度変更 (dropout, L1/L2)
- ✅ class weights / threshold 調整
- ✅ augmentation strategy 変更

### 命名規則
```
EXP{N}/                  # N = 実験番号（001から始める）
├── train.py            # この EXP固有の訓練スクリプト
├── infer.py            # 推論スクリプト
└── config/
    ├── child-exp000.yaml   # child-exp001 の設定
    ├── child-exp001.yaml
    └── ...
```

### ファイル互換性ルール（重要！）
新しい EXP で以下の行為は **禁止**：
- ❌ `train.csv` のカラム順序を勝手に変更
- ❌ 前の EXP との互換性を無視した config キー変更
- ❌ OOF 出力フォーマットの変更

新しい EXP 追加時：
- ✅ 前の EXP の config 形式を尊重
- ✅ 新キーは「新EXPのみで使う」と明記
- ✅ 旧キー削除時は全 child-exp に通知

---

## 🔴 絶対に避けるべきパターン（失敗リスト）

### 評価指標関連
❌ **指標計算の誤り**
- per-target average してから weight を掛ける ← 間違い
- global sum（全予測点統合）してから weight を掛ける ← 正解
- 参考：csiro 実装では EXP060で R²計算の大バグを修正（17%改善）

❌ **フォールド漏洩**
- テストセットに train 期間のデータが混ざる
- バック填充（future leakage）の発生
- 層化なし分割 → 不均衡フォールド

### モデル関連
❌ **過度な正則化**
- dropout > 0.2 は通常過剰
- L2 weight_decay > 0.01 はアンダーフィット傾向
- Early stopping が厳しすぎる（内部CV で評価）

❌ **複雑化の陥穽**
- Auxiliary tasks（補助分類タスク）の追加 → LB低下
- Hand-crafted features の過剰準備 → 学習済モデルでは冗長
- 複雑な post-processing → テスト分布でルール化失敗

❌ **推論ミスマッチ**
- 訓練時：データ前処理なし → 推論時：preprocessing あり（逆も）
- test-time augmentation の実装ミス
- 未知のカテゴリ値の処理忘れ

### 特徴エンジニアリング
❌ **外部特徴の不適切な利用**
- 外部データセット追加 → test LB 低下（分布が異なる）
- メタデータ利用 → train-test 不一致

❌ **過度な augmentation**
- Mixup 強度大 →  LB低下
- Color jitter 強度大 → 自然な多様性を損失
- Random erasure → 重要な情報消失

### 推論パイプライン
❌ **アンサンブルの非効率**
- 同じモデルを繰り返し ensemble
- アンサンブル重みを OOF で最適化 → test で失敗（過学習）

❌ **Threshold チューニング**
- OOF で threshold を最適化 → test で失敗
- Threshold は validation/public LB では固定

---

## ✅ 成功パターン集

### モデル選択
✅ **勾配ブースティング**（XGBoost, LightGBM）
- 表型データに強い
- 特徴の重要度が明確
- チューニングで高精度

✅ **アンサンブル戦略**
- 複数モデルの組み合わせ（RF + GB + Linear）
- 異なるシード（不確実性削減）
- 異なる前処理パイプライン

### 特徴エンジニアリング
✅ **ドメイン知識の活用**
- 業務知識から有意義な特徴を生成
- 相互作用項（contract duration × monthly charge など）
- パターンに基づく特徴（契約形態別の平均チャーン率など）

✅ **統計的特徴**
- 顧客セグメント別の統計量
- 時系列差分（available if 期間情報あり）

### 訓練フロー
✅ **2段階訓練戦略**（Neural Netの場合）
- Stage 1：主要パラメータで広く探索（epochs=10）
- Stage 2：最適パラメータで長期訓練（epochs=30+）

✅ **適応的学習率スケジューリング**
- Cosine Annealing vs Linear Decay（両方試す）
- Warm-up phase で安定化

✅ **クラス不均衡対策**
```python
# 推奨：class_weight='balanced' または手動設定
class_weight = {0: 1.0, 1: n_neg / n_pos}
```

### 評価戦略
✅ **正確な CV 実装**
```python
# キーポイント：
# 1. 層化K-Fold（ターゲット分布を保持）
# 2. グループ分割（時間的/空間的漏洩防止）
# 3. 正式評価指標のカスタム関数実装
# 4. CV一貫性チェック（folds間のばらつき分析）
```

✅ **CV-LB 相関分析**
- CV スコアが LB と相関 → 信頼できる
- CV-LB ギャップが大きい → test分布が異なる可能性

---

## 🔧 実装時の重要チェックリスト

### train.py 作成時
- [ ] 入力：config ファイルから全パラメータを読み込む
- [ ] 出力：`outputs/{child-exp_name}/` に以下を生成
  - `config.yaml` （実行設定のコピー）
  - `oof_predictions.csv` （フォール毎の OOF予測）
  - `results.json` （CV/AUC等の結果）
- [ ] フォールド分割は層化 K-Fold、seed固定
- [ ] Random seed を全所で固定（numpy, torch 等）
- [ ] GPU/CPU フロートの精度チェック（float32 推奨）

### config.yaml 作成時
```yaml
# 必須フィールド
model:
  type: "lightgbm"  # or "xgboost", "neural_net"
  params:
    num_leaves: 100
    learning_rate: 0.05
    num_boost_rounds: 500

cv:
  n_splits: 5
  stratify: true  # ターゲット分布を保持
  random_state: 42

training:
  epochs: 10  # boosting or neural net iterations
  early_stopping_rounds: 50

# オプション
augmentation:
  enabled: false  # 表型データでは通常不要
  
target:
  name: "Churn"  # ターゲットカラム名
  type: "binary"  # "binary" or "multiclass"
```

### infer.py 作成時
- [ ] test.csv から同じ特徴を抽出
- [ ] 訓練時と同じ前処理順序
- [ ] 全フォールドの予測を平均化
- [ ] 出力形式が `sample_submission.csv` 準拠
- [ ] 予測確率は [0, 1] 範囲（sigmoid/softmax適用済）

### テストコード（推奨）
```python
# 訓練-推論の互換性チェック
def test_train_infer_compatibility():
    # 訓練データで学習
    train_preds = train_main(...)
    
    # インファレンスで同じ訓練データを予測
    infer_preds = infer_main(...)
    
    # 精度は±0.001以内
    assert np.allclose(train_preds, infer_preds, atol=1e-3)
```

---

## 📊 実験記録の方法

実験完了後、以下の情報を **EXP/EXP_SUMMARY.md** に記入：

```markdown
## EXP001 - Baseline LightGBM

**主な変更**: LightGBM baseline、標準前処理

**設定**:
- Model: LightGBM
- CV折数: 5
- Child experiments: 3 (child-exp000, 001, 002)

**結果**:
| child-exp | LR   | num_leaves | CV AUC | Public LB | Private LB | Note |
|-----------|------|-----------|--------|-----------|------------|------|
| 000       | 0.05 | 100       | 0.856  | 0.844     | -          | baseline |
| 001       | 0.03 | 150       | 0.858  | 0.846     | -          | better |
| 002       | 0.07 | 80        | 0.853  | 0.840     | -          | worse  |

**学習内容**:
- LR 0.03 が良好
- num_leaves 増加は過学習傾向
- CV-LB ギャップ: -0.012（良好）

**次のステップ**: EXP002 で feature engineering を追加
```

---

## 🚀 新実験立案のテンプレート

新 EXP または child-exp を立案する前に：

1. **EXP_SUMMARY をレビュー**
   - 過去のスコア/失敗を確認
   - 似た設定の結果を参照

2. **ガードレール確認**
   - このセクション（_CLAUDE.md）の「避けるべきパターン」をチェック

3. **仮説設定**
   ```
   - "Feature X が重要と考える理由"
   - "モデル Y を選んだ理由"
   - "期待される改善度"
   ```

4. **実装 & 実行**
   - execute_train.ipynb で実行
   - 結果を记录

5. **失敗時の分析**
   - "なぜ低下したか" を記入
   - 次回への教訓を記入

---

## 📝 よくある質問（Q&A）

**Q: 新 EXP を作るべきか、child-exp で十分か？**  
A: config ファイルで表現できる変更 = child-exp。コード改変が必要 = EXP。

**Q: ハイパラチューニングは自動で？手動で？**  
A: 手動推奨（小規模サーチ, 解釈可能性）。Optuna等は中大規模コンペで。

**Q: 複数モデルを ensemble する場合？**  
A: 各モデルを別 EXP として開発。最後に `palla-customer-churn-inference.ipynb` で統合。

**Q: Cross-validation fold 数は？**  
A: 小規模データ（<5000） = 5-fold or 10-fold。中規模（5K-50K） = 5-fold。大規模（>50K） = 3-fold。

**Q: Early stopping は必要か？**  
A: Boosting tree = yes。Linear model = no。Neural Net = yes。

---

## 🔗 関連ファイル

- **_AGENTS.md** - Codex 用の簡潔版
- **README.md** - プロジェクト概要
- **EXP/EXP_SUMMARY.md** - 実験履歴＆学習
- **docs/OVERVIEW.md** - 手法概要
- **docs/DATASET.md** - データセット詳細

---

**最終更新**：2026-03-18
