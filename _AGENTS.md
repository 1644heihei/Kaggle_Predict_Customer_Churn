# AIエージェント向けサマリー - Predict Customer Churn

**詳細は _CLAUDE.md を参照**。ここは要点のみ。

---

## 🎯 タスク
顧客離脱（二項分類）をモデル化。AUC or Logloss が評価指標。

---

## 📋 ディレクトリ構成
```
EXP/
├── EXP_SUMMARY.md
├── EXP001/config/child-exp{N}.yaml
├── EXP001/train.py, infer.py, outputs/
├── EXP002/...
```

## ⚙️ EXP vs child-exp

| 条件            | 作成物       |
|-----------------|-------------|
| config 変更のみ  | child-exp   |
| コード変更      | 新 EXP      |

## 🔴 禁止パターン（重要！）

```
❌ 評価指標の計算誤り
❌ フォールド漏洩
❌ 訓練-推論ミスマッチ
❌ 過度な正則化/augmentation
❌ OOFで threshold チューニング
❌ 外部データの無計画な追加
```

## ✅ 推奨パターン

```
✅ 表型データ → LightGBM/XGBoost
✅ 層化K-Fold（クラス分布保持）
✅ アンサンブル（複数モデル）
✅ ドメイン知識による特徴生成
✅ CV-LB 相関分析
✅ 段階的改善（小さい変更を積み重ねる）
```

## 📊 実装チェックリスト

**train.py**:
- [ ] config から全パラメータ読込
- [ ] outputs/{child-exp}/に config.yaml, oof_predictions.csv, results.json を出力
- [ ] フォールド分割は層化K-Fold
- [ ] Random seed 固定

**infer.py**:
- [ ] test データの特徴抽出（訓練時と同じ方法）
- [ ] 全フォールド予測を平均化
- [ ] sample_submission.csv 形式で出力

**config.yaml**:
```yaml
model:
  type: "lightgbm"
  params: {...}
cv:
  n_splits: 5
  stratify: true
training:
  epochs: 500
target:
  name: "Churn"
  type: "binary"
```

## 🚀 新実験前のステップ

1. EXP/EXP_SUMMARY.md を読む
2. _CLAUDE.md の ❌ パターンを確認
3. 仮説を設定
4. 実装＆実行
5. EXP_SUMMARY に記入（成功/失敗問わず）

## 📝 よくある質問

**新EXPを作るべき？** → コード変更が必要なら yes。  
**ハイパラチューニングは自動？** → 手動推奨（小規模）。  
**複数モデル ensemble？** → 別EXPで開発 → 最後に統合。  
**CV-LB ギャップが大きい？** → test分布が異なる可能性。  

---

**詳細は _CLAUDE.md を参照**

**最終更新**: 2026-03-18
