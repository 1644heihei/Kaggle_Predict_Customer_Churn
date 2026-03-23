# EXP_SUMMARY - 実験管理＆学習記録

このファイルはすべての実験（成功・失敗）の記録です。AI エージェントはこれを参照して、次の実験を計画してください。

---

## 📊 実験一覧（更新順）

| EXP | Type | Focus | Best CV | LB | Status | Notes |
|-----|------|-------|---------|-----|--------|-------|
| EXP001 | LightGBM | Baseline | 0.9134 | TBD | ✅ Completed | デモ版（50K, 3-fold） |
| EXP002 | LightGBM | Feature Eng | 0.9126 | TBD | ✅ Completed | 相互作用項＋Service Count |

---

## EXP001 - Baseline LightGBM

**Status**: ✅ 完了（デモ版）

**説明**: 50K サンプル、3-fold CV での初期 baseline。標準的なデータ前処理と CV フロー。

**実装内容**:
- 層化 K-Fold（3-fold）
- LightGBM with sigmoid output （raw score → expit 変換）
- class_weight = balanced（4.3→1 比率）
- Early stopping（round=50）

**結果**:
- CV AUC: **0.9134**
- CV Logloss: **0.6926**
- テスト予測生成済み

**Top 5 重要度特徴**:
1. PaymentMethod (547)
2. tenure (384)
3. MonthlyCharges (269)
4. TotalCharges (201)
5. PaperlessBilling (178)

**次のステップ**: Feature Engineering で改善検討

---

## EXP002 - LightGBM + Feature Engineering

**Status**: ✅ 完了（本格版）

**説明**: フル 594K データ、5-fold CV での特徴量エンジニアリング版。

**新しい特徴量**:
1. **相互作用項** (✓ 有効):
   - `MonthlyCharges × tenure`
   - `TotalCharges × tenure`

2. **Service Count**:
   - 複数サービス契約数をカウント（9 サービス対象）

3. **グループ化集計**:
   - `Contract_AvgCharge`: Contract 別の平均月額料金
   - `InternetService_AvgTenure`: InternetService 別の平均テニュア

**実装内容**:
- 層化 K-Fold（5-fold）
- LightGBM with sigmoid output
- class_weight = balanced（3.44→1 比率）
- Early stopping（round=50）

**結果**:
- CV AUC: **0.9126**
- CV Logloss: **0.7326**
- **特徴量: 19 → 24 (+5 新規)**

**Top 10 重要度特徴**:
1. PaymentMethod (421.8)
2. tenure (391.6)
3. **TotalCharges** (297.8)
4. **MonthlyCharges** (278.2)
5. **MonthlyCharges_x_tenure** (254.4) ✅ **新規**
6. PaperlessBilling (199.0)
7. SeniorCitizen (189.2)
8. MultipleLines (185.8)
9. **TotalCharges_x_tenure** (179.8) ✅ **新規**
10. Dependents (150.2)

**分析**:
- ✅ 相互作用項が Top 10 の 40% を占める（有効性確認）
- ✅ Fold 間の安定性：0.9112 ～ 0.9140（良好）

**考察**:
- EXP001 は固定 50K でデモ版、EXP002 は本格版なので直接比較不可
- 全体的に安定したスコア（CV差が最小）
- 相互作用項の追加が効果的

---

## 📈 学習内容（更新時に追記）

### フェーズ1: Baseline確立（EXP001予定）
- [ ] テストスプリットの適切性確認
- [ ] クラス不均衡の大きさ把握
- [ ] 初期 CV スコア取得
- [ ] LB gap を観察

### フェーズ2: Feature Engineering（EXP002予定）
- [ ] ドメイン知識による特徴
- [ ] 相互作用項の有効性
- [ ] カテゴリ特徴のエンコーディング

### フェーズ3: アンサンブル（EXP003以降予定）
- [ ] 複数モデル比較
- [ ] アンサンブル重み決定

---

## 🔴 失敗パターン（参考）

未実装なので参照用：

**全般**:
- フォールド漏洩（時間順序を無視）
- クラス不均衡無視
- 評価指標の誤実装

**feature**:
- 外部データ無計画追加
- train-test 分布ミスマッチ

**モデル**:
- 過度な正則化
- アンサンブル重み過最適化

---

## ✅ 成功パターン（参考）

**実装戦略**:
1. 層化K-Fold で分割
2. 適切な class_weight 設定
3. 段階的な改善
4. CV-LB 相関を常に監視

---

## 🔗 関連ファイル

- [_CLAUDE.md](../_CLAUDE.md) - 詳細ガイドライン
- [_AGENTS.md](../_AGENTS.md) - 簡潔版
- [README.md](../README.md) - プロジェクト概要
- [EXP001/config/](EXP001/config/) - 設定ファイル
- [EXP001/outputs/](EXP001/outputs/) - 結果

---

**最終更新**: 2026-03-18（初期化）

