# EXP_SUMMARY - 実験管理＆学習記録

このファイルはすべての実験（成功・失敗）の記録です。AI エージェントはこれを参照して、次の実験を計画してください。

---

## 📊 実験一覧（更新順）

| EXP | Type | Focus | Best CV | LB | Status | Notes |
|-----|------|-------|---------|-----|--------|-------|
| EXP001 | LightGBM | Baseline | TBD | TBD | Not started | 初期実装 |

---

## EXP001 - Baseline LightGBM

**Status**: 未実装

**説明**: 顧客離脱の LightGBM baseline。標準的なデータ前処理と CV フロー。

**Child Experiments計画**:
- child-exp000：デフォルト設定（num_leaves=100, lr=0.05）
- child-exp001：調整版（微調整）
- child-exp002：調整版（別方針）

**期待される改善**:
- LightGBM は表型データに強い
- class weight で不均衡に対応
- CV-LB gap を観察

**失敗しやすいポイント**:
- フォールド漏洩（datetime分割が必要な場合）
- クラス不均衡への対応不足
- ハイパラ過最適化（OOFでオーバーチューニング）

**次のステップ**:
- EXP001 の結果を確認 → 
- EXP002で feature engineering 追加 →
- EXP003でアンサンブル検討

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

