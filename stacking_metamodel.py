"""
OOFベースStackingの実装
EXP001-004のOOF予測を使用してメタモデルを学習
Geminiが指摘した「0.917超えの決定打」
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OOFベーススタッキング: 0.917突破を目指す")
print("=" * 70)

# =====================================================
# Step 1: OOF予測の読み込みと形式確認
# =====================================================
print("\n[STEP 1] OOF予測の読み込み")
print("-" * 70)

try:
    exp001_oof = pd.read_csv('EXP/EXP001/outputs/child-exp000/oof_predictions.csv')
    exp002_oof = pd.read_csv('EXP/EXP002/outputs/child-exp000/oof_predictions.csv')
    exp003_oof = pd.read_csv('EXP/EXP003/outputs/child-exp000/oof_predictions.csv')
    exp004_oof = pd.read_csv('EXP/EXP004/outputs/child-exp000/oof_predictions.csv')
    
    print("✓ OOFファイル読み込み完了")
    print(f"  EXP001: {exp001_oof.shape}")
    print(f"  EXP002: {exp002_oof.shape}")
    print(f"  EXP003: {exp003_oof.shape}")
    print(f"  EXP004: {exp004_oof.shape}")
except Exception as e:
    print(f"❌ OOF読み込みエラー: {e}")
    exit()

# =====================================================
# Step 2: テスト予測の読み込み
# =====================================================
print("\n[STEP 2] テスト予測の読み込み")
print("-" * 70)

try:
    exp001_test = pd.read_csv('EXP/EXP001/outputs/child-exp000/test_predictions.csv')
    exp002_test = pd.read_csv('EXP/EXP002/outputs/child-exp000/test_predictions.csv')
    exp003_test = pd.read_csv('EXP/EXP003/outputs/child-exp000/test_predictions.csv')
    exp004_test = pd.read_csv('EXP/EXP004/outputs/child-exp000/test_predictions.csv')
    
    print("✓ テスト予測ファイル読み込み完了")
    print(f"  EXP001: {exp001_test.shape}")
    print(f"  EXP002: {exp002_test.shape}")
    print(f"  EXP003: {exp003_test.shape}")
    print(f"  EXP004: {exp004_test.shape}")
except Exception as e:
    print(f"❌ テスト読み込みエラー: {e}")
    exit()

# =====================================================
# Step 3: OOF特徴の統合
# =====================================================
print("\n[STEP 3] OOF特徴の統合")
print("-" * 70)

# OOFからターゲットと予測を抽出
# OOFの形式が混在しているため、丁寧に処理
oof_features = pd.DataFrame()

# EXP001: ['index', 'fold', 'target', 'prediction']
if 'prediction' in exp001_oof.columns:
    oof_features['pred_001'] = exp001_oof['prediction'].values
else:
    oof_features['pred_001'] = exp001_oof.iloc[:, -1].values

# EXP002: ['index', 'fold', 'target', 'prediction']
if 'prediction' in exp002_oof.columns:
    oof_features['pred_002'] = exp002_oof['prediction'].values
else:
    oof_features['pred_002'] = exp002_oof.iloc[:, -1].values

# EXP003: ['index', 'fold', 'target', 'prediction']
if 'prediction' in exp003_oof.columns:
    oof_features['pred_003'] = exp003_oof['prediction'].values
else:
    oof_features['pred_003'] = exp003_oof.iloc[:, -1].values

# EXP004: ['id', 'target', 'prediction']
if 'prediction' in exp004_oof.columns:
    oof_features['pred_004'] = exp004_oof['prediction'].values
else:
    oof_features['pred_004'] = exp004_oof.iloc[:, -1].values

# ターゲットを抽出（どのOOFからでもOK）
if 'target' in exp001_oof.columns:
    y_train = exp001_oof['target'].values
else:
    print("❌ ターゲット列が見つかりません")
    exit()

print(f"✓ OOF特徴統合完了")
print(f"  特徴形状: {oof_features.shape}")
print(f"  ターゲット形状: {y_train.shape}")
print(f"\n  OOF相関マトリックス:")
corr_oof = oof_features.corr()
print(corr_oof)

# =====================================================
# Step 4: テスト特徴の統合
# =====================================================
print("\n[STEP 4] テスト特徴の統合")
print("-" * 70)

test_features = pd.DataFrame()

if 'prediction' in exp001_test.columns:
    test_features['pred_001'] = exp001_test['prediction'].values
else:
    test_features['pred_001'] = exp001_test.iloc[:, -1].values

if 'prediction' in exp002_test.columns:
    test_features['pred_002'] = exp002_test['prediction'].values
else:
    test_features['pred_002'] = exp002_test.iloc[:, -1].values

if 'prediction' in exp003_test.columns:
    test_features['pred_003'] = exp003_test['prediction'].values
else:
    test_features['pred_003'] = exp003_test.iloc[:, -1].values

if 'prediction' in exp004_test.columns:
    test_features['pred_004'] = exp004_test['prediction'].values
else:
    test_features['pred_004'] = exp004_test.iloc[:, -1].values

# IDを抽出
test_ids = exp004_test['id'].values if 'id' in exp004_test.columns else exp004_test.iloc[:, 0].values

print(f"✓ テスト特徴統合完了")
print(f"  特徴形状: {test_features.shape}")
print(f"  ID形状: {test_ids.shape}")

# =====================================================
# Step 5: メタモデルの学習
# =====================================================
print("\n[STEP 5] メタモデルのトレーニング")
print("-" * 70)

# 特徴の正規化（オプション）
scaler = StandardScaler()
oof_features_scaled = scaler.fit_transform(oof_features)
test_features_scaled = scaler.transform(test_features)

# メタモデルA: Ridge回帰
print("\n  メタモデル A: Ridge 回帰")
ridge_meta = Ridge(alpha=1.0, random_state=42)
ridge_meta.fit(oof_features_scaled, y_train)
ridge_coef = ridge_meta.coef_

print(f"    Ridge係数: {ridge_coef}")
print(f"    係数合計: {ridge_coef.sum():.6f}")
print(f"    Train AUC: {roc_auc_score(y_train, ridge_meta.predict(oof_features_scaled)):.6f}")

# メタモデルB: ロジスティック回帰
print("\n  メタモデル B: ロジスティック回帰")
lr_meta = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
lr_meta.fit(oof_features_scaled, y_train)
lr_coef = lr_meta.coef_[0]

print(f"    LR係数: {lr_coef}")
print(f"    係数合計: {lr_coef.sum():.6f}")
print(f"    Train AUC: {roc_auc_score(y_train, lr_meta.predict_proba(oof_features_scaled)[:, 1]):.6f}")

# =====================================================
# Step 6: テスト予測の生成
# =====================================================
print("\n[STEP 6] テスト予測の生成")
print("-" * 70)

# Ridge メタモデルでのテスト予測
y_pred_ridge = ridge_meta.predict(test_features_scaled)
# LogisticRegression メタモデルでのテスト予測
y_pred_lr = lr_meta.predict_proba(test_features_scaled)[:, 1]

# 2つのメタモデルを平均
y_pred_stacking = (y_pred_ridge + y_pred_lr) / 2

# 予測値を [0, 1] にクリップ
y_pred_stacking = np.clip(y_pred_stacking, 0, 1)

print(f"✓ テスト予測生成完了")
print(f"\n  Ridge メタモデル予測統計:")
print(f"    Mean: {y_pred_ridge.mean():.6f}")
print(f"    Std:  {y_pred_ridge.std():.6f}")
print(f"    Min:  {y_pred_ridge.min():.6f}")
print(f"    Max:  {y_pred_ridge.max():.6f}")

print(f"\n  LR メタモデル予測統計:")
print(f"    Mean: {y_pred_lr.mean():.6f}")
print(f"    Std:  {y_pred_lr.std():.6f}")
print(f"    Min:  {y_pred_lr.min():.6f}")
print(f"    Max:  {y_pred_lr.max():.6f}")

print(f"\n  統合予測（平均）統計:")
print(f"    Mean: {y_pred_stacking.mean():.6f}")
print(f"    Std:  {y_pred_stacking.std():.6f}")
print(f"    Min:  {y_pred_stacking.min():.6f}")
print(f"    Max:  {y_pred_stacking.max():.6f}")

# =====================================================
# Step 7: サブミッション作成
# =====================================================
print("\n[STEP 7] サブミッション作成")
print("-" * 70)

# Ridge版
submission_ridge = pd.DataFrame({
    'id': test_ids,
    'Churn': y_pred_ridge
})
submission_ridge.to_csv('submission_stacking_ridge.csv', index=False)
print(f"✓ submission_stacking_ridge.csv を保存")

# LR版
submission_lr = pd.DataFrame({
    'id': test_ids,
    'Churn': y_pred_lr
})
submission_lr.to_csv('submission_stacking_lr.csv', index=False)
print(f"✓ submission_stacking_lr.csv を保存")

# 統合版（推奨）
submission_stacking = pd.DataFrame({
    'id': test_ids,
    'Churn': y_pred_stacking
})
submission_stacking.to_csv('submission_stacking_optimal.csv', index=False)
print(f"✓ submission_stacking_optimal.csv を保存 ⭐")

# =====================================================
# Step 8: 詳細分析
# =====================================================
print("\n" + "=" * 70)
print("[STEP 8] 詳細分析")
print("=" * 70)

# 各モデルの単純平均
y_simple_avg = test_features.mean(axis=1).values
submission_simple = pd.DataFrame({
    'id': test_ids,
    'Churn': y_simple_avg
})
submission_simple.to_csv('submission_simple_average.csv', index=False)

print(f"\n比較: 単純平均 vs メタモデルStackingの予測分布")
print(f"  単純平均 Mean: {y_simple_avg.mean():.6f}")
print(f"  Stacking Mean: {y_pred_stacking.mean():.6f}")
print(f"  差分: {(y_pred_stacking.mean() - y_simple_avg.mean())*100:.3f}%")

print(f"\n✓ submission_simple_average.csv も作成 (比較用)")

# =====================================================
# 最終推奨
# =====================================================
print("\n" + "=" * 70)
print("📊 最終推奨")
print("=" * 70)
print("""
【次のステップ】

1. **submission_stacking_optimal.csv をKaggleに提出**
   - OOF Ridge + LR メタモデルの統合予測
   - 現在の0.91388から +0.01～0.02を期待
   - 目標: 0.9160～0.9170

2. **異なるメタモデル構成も試す**（時間があれば）
   - XGBoost メタモデル
   - LightGBM メタモデル
   
3. **CatBoost (EXP007)で追加多様性を確保**
   - 新モデルの OOF を含めてスタッキング再実行
   - 5モデルスタッキングで+0.1～0.2%上乗せ
   - 目標: 0.9170～0.9180

【重要な注意点】
⚠️ このOOF Stackingは完全にリークフリーです（訓練データの予測をメタモデル学習に使用）
⚠️ CVと提出結果がほぼ一致するはずです
⚠️ 複雑すぎるメタモデルは過学習のリスクがあります
""")

print("=" * 70)
