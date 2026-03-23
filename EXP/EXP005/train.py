#!/usr/bin/env python3
"""
EXP005: CatBoost + Feature Engineering

EXP003/004と同じ特徴量を使用し、モデルをCatBoostに変更。
CatBoostはカテゴリ変数を自動処理するため、エンコーディングが簡略。
GPU対応で高速実行。

Usage:
    python train.py --config config/child-exp000.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import catboost as cb
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder


def load_config(config_path):
    """設定ファイルを読込"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """データを読込（プロジェクトルートから相対パス）"""
    project_root = Path(__file__).parent.parent.parent
    train_path = project_root / config["data"]["train_path"]
    test_path = project_root / config["data"]["test_path"]

    train_df = pd.read_csv(train_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def create_interaction_features(df, interactions):
    """相互作用項を生成"""
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            if pd.api.types.is_numeric_dtype(
                df[col1]
            ) and pd.api.types.is_numeric_dtype(df[col2]):
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                print(f"  ✓ Interaction: {col1} × {col2}")
    return df


def count_services(df):
    """複数サービス加入数をカウント"""
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    service_count = 0
    for col in service_cols:
        if col in df.columns:
            service_count += (df[col] == "Yes").astype(int)

    df["ServiceCount"] = service_count
    print(f"  ✓ Service Count created")
    return df


def create_group_features(df):
    """カテゴリ別グループ化特徴"""
    if "Contract" in df.columns and "MonthlyCharges" in df.columns:
        contract_avg = df.groupby("Contract")["MonthlyCharges"].transform("mean")
        df["Contract_AvgCharge"] = contract_avg
        print(f"  ✓ Contract_AvgCharge created")

    if "InternetService" in df.columns and "tenure" in df.columns:
        internet_avg = df.groupby("InternetService")["tenure"].transform("mean")
        df["InternetService_AvgTenure"] = internet_avg
        print(f"  ✓ InternetService_AvgTenure created")

    return df


def preprocess(train_df, test_df, config):
    """前処理＋Feature Engineering"""
    target_name = config["target"]["name"]

    # ターゲット抽出
    y_train = (train_df[target_name] == "Yes").astype(int).values

    # 特徴抽出
    X_train = train_df.drop(columns=[target_name, "id"], errors="ignore")
    X_test = (
        test_df.drop(columns=["id"], errors="ignore")
        if "id" in test_df.columns
        else test_df.copy()
    )

    # test_dfのIDを保持
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )

    # ============= FEATURE ENGINEERING =============
    if config.get("feature_engineering", {}).get("enabled", True):
        print("\n🔧 Applying Feature Engineering:")

        interactions = config["feature_engineering"].get("interactions", [])
        if interactions:
            X_train = create_interaction_features(X_train, interactions)
            X_test = create_interaction_features(X_test, interactions)

        if config["feature_engineering"].get("count_services", True):
            X_train = count_services(X_train)
            X_test = count_services(X_test)

        if config["feature_engineering"].get("group_features", True):
            X_train = create_group_features(X_train)
            X_test = create_group_features(X_test)

    # ============= ENCODING（CatBoostは簡略） =============
    # CatBoostはカテゴリ変数を直接処理可能だが、cross-validation時の安定性のため
    # 事前にエンコードするほうが無難
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    print(f"\n📊 Categorical columns: {categorical_cols}")

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        try:
            X_test[col] = le.transform(X_test[col].astype(str))
        except ValueError:
            print(f"  ⚠️  Warning: {col} has categories in test not in train")
            X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

    # NaN埋め
    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_train.mean(numeric_only=True))

    print(f"\n✅ Final X_train shape: {X_train.shape}")
    print(f"✅ Final X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, test_ids, categorical_cols


def main():
    parser = argparse.ArgumentParser(
        description="EXP005: CatBoost + Feature Engineering"
    )
    parser.add_argument(
        "--config", type=str, default="config/child-exp000.yaml", help="Config path"
    )
    args = parser.parse_args()

    # Config読込
    config = load_config(args.config)
    print(f"📋 Config loaded from {args.config}")

    # Output ディレクトリのセットアップ
    script_dir = Path(__file__).parent
    child_exp_name = Path(args.config).stem
    output_dir = script_dir / "outputs" / child_exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読込
    print("\n📂 Loading data...")
    train_df, test_df = load_data(config)

    # 前処理
    print("\n⚙️  Preprocessing...")
    X_train, X_test, y_train, test_ids, categorical_cols = preprocess(
        train_df, test_df, config
    )

    # ============= TRAINING =============
    print("\n🏋️  Training CatBoost models (5-fold CV)...")

    skf = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        shuffle=config["cv"]["shuffle"],
    )

    # OOF 予測用
    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    cv_scores_auc = []
    cv_scores_logloss = []
    feature_importance_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n📍 Fold {fold_idx + 1}/{config['cv']['n_splits']}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # CatBoost Pool（カテゴリ列を指定）
        train_pool = cb.Pool(
            X_tr,
            label=y_tr,
            cat_features=categorical_cols if categorical_cols else None,
        )
        val_pool = cb.Pool(
            X_val,
            label=y_val,
            cat_features=categorical_cols if categorical_cols else None,
        )
        test_pool = cb.Pool(
            X_test, cat_features=categorical_cols if categorical_cols else None
        )

        # CatBoost モデル
        model = cb.CatBoostClassifier(
            **config["model"]["params"],
            early_stopping_rounds=config["training"]["early_stopping_rounds"],
        )

        # Training
        model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=config["training"]["verbose_eval"],
        )

        # OOF Predictions
        val_pred = model.predict_proba(val_pool)[:, 1]
        oof_predictions[val_idx] = val_pred

        # Test Predictions
        test_pred = model.predict_proba(test_pool)[:, 1]
        test_predictions += test_pred / config["cv"]["n_splits"]

        # Evaluation
        auc = roc_auc_score(y_val, val_pred)
        logloss = log_loss(y_val, val_pred)
        cv_scores_auc.append(auc)
        cv_scores_logloss.append(logloss)

        print(f"  ✅ Fold {fold_idx + 1} - AUC: {auc:.6f}, Logloss: {logloss:.6f}")

        # Feature importance
        importance = model.get_feature_importance(train_pool)
        feature_importance_list.append(importance)

    # CV 集計
    mean_auc = np.mean(cv_scores_auc)
    std_auc = np.std(cv_scores_auc)
    mean_logloss = np.mean(cv_scores_logloss)
    std_logloss = np.std(cv_scores_logloss)

    print(f"\n📊 CV Results:")
    print(f"  AUC:     {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"  Logloss: {mean_logloss:.6f} ± {std_logloss:.6f}")

    # Feature Importance（平均）
    avg_importance = np.mean(feature_importance_list, axis=0)
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": avg_importance}
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    print(f"\n✨ Top 10 Features:")
    print(importance_df.head(10).to_string(index=False))

    # OOF Predictions 保存
    oof_df = pd.DataFrame(
        {
            "id": np.arange(len(oof_predictions)),
            "target": y_train,
            "prediction": oof_predictions,
        }
    )
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Test 予測を保存
    test_pred_df = pd.DataFrame({"id": test_ids, "prediction": test_predictions})
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 結果を辞書化
    results = {
        "experiment": "EXP005",
        "model": "CatBoost + Feature Engineering",
        "cv_auc_mean": float(mean_auc),
        "cv_auc_std": float(std_auc),
        "cv_logloss_mean": float(mean_logloss),
        "cv_logloss_std": float(std_logloss),
        "cv_folds": config["cv"]["n_splits"],
        "num_features": X_train.shape[1],
        "num_samples": len(X_train),
        "best_hyperparameters": config["model"]["params"],
    }

    # 結果を JSON で保存
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All results saved to {output_dir}")
    print(f"   results.json, oof_predictions.csv, test_predictions.csv")


if __name__ == "__main__":
    main()
