#!/usr/bin/env python3
"""
EXP002: LightGBM + Feature Engineering

Feature Engineering Implementation:
- 相互作用項 (Interaction terms)
- Service Count (複数サービス加入数)
- カテゴリ別グループ化特徴
- Polynomial features (オプション)

Usage:
    python train.py --config config/child-exp000.yaml
"""

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder


def load_config(config_path):
    """設定ファイルを読込"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """データを読込"""
    train_df = pd.read_csv(config["data"]["train_path"], encoding="utf-8")
    test_df = pd.read_csv(config["data"]["test_path"], encoding="utf-8")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df


def create_interaction_features(df, interactions):
    """相互作用項を生成"""
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            # 数値型のみ対象
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

    # Yes/No をカウント
    service_count = 0
    for col in service_cols:
        if col in df.columns:
            service_count += (df[col] == "Yes").astype(int)

    df["ServiceCount"] = service_count
    print(f"  ✓ Service Count created")
    return df


def create_group_features(df):
    """カテゴリ別グループ化特徴"""

    # Contract別の平均課金を計算
    if "Contract" in df.columns and "MonthlyCharges" in df.columns:
        contract_avg = df.groupby("Contract")["MonthlyCharges"].transform("mean")
        df["Contract_AvgCharge"] = contract_avg
        print(f"  ✓ Contract_AvgCharge created")

    # InternetService別の平均テニュアを計算
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

        # 相互作用項
        interactions = config["feature_engineering"].get("interactions", [])
        if interactions:
            X_train = create_interaction_features(X_train, interactions)
            X_test = create_interaction_features(X_test, interactions)

        # Service Count
        if config["feature_engineering"].get("count_services", True):
            X_train = count_services(X_train)
            X_test = count_services(X_test)

        # グループ化特徴
        if config["feature_engineering"].get("group_features", True):
            X_train = create_group_features(X_train)
            X_test = create_group_features(X_test)

    # ============= ENCODING & PREPROCESSING =============
    # カテゴリカル特徴のラベルエンコーディング
    categorical_cols = X_train.select_dtypes(include="object").columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        try:
            X_test[col] = le.transform(X_test[col].astype(str))
        except ValueError:
            test_values = X_test[col].astype(str)
            test_encoded = np.where(
                test_values.isin(le.classes_), le.transform(test_values), -1
            )
            X_test[col] = test_encoded

    # 欠損値埋める
    train_mean = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_mean)
    X_test = X_test.fillna(train_mean)

    print(f"\n✓ Data processed:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Features added: {X_train.shape[1] - len(train_df.columns) + 2}")
    print(f"  Target distribution: {np.bincount(y_train).tolist()}")

    return X_train, y_train, X_test, test_ids


def train_cv(X_train, y_train, config):
    """K-Fold Cross-Validation"""

    cv = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=config["cv"]["random_state"],
    )

    model_params = config["model"]["params"].copy()
    training_params = config["training"].copy()

    # 二値分類の設定
    model_params["objective"] = "binary"
    if "metric" not in model_params:
        model_params["metric"] = "binary_logloss"

    # クラスウェイト設定
    if config["class_weight"]["enabled"]:
        if config["class_weight"]["method"] == "balanced":
            n_class_0 = (y_train == 0).sum()
            n_class_1 = (y_train == 1).sum()
            model_params["scale_pos_weight"] = n_class_0 / n_class_1
            print(f"Class weights: 0={1.0:.2f}, 1={n_class_0/n_class_1:.2f}")

    oof_indices = np.zeros(len(X_train), dtype=int)
    oof_folds = np.zeros(len(X_train), dtype=int)
    oof_targets = np.zeros(len(X_train), dtype=int)
    oof_preds = np.zeros(len(X_train), dtype=float)

    models = []
    feature_importance_list = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"\n--- Fold {fold} ---")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params=model_params,
            train_set=train_data,
            valid_sets=[valid_data],
            num_boost_round=training_params["num_iterations"],
            callbacks=[
                lgb.early_stopping(training_params["early_stopping_rounds"]),
                lgb.log_evaluation(period=training_params["verbose_eval"]),
            ],
        )

        # OOF 予測（raw score）→ sigmoid処理
        val_pred_raw = model.predict(X_val)
        val_pred = expit(val_pred_raw)

        val_auc = roc_auc_score(y_val, val_pred)
        val_logloss = log_loss(y_val, val_pred)

        print(f"Fold {fold} AUC: {val_auc:.4f}, Logloss: {val_logloss:.4f}")

        oof_indices[val_idx] = val_idx
        oof_folds[val_idx] = fold
        oof_targets[val_idx] = y_val
        oof_preds[val_idx] = val_pred

        models.append(model)
        feature_importance_list.append(model.feature_importance())

    oof_df = pd.DataFrame(
        {
            "index": oof_indices,
            "fold": oof_folds,
            "target": oof_targets,
            "prediction": oof_preds,
        }
    )

    # CV スコア計算
    cv_auc = roc_auc_score(oof_df["target"], oof_df["prediction"])
    cv_logloss = log_loss(oof_df["target"], oof_df["prediction"])

    print(f"\n=== CV Results ===")
    print(f"CV AUC: {cv_auc:.4f}")
    print(f"CV Logloss: {cv_logloss:.4f}")

    # 特徴重要度
    feature_importance_mean = np.mean(feature_importance_list, axis=0)
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance_mean}
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Features:")
    print(importance_df.head(10))

    return oof_df, models, cv_auc, cv_logloss, importance_df


def inference(X_train, X_test, models):
    """テスト予測"""
    test_predictions = np.zeros(len(X_test))

    for model in models:
        test_pred_raw = model.predict(X_test)
        test_pred = expit(test_pred_raw)
        test_predictions += test_pred / len(models)

    return test_predictions


def save_results(
    oof_df,
    test_predictions,
    cv_auc,
    cv_logloss,
    importance_df,
    config,
    config_path,
    child_exp_name,
):
    """結果を保存"""

    output_dir = Path(f"outputs/{child_exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # OOF 予測を保存
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 結果を保存
    results = {
        "cv_auc": float(cv_auc),
        "cv_logloss": float(cv_logloss),
        "n_folds": config["cv"]["n_splits"],
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 特徴重要度を保存
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # 設定ファイルをコピー
    import shutil

    shutil.copy(config_path, output_dir / "config.yaml")

    # テスト予測を保存
    test_pred_df = pd.DataFrame({"prediction": test_predictions})
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    # 設定読込
    config = load_config(args.config)

    # child-exp 名を抽出
    child_exp_name = Path(args.config).stem

    print(f"Training {child_exp_name}...")
    print(f"Config: {args.config}")

    # データ読込
    train_df, test_df = load_data(config)

    # 前処理＋Feature Engineering
    X_train, y_train, X_test, test_ids = preprocess(train_df, test_df, config)

    # CV 訓練
    oof_df, models, cv_auc, cv_logloss, importance_df = train_cv(
        X_train, y_train, config
    )

    # テスト予測
    test_predictions = inference(X_train, X_test, models)

    # 結果保存
    save_results(
        oof_df,
        test_predictions,
        cv_auc,
        cv_logloss,
        importance_df,
        config,
        args.config,
        child_exp_name,
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
