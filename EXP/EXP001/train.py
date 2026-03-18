#!/usr/bin/env python3
"""
EXP001: LightGBM Baseline Training Script

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


def preprocess(train_df, test_df, config):
    """前処理（簡易版）"""
    # この部分は実データに合わせて カスタマイズ必須

    target_name = config["target"]["name"]

    # ターゲット抽出（Yes/No → 1/0 に変換）
    y_train = (train_df[target_name] == "Yes").astype(int).values

    # 特徴抽出（ターゲット＆IDを除外）
    X_train = train_df.drop(columns=[target_name, "id"], errors="ignore")
    X_test = (
        test_df.drop(columns=["id"], errors="ignore")
        if "id" in test_df.columns
        else test_df.copy()
    )

    # test_dfのIDを保持（推論時に必要）
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )

    # カテゴリカル特徴のラベルエンコーディング
    categorical_cols = X_train.select_dtypes(include="object").columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        le_dict[col] = le

        # Test は Train の encoder を使う
        try:
            X_test[col] = le.transform(X_test[col].astype(str))
        except ValueError:
            # 未知のカテゴリがある場合は-1で埋める
            test_values = X_test[col].astype(str)
            test_encoded = np.where(
                test_values.isin(le.classes_), le.transform(test_values), -1
            )
            X_test[col] = test_encoded

    # 欠損値埋める
    train_mean = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_mean)
    X_test = X_test.fillna(train_mean)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Target distribution:\n{pd.Series(y_train).value_counts(normalize=True)}")

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

    # クラスウェイト設定
    if config["class_weight"]["enabled"]:
        if config["class_weight"]["method"] == "balanced":
            n_class_0 = (y_train == 0).sum()
            n_class_1 = (y_train == 1).sum()
            model_params["scale_pos_weight"] = n_class_0 / n_class_1

    oof_predictions = []
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

        # OOF 予測
        val_pred = model.predict(X_val)
        val_auc = roc_auc_score(y_val, val_pred)
        val_logloss = log_loss(y_val, val_pred)

        print(f"Fold {fold} AUC: {val_auc:.4f}, Logloss: {val_logloss:.4f}")

        for idx, pred in zip(val_idx, val_pred):
            oof_predictions.append(
                {"index": idx, "fold": fold, "target": y_train[idx], "prediction": pred}
            )

        models.append(model)
        feature_importance_list.append(model.feature_importance())

    oof_df = pd.DataFrame(oof_predictions)

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

    return oof_df, models, cv_auc, cv_logloss


def inference(X_train, X_test, models):
    """テセット予測"""
    test_predictions = np.zeros(len(X_test))

    for model in models:
        test_predictions += model.predict(X_test) / len(models)

    return test_predictions


def save_results(
    oof_df, test_predictions, cv_auc, cv_logloss, config, config_path, child_exp_name
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

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 設定ファイルをコピー
    import shutil

    shutil.copy(config_path, output_dir / "config.yaml")

    # テスト予測を一時保存
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

    # 前処理
    X_train, y_train, X_test, test_ids = preprocess(train_df, test_df, config)

    # CV 訓練
    oof_df, models, cv_auc, cv_logloss = train_cv(X_train, y_train, config)

    # テスト予測
    test_predictions = inference(X_train, X_test, models)

    # 結果保存
    save_results(
        oof_df,
        test_predictions,
        cv_auc,
        cv_logloss,
        config,
        args.config,
        child_exp_name,
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
