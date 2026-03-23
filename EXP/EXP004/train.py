#!/usr/bin/env python3
"""
EXP004: XGBoost + Optuna ハイパーパラメータ自動最適化

EXP003の特徴量エンジニアリングを使いながら、
Optuna で最適なハイパーパラメータを探索。

Usage:
    python train.py --config config/child-exp000.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Optuna関連のインポート
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def load_config(config_path):
    """設定ファイルを読込"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """データを読込（プロジェクトルートから相対パス）"""
    # プロジェクトルートを基準にパスを解決
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

    # ============= ENCODING & PREPROCESSING =============
    categorical_cols = X_train.select_dtypes(include="object").columns
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

    return X_train, X_test, y_train, test_ids


def evaluate_cv(X_train, y_train, params, config):
    """
    k-fold CV で平均 AUC を計算
    Optuna の目的関数として使用
    """
    skf = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        shuffle=config["cv"]["shuffle"],
    )

    auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # XGBoost DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Training
        evals = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=config["training"]["num_iterations"],
            evals=evals,
            early_stopping_rounds=config["training"]["early_stopping_rounds"],
            verbose_eval=False,
            evals_result={},
        )

        # Prediction
        val_pred_raw = model.predict(dval)
        val_pred = expit(val_pred_raw)

        # AUC計算
        auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(auc)

        print(f"    Fold {fold_idx + 1}/5: AUC = {auc:.6f}")

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    print(f"  → Mean AUC: {mean_auc:.6f} (std: {std_auc:.6f})")

    return mean_auc


def create_objective(X_train, y_train, config, base_params):
    """Optuna 用の目的関数を生成"""

    def objective(trial):
        # ハイパーパラメータのサジェスト
        params = base_params.copy()
        optuna_config = config["optuna"]["hyperparams"]

        for param_name, param_spec in optuna_config.items():
            if param_spec["type"] == "int":
                value = trial.suggest_int(
                    param_name,
                    param_spec["low"],
                    param_spec["high"],
                )
            elif param_spec["type"] == "float":
                log = param_spec.get("log", False)
                value = trial.suggest_float(
                    param_name,
                    param_spec["low"],
                    param_spec["high"],
                    log=log,
                )
            params[param_name] = value

        print(f"\n📊 Trial {trial.number + 1}:")
        print(f"  Hyperparameters: {params}")

        # CV 評価
        mean_auc = evaluate_cv(X_train, y_train, params, config)

        return mean_auc

    return objective


def save_results(output_dir, model, y_val, val_pred, test_pred, feature_names, results):
    """結果を保存"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature Importance
    importance = model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Results JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Test Predictions
    test_pred_df = pd.DataFrame(
        {"id": np.arange(len(test_pred)), "prediction": test_pred}
    )
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\n✅ Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EXP004: XGBoost + Optuna")
    parser.add_argument(
        "--config", type=str, default="config/child-exp000.yaml", help="Config path"
    )
    args = parser.parse_args()

    # Config読込
    config = load_config(args.config)
    print(f"📋 Config loaded from {args.config}")

    # Output ディレクトリのセットアップ（Path(__file__).parentを使用）
    script_dir = Path(__file__).parent
    child_exp_name = Path(args.config).stem
    output_dir = script_dir / "outputs" / child_exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読込
    print("\n📂 Loading data...")
    train_df, test_df = load_data(config)

    # 前処理
    print("\n⚙️  Preprocessing...")
    X_train, X_test, y_train, test_ids = preprocess(train_df, test_df, config)

    # ============= OPTUNA OPTIMIZATION =============
    print("\n🔍 Starting Optuna Optimization...")

    base_params = config["model"]["base_params"].copy()

    if config["optuna"]["enabled"]:
        # Sampler & Pruner
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        # Study 作成
        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction=config["optuna"]["direction"],
        )

        # Objective 関数
        objective = create_objective(X_train, y_train, config, base_params)

        # 最適化実行
        print(f"\n🚀 Optimizing for {config['optuna']['n_trials']} trials...\n")
        study.optimize(
            objective,
            n_trials=config["optuna"]["n_trials"],
            n_jobs=config["optuna"]["n_jobs"],
            show_progress_bar=True,
        )

        # 最良パラメータ取得
        best_trial = study.best_trial
        best_params = base_params.copy()
        best_params.update(best_trial.params)

        print(f"\n✅ Best Trial #{best_trial.number}:")
        print(f"   Best AUC: {best_trial.value:.6f}")
        print(f"   Best Params: {best_trial.params}")

        # Best Trial の パラメータで最終学習
        print(f"\n🎯 Training final model with best hyperparameters...")

    else:
        # Optuna 無効の場合は config の params を使用
        best_params = base_params.copy()
        best_params.update(config["model"]["params"])
        print("  (Optuna disabled, using config parameters)")

    # ============= FINAL TRAINING (全データで学習) =============
    print("\n🏋️  Final training on full data...")

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

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n📍 Fold {fold_idx + 1}/{config['cv']['n_splits']}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # XGBoost DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        # Training
        evals = [(dtrain, "train"), (dval, "val")]
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=config["training"]["num_iterations"],
            evals=evals,
            early_stopping_rounds=config["training"]["early_stopping_rounds"],
            verbose_eval=config["training"]["verbose_eval"],
        )

        # OOF Predictions
        val_pred_raw = model.predict(dval)
        val_pred = expit(val_pred_raw)
        oof_predictions[val_idx] = val_pred

        # Test Predictions
        test_pred_raw = model.predict(dtest)
        test_pred = expit(test_pred_raw)
        test_predictions += test_pred / config["cv"]["n_splits"]

        # Evaluation
        auc = roc_auc_score(y_val, val_pred)
        logloss = log_loss(y_val, val_pred)
        cv_scores_auc.append(auc)
        cv_scores_logloss.append(logloss)

        print(f"  ✅ Fold {fold_idx + 1} - AUC: {auc:.6f}, Logloss: {logloss:.6f}")

    # CV 集計
    mean_auc = np.mean(cv_scores_auc)
    std_auc = np.std(cv_scores_auc)
    mean_logloss = np.mean(cv_scores_logloss)
    std_logloss = np.std(cv_scores_logloss)

    print(f"\n📊 CV Results:")
    print(f"  AUC:     {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"  Logloss: {mean_logloss:.6f} ± {std_logloss:.6f}")

    # OOF Predictions 保存
    oof_df = pd.DataFrame(
        {
            "id": np.arange(len(oof_predictions)),
            "target": y_train,
            "prediction": oof_predictions,
        }
    )
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 結果を辞書化
    results = {
        "experiment": "EXP004",
        "model": "XGBoost + Optuna",
        "cv_auc_mean": float(mean_auc),
        "cv_auc_std": float(std_auc),
        "cv_logloss_mean": float(mean_logloss),
        "cv_logloss_std": float(std_logloss),
        "cv_folds": config["cv"]["n_splits"],
        "best_hyperparameters": best_params,
        "num_features": X_train.shape[1],
        "num_samples": len(X_train),
    }

    if config["optuna"]["enabled"]:
        results["optuna_n_trials"] = config["optuna"]["n_trials"]
        results["optuna_best_trial"] = best_trial.number
        results["optuna_best_value"] = float(best_trial.value)

    # Test 予測を保存
    test_pred_df = pd.DataFrame({"id": test_ids, "prediction": test_predictions})
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # 結果を JSON で保存
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All results saved to {output_dir}")
    print(f"   results.json, oof_predictions.csv, test_predictions.csv")


if __name__ == "__main__":
    main()
