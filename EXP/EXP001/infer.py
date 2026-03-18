#!/usr/bin/env python3
"""
EXP001: LightGBM Inference Script

テストデータに対して予測を行い、Kaggle 提出形式で出力
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder


def load_config(config_path):
    """設定ファイルを読込"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """データを読込"""
    train_df = pd.read_csv(config["data"]["train_path"])
    test_df = pd.read_csv(config["data"]["test_path"])

    return train_df, test_df


def preprocess_for_inference(train_df, test_df, config):
    """推論用の前処理（訓練時と同じ処理を実行）"""

    target_name = config["target"]["name"]

    # ターゲット抽出
    y_train = train_df[target_name].values

    # 特徴抽出
    X_train = train_df.drop(columns=[target_name])
    X_test = test_df.copy()

    # カテゴリカル特徴のラベルエンコーディング
    # （重要：訓練時のエンコーダを使う必要があるが、ここは簡略化）
    categorical_cols = X_train.select_dtypes(include="object").columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))

        # Test は Train の encoder を使う
        test_values = X_test[col].astype(str)
        test_values = np.where(
            test_values.isin(le.classes_),
            le.transform(test_values),
            -1,  # unknown を-1で埋める
        )
        X_test[col] = test_values

    # 欠損値埋める（訓練データの平均値を使う）
    train_mean = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_mean)
    X_test = X_test.fillna(train_mean)

    return X_train, X_test


def load_models(output_dir, n_folds):
    """訓練済みモデルを読込（実装: 別途modelを保存する場合）"""
    # 注：train.py で model を pkl で保存している場合
    # import pickle
    # models = []
    # for fold in range(n_folds):
    #     with open(output_dir / f'model_fold{fold}.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     models.append(model)
    # return models

    # 簡略版：train.py が保存した test_predictions.csv を使う
    return None


def inference_from_oof(oof_df, test_df, config, output_dir):
    """
    OOF予測から、test_predictions.csv を使用（推奨）

    train.py で生成された test_predictions.csv を読込して使用
    """

    # test_predictions.csv を読込
    test_preds_path = output_dir / "test_predictions.csv"
    if test_preds_path.exists():
        test_pred_df = pd.read_csv(test_preds_path)
        return test_pred_df["prediction"].values
    else:
        print(f"Warning: {test_preds_path} not found")
        return None


def create_submission(test_df, predictions, child_exp_name):
    """Kaggle提出形式で結果を保存"""

    output_path = Path("data") / f"submission_{child_exp_name}.csv"

    submission_df = pd.DataFrame(
        {
            "Id": test_df.index if "Id" not in test_df.columns else test_df["Id"],
            "Churn": predictions,
        }
    )

    # 確率が [0, 1] 範囲内か確認
    assert (
        submission_df["Churn"].min() >= 0 and submission_df["Churn"].max() <= 1
    ), "Predictions not in [0, 1] range"

    submission_df.to_csv(output_path, index=False)

    print(f"Submission saved to {output_path}")
    print(f"Shape: {submission_df.shape}")
    print(f"Sample:\n{submission_df.head()}")

    return submission_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="OOF output directory"
    )
    args = parser.parse_args()

    # 設定読込
    config = load_config(args.config)

    # child-exp 名を抽出
    child_exp_name = Path(args.config).stem

    if args.output_dir is None:
        output_dir = Path(f"outputs/{child_exp_name}")
    else:
        output_dir = Path(args.output_dir)

    print(f"Inference for {child_exp_name}...")
    print(f"Output dir: {output_dir}")

    # データ読込
    train_df, test_df = load_data(config)

    # 前処理
    X_train, X_test = preprocess_for_inference(train_df, test_df, config)

    # OOF から test_predictions を読込
    oof_df = pd.read_csv(output_dir / "oof_predictions.csv")
    test_predictions = inference_from_oof(oof_df, test_df, config, output_dir)

    if test_predictions is None:
        print("ERROR: Could not load test predictions")
        return

    # Kaggle提出形式で保存
    submission_df = create_submission(test_df, test_predictions, child_exp_name)

    print("\nInference completed!")


if __name__ == "__main__":
    main()
