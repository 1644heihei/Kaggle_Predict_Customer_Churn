#!/usr/bin/env python3
"""
EXP009: PyTorch + ResNet型ニューラルネット（タビュラーデータ用）

EXP004の特徴量エンジニアリングを使いながら、
PyTorchで残差接続を持つニューラルネットワークを構築・訓練。

Usage:
    python train.py --config config/child-exp000.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import warnings

warnings.filterwarnings("ignore")


# ===================== CONFIG & DATA =====================


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


# ===================== FEATURE ENGINEERING =====================


def create_interaction_features(df, interactions):
    """相互作用項を生成"""
    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            if pd.api.types.is_numeric_dtype(
                df[col1]
            ) and pd.api.types.is_numeric_dtype(df[col2]):
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
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
    return df


def create_group_features(df):
    """カテゴリ別グループ化特徴"""
    if "Contract" in df.columns and "MonthlyCharges" in df.columns:
        contract_avg = df.groupby("Contract")["MonthlyCharges"].transform("mean")
        df["Contract_AvgCharge"] = contract_avg

    if "InternetService" in df.columns and "tenure" in df.columns:
        internet_avg = df.groupby("InternetService")["tenure"].transform("mean")
        df["InternetService_AvgTenure"] = internet_avg

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
    print("\n[Feature Engineering:]")

        interactions = config["feature_engineering"].get("interactions", [])
        if interactions:
            X_train = create_interaction_features(X_train, interactions)
            X_test = create_interaction_features(X_test, interactions)
            print(f"  ✓ Interaction features created")

        if config["feature_engineering"].get("count_services", True):
            X_train = count_services(X_train)
            X_test = count_services(X_test)
            print(f"  ✓ Service Count created")

        if config["feature_engineering"].get("group_features", True):
            X_train = create_group_features(X_train)
            X_test = create_group_features(X_test)
            print(f"  ✓ Group features created")

    # ============= ENCODING ================
    categorical_cols = X_train.select_dtypes(include="object").columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        try:
            X_test[col] = le.transform(X_test[col].astype(str))
        except ValueError:
            X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

    # NaN埋め
    X_train = X_train.fillna(X_train.mean(numeric_only=True))
    X_test = X_test.fillna(X_train.mean(numeric_only=True))

    print(f"\n✅ Final X_train shape: {X_train.shape}")
    print(f"✅ Final X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, test_ids


# ===================== PYTORCH MODEL =====================


class ResidualBlock(nn.Module):
    """残差ブロック（Linear版）"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.3, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(output_dim)
            self.bn2 = nn.BatchNorm1d(output_dim)

        # Skip connection の次元調整
        self.skip = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.skip(x)

        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        out = self.dropout(out)

        out = out + identity  # Skip connection
        out = self.relu(out)

        return out


class TabularResNet(nn.Module):
    """タビュラーデータ向けResNet"""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout_rate=0.3,
        use_batch_norm=True,
        use_residual=True,
    ):
        super(TabularResNet, self).__init__()
        self.use_residual = use_residual

        if use_residual:
            # 残差ブロック版
            self.input_bn = nn.BatchNorm1d(input_dim)
            self.initial = nn.Linear(input_dim, hidden_dims[0])

            self.residual_blocks = nn.ModuleList()
            dims = [hidden_dims[0]] + hidden_dims[1:] + [hidden_dims[-1]]
            for i in range(len(hidden_dims)):
                self.residual_blocks.append(
                    ResidualBlock(dims[i], dims[i + 1], dropout_rate, use_batch_norm)
                )

            self.output = nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:
            # シンプルなMLP版
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
                )
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            layers.extend(
                [
                    nn.Linear(prev_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                ]
            )
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            x = self.input_bn(x)
            x = self.initial(x)
            for block in self.residual_blocks:
                x = block(x)
            x = self.output(x)
        else:
            x = self.model(x)
        return x.squeeze(-1)


# ===================== TRAINING =====================


def train_epoch(model, train_loader, criterion, optimizer, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(train_loader.dataset)


def evaluate(model, val_loader, criterion, device):
    """検証"""
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(logits.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    avg_loss = total_loss / len(val_loader.dataset)
    auc = roc_auc_score(y_true, y_pred)

    return avg_loss, auc, y_pred


def predict_fold(model, data_loader, device):
    """fold内の予測"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            predictions.extend(logits.cpu().numpy())

    return np.array(predictions)


# ===================== MAIN =====================


def main():
    parser = argparse.ArgumentParser(description="EXP009: PyTorch ResNet")
    parser.add_argument(
        "--config", type=str, default="config/child-exp000.yaml", help="Config path"
    )
    args = parser.parse_args()

    # Config読込
    config = load_config(args.config)
    print(f"[*] Config loaded from {args.config}")

    # Output ディレクトリのセットアップ
    script_dir = Path(__file__).parent
    child_exp_name = Path(args.config).stem
    output_dir = script_dir / "outputs" / child_exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device設定
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"[Device] {device}")

    # データ読込
    print("\n[Loading data...]")
    train_df, test_df = load_data(config)

    # 前処理
    print("\n[Preprocessing...]")
    X_train, X_test, y_train, test_ids = preprocess(train_df, test_df, config)

    # スケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    print(f"📊 Input dimension: {input_dim}")

    # ============= CV TRAINING =============
    print("\n🎯 Starting 5-Fold Cross-Validation...")

    skf = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        random_state=config["cv"]["random_state"],
        shuffle=config["cv"]["shuffle"],
    )

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))
    cv_scores_auc = []
    cv_scores_logloss = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\n📍 Fold {fold_idx + 1}/{config['cv']['n_splits']}")

        # データ分割
        X_tr = torch.FloatTensor(X_train_scaled[train_idx])
        y_tr = torch.FloatTensor(y_train[train_idx])
        X_val = torch.FloatTensor(X_train_scaled[val_idx])
        y_val = torch.FloatTensor(y_train[val_idx])
        X_test_t = torch.FloatTensor(X_test_scaled)

        # DataLoader
        train_dataset = TensorDataset(X_tr, y_tr)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test_t, torch.zeros(len(X_test_t)))

        train_loader = DataLoader(
            train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
        )

        # モデル初期化
        model = TabularResNet(
            input_dim=input_dim,
            hidden_dims=config["model"]["hidden_dims"],
            dropout_rate=config["model"]["dropout_rate"],
            use_batch_norm=config["model"]["batch_norm"],
            use_residual=config["model"]["use_residual"],
        ).to(device)

        # 最適化設定
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Train loop
        best_auc = 0.0
        patience_counter = 0
        best_model_state = None

        for epoch in range(config["training"]["num_epochs"]):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_auc, _ = evaluate(model, val_loader, criterion, device)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.6f}"
                )

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Best model をロード
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # OOF & Test 予測
        oof_predictions[val_idx] = predict_fold(model, val_loader, device)

        test_pred_fold = predict_fold(model, test_loader, device)
        test_predictions += test_pred_fold / config["cv"]["n_splits"]

        # Evaluation
        auc = roc_auc_score(y_train[val_idx], oof_predictions[val_idx])
        logloss = log_loss(y_train[val_idx], oof_predictions[val_idx])
        cv_scores_auc.append(auc)
        cv_scores_logloss.append(logloss)

        print(f"  ✅ Fold {fold_idx + 1}: AUC={auc:.6f}, LogLoss={logloss:.6f}")

    # ============= RESULTS =============
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(
        f"Mean CV AUC:     {np.mean(cv_scores_auc):.6f} (+/- {np.std(cv_scores_auc):.6f})"
    )
    print(
        f"Mean CV LogLoss: {np.mean(cv_scores_logloss):.6f} (+/- {np.std(cv_scores_logloss):.6f})"
    )
    print("=" * 60)

    # 結果保存
    results = {
        "model": "pytorch_resnet",
        "cv_auc": {
            "mean": float(np.mean(cv_scores_auc)),
            "std": float(np.std(cv_scores_auc)),
            "folds": [float(x) for x in cv_scores_auc],
        },
        "cv_logloss": {
            "mean": float(np.mean(cv_scores_logloss)),
            "std": float(np.std(cv_scores_logloss)),
            "folds": [float(x) for x in cv_scores_logloss],
        },
        "input_dim": int(input_dim),
        "hidden_dims": config["model"]["hidden_dims"],
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Test Predictions
    test_pred_df = pd.DataFrame(
        {"id": np.arange(len(test_predictions)), "prediction": test_predictions}
    )
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    # OOF Predictions (5列: id, fold1~fold5のpred値 or シンプルに id + pred)
    oof_pred_df = pd.DataFrame(
        {
            "id": np.arange(len(oof_predictions)),
            "prediction": oof_predictions,
            "target": y_train,
        }
    )
    oof_pred_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    print(f"\n✅ Results saved to {output_dir}")
    print(f"   - results.json")
    print(f"   - test_predictions.csv ({len(test_pred_df)} rows)")
    print(f"   - oof_predictions.csv ({len(oof_pred_df)} rows)")


if __name__ == "__main__":
    main()
