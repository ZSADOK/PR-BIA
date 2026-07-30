"""
Pipeline d'Entraînement Multi-Epochs pour TabFM Deep Tabular Neural Network
Entraîne le modèle sur 250 Epochs avec Rétropropagation des Gradients, Auto-Correction (AdamW + Cosine Scheduler),
Évaluation à chaque Epoch sur la métrique Custom CAUM et Early Stopping.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.crypto_utility_metric import CryptoCustomUtilityMetric

print("=" * 80)
print(" 🚀 PIPELINE MULTI-EPOCHS TABFM DEEP TABULAR : ENTRAÎNEMENT & RÉTROPROPAGATION")
print(" Auto-Correction par Gradients, Early Stopping & Métrique Custom CAUM")
print("=" * 80)

# 1. Dataset Crypto
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
print(f"\n[1/5] 📈 Téléchargement du Dataset Crypto (2020 - 2026)...")
data = yf.download(CRYPTO_TICKERS, start="2020-01-01", interval="1d", group_by="ticker", progress=False)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    log_ret = np.log(close / close.shift(1))
    
    feat = pd.DataFrame(index=close.index)
    feat["trend_dev_7d"] = (close - close.rolling(7).mean()) / close.rolling(7).mean()
    feat["trend_dev_30d"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    
    vol_5d = log_ret.rolling(5).std()
    vol_30d = log_ret.rolling(30).std()
    feat["noise_ratio"] = vol_5d / (vol_30d + 1e-8)
    feat["vol_zscore"] = (vol_30d - vol_30d.rolling(60).mean()) / (vol_30d.rolling(60).std() + 1e-8)
    
    feat["volume_zscore"] = (volume - volume.rolling(14).mean()) / (volume.rolling(14).std() + 1e-8)
    feat["momentum_7d"] = log_ret.rolling(7).sum()
    
    future_ret_5d = log_ret.shift(-5).rolling(5).sum()
    feat["target"] = (future_ret_5d > 0.03).astype(int)
    feat["future_return_5d"] = future_ret_5d
    
    return feat.dropna()

all_dfs = []
for t in CRYPTO_TICKERS:
    try:
        all_dfs.append(extract_features(data[t]))
    except Exception:
        pass

full_df = pd.concat(all_dfs).sort_index()

# 2. Découpage 3-Split (Train 60% / Val 20% / Test Holdout 20%)
n_total = len(full_df)
n_train = int(n_total * 0.60)
n_val = int(n_total * 0.80)

train_df = full_df.iloc[:n_train]
val_df = full_df.iloc[n_train:n_val]
test_df = full_df.iloc[n_val:]

feature_cols = [c for c in train_df.columns if c not in ["target", "future_return_5d"]]

# Normalisation Standard Scaler
mean = train_df[feature_cols].mean()
std = train_df[feature_cols].std() + 1e-8

X_train_norm = (train_df[feature_cols] - mean) / std
X_val_norm = (val_df[feature_cols] - mean) / std
X_test_norm = (test_df[feature_cols] - mean) / std

y_train = train_df["target"].values
y_val = val_df["target"].values
ret_val = val_df["future_return_5d"].values
y_test = test_df["target"].values
ret_test = test_df["future_return_5d"].values

# Conversion Tensors PyTorch
train_dataset = TensorDataset(torch.tensor(X_train_norm.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

X_val_tensor = torch.tensor(X_val_norm.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.float32)

# 3. Architecture Deep Tabular Transformer / Residual Gated Net
class TabFMDeepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = TabFMDeepNet(len(feature_cols))
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

EPOCHS = 900
best_caum_score = -999.0
patience_counter = 0
PATIENCE_LIMIT = 150

os.makedirs("models", exist_ok=True)
checkpoint_path = "models/tabfm_best_epoch.pt"

print(f"\n[3/5] 🏋️ DÉMARRAGE DE L'ENTRAÎNEMENT MULTI-EPOCHS ({EPOCHS} EPOCHS MAX)...")
print("-" * 85)
print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Acc (%)':<12} | {'Val WinRate':<12} | {'Val CAUM Score':<15} | Statut")
print("-" * 85)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)

    train_loss = running_loss / len(train_df)

    # Évaluation à chaque Epoch sur le Validation Set
    model.eval()
    with torch.no_grad():
        val_probs = model(X_val_tensor).numpy()

    val_preds = (val_probs >= 0.54).astype(int)
    val_acc = accuracy_score(y_val, val_preds) * 100.0
    val_metrics = evaluator.compute_asymmetric_utility(val_probs, ret_val)
    val_caum = val_metrics["crypto_utility_score"]
    val_wr = val_metrics["win_rate"]

    scheduler.step(val_caum)

    status = ""
    if val_caum > best_caum_score:
        best_caum_score = val_caum
        torch.save(model.state_dict(), checkpoint_path)
        status = "[CHECKPOINT SAUVEGARDÉ]"
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or status != "":
        print(f"Epoch {epoch:<5}/{EPOCHS} | Loss: {train_loss:.4f}  | Acc: {val_acc:5.1f}%  | WinRate: {val_wr:5.1f}% | CAUM: {val_caum:7.2f}     | {status}")

    if patience_counter >= PATIENCE_LIMIT:
        print(f"\n✋ Early Stopping déclenché à l'Epoch {epoch} (Aucune amélioration depuis {PATIENCE_LIMIT} epochs).")
        break

print("-" * 85)
print(f"  🏆 Meilleurs Poids Sélectionnés à l'Epoch Optimale (Meilleur Score CAUM: {best_caum_score:.2f})")

# 4. Évaluation Finale sur le Jeu de Test Holdout (NON-VU)
print("\n[4/5] 🧪 ÉVALUATION FINALE SUR LE JEU DE TEST HOLDOUT (NON-VU)...")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()

test_preds = (test_probs >= 0.54).astype(int)
test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_metrics = evaluator.compute_asymmetric_utility(test_probs, ret_test)

print("\n" + "=" * 80)
print(" 🏆 RÉSULTATS DU MODÈLE TABFM MULTI-EPOCHS SUR LE JEU DE TEST HOLDOUT")
print("=" * 80)
print(f"  • Précision Globale (Accuracy)   : {test_acc*100:.2f}%")
print(f"  • Précision Signaux ACHAT       : {test_prec*100:.2f}%")
print(f"  • Score ROC-AUC                 : {test_auc:.3f}")
print(f"  • Profit Factor                 : {test_metrics.get('profit_factor', 0.0):.2f}")
print(f"  • Sharpe Ratio Crypto (24/7)    : {test_metrics.get('sharpe_ratio_crypto', 0.0):.2f}")
print(f"  • Win Rate Signaux (%)          : {test_metrics.get('win_rate', 0.0):.1f}%")
print(f"  • Score Utilité CAUM            : {test_metrics.get('crypto_utility_score', 0.0):.2f}")
print("=" * 80 + "\n")
