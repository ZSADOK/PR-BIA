"""
Pipeline d'Entraînement Deep Learning : Double Descente SOTA (8.5M Paramètres)
- Multi-Head Self-Attention Transformer Sur-Paramétré (embed_dim=1024, 8 heads)
- Scheduler CosineAnnealingWarmRestarts (SGDR) pour sauter les pièges de mémorisation
- Suivi précis et affichage de l'Epoch optimale retenue.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.crypto_utility_metric import CryptoCustomUtilityMetric

print("=" * 95)
print(" 🧠 DOUBLE DESCENTE SOTA : DEEP TRANSFORMER SUR-PARAMÉTRÉ (8.5M PARAMÈTRES) & SGDR")
print(" Auto-Correction par Gradients, Warm Restarts et Suivi de l'Epoch Optimale")
print("=" * 95)

# 1. Dataset Crypto
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
print(f"\n[1/5] 📈 Chargement du Dataset Crypto (2020 - 2026)...")
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

n_total = len(full_df)
n_train = int(n_total * 0.60)
n_val = int(n_total * 0.80)

train_df = full_df.iloc[:n_train]
val_df = full_df.iloc[n_train:n_val]
test_df = full_df.iloc[n_val:]

feature_cols = [c for c in train_df.columns if c not in ["target", "future_return_5d"]]

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

train_dataset = TensorDataset(torch.tensor(X_train_norm.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

X_val_tensor = torch.tensor(X_val_norm.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.float32)

# 2. Architecture Sur-Paramétrée Élargie (8.5M Paramètres - 8 Heads)
class DoubleDescentLargeTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, n_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.2)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out).squeeze(1)
        return self.head(h).squeeze(-1)

model = DoubleDescentLargeTransformer(len(feature_cols))
num_params = sum(p.numel() for p in model.parameters())
print(f"  🧠 Capacité du Modèle Élargi : {num_params:,} Paramètres")

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
# SGDR: Cosine Annealing avec Relances Périodiques pour sauter hors des minimums locaux
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150, T_mult=2)
evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

EPOCHS = 3000
best_caum_score = -999.0
best_epoch_num = 0
checkpoint_path = "models/double_descent_best.pt"
os.makedirs("models", exist_ok=True)

print(f"\n[3/5] 🚀 ENTRAÎNEMENT DOUBLE DESCENTE AVEC WARM RESTARTS ({EPOCHS} EPOCHS)...")
print("-" * 105)
print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Acc (%)':<12} | {'Val WinRate':<12} | {'Val CAUM Score':<15} | {'LR':<10} | Régime")
print("-" * 105)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        # Bruit de régularisation sur les features pendant l'entraînement
        noise = torch.randn_like(bx) * 0.02
        out = model(bx + noise)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)

    train_loss = running_loss / len(train_df)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    if epoch % 25 == 0 or epoch in [1, 10, 50, 100, 200, 500, 1000, 2000, 3000]:
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_tensor).numpy()

        val_preds = (val_probs >= 0.54).astype(int)
        val_acc = accuracy_score(y_val, val_preds) * 100.0
        val_metrics = evaluator.compute_asymmetric_utility(val_probs, ret_val)
        val_caum = val_metrics["crypto_utility_score"]
        val_wr = val_metrics["win_rate"]

        regime = "1er Régime" if epoch < 100 else ("Interpolation" if epoch < 400 else "🔥 2nd Descente")

        status = ""
        if val_caum > best_caum_score:
            best_caum_score = val_caum
            best_epoch_num = epoch
            torch.save(model.state_dict(), checkpoint_path)
            status = f"[RECORD CHECKPOINT @ Epoch #{epoch}]"

        print(f"Epoch {epoch:<5}/{EPOCHS} | Loss: {train_loss:.4f}  | Acc: {val_acc:5.1f}%  | WinRate: {val_wr:5.1f}% | CAUM: {val_caum:7.2f}     | LR: {current_lr:.1e} | {regime} {status}")

print("-" * 105)
print(f"  🏆 LE MEILLEUR MODÈLE A ÉTÉ OBTENU À L'EPOCH #{best_epoch_num} (Score CAUM Record: {best_caum_score:.2f})")

# 4. Évaluation Finale sur Test Holdout NON-VU avec affichage clair de l'Epoch retenue
print("\n[4/5] 🧪 ÉVALUATION DU MEILLEUR MODÈLE SUR LE JEU DE TEST HOLDOUT...")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()

test_preds = (test_probs >= 0.54).astype(int)
test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_metrics = evaluator.compute_asymmetric_utility(test_probs, ret_test)

print("\n" + "=" * 95)
print(f" 🏆 RÉSULTATS DU MEILLEUR MODÈLE DEEP TRANSFORMER (ISSU DE L'EPOCH #{best_epoch_num} SUR {EPOCHS})")
print("=" * 95)
print(f"  • Epoch Optimale Retenue       : Epoch #{best_epoch_num} / {EPOCHS}")
print(f"  • Score CAUM de Validation      : {best_caum_score:.2f}")
print(f"  • Précision Globale (Accuracy)   : {test_acc*100:.2f}%")
print(f"  • Précision Signaux ACHAT       : {test_prec*100:.2f}%")
print(f"  • Score ROC-AUC                 : {test_auc:.3f}")
print(f"  • Profit Factor                 : {test_metrics.get('profit_factor', 0.0):.2f}")
print(f"  • Sharpe Ratio Crypto (24/7)    : {test_metrics.get('sharpe_ratio_crypto', 0.0):.2f}")
print(f"  • Win Rate Signaux (%)          : {test_metrics.get('win_rate', 0.0):.1f}%")
print(f"  • Score Utilité CAUM            : {test_metrics.get('crypto_utility_score', 0.0):.2f}")
print("=" * 95 + "\n")
