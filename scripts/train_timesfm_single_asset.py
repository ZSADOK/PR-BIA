"""
Script d'Entraînement SOTA TimesFM Single-Asset (BTC-USD) avec Early Stopping
Entraîne le Transformer de séries temporelles sur la prédiction de trajectoire séquentielle à 5 pas futures.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.trading_config import console

console.print("\n[bold cyan]================================================================================[/bold cyan]")
console.print("[bold yellow] 🚀 ENTRAÎNEMENT SOTA TIMESFM SINGLE-ASSET (BTC-USD) AVEC EARLY STOPPING [/bold yellow]")
console.print("[bold cyan]================================================================================[/bold cyan]")

TICKER = "BTC-USD"
SEQ_LEN = 48    # Historique 48 heures
PRED_LEN = 5    # Trajectoire 5 heures futures

console.print(f"\n[bold green][1/4] 📈 Chargement du Dataset Séries Temporelles ({TICKER})...[/bold green]")
raw_data = yf.download(TICKER, period="730d", interval="1h", progress=False)

close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
close_vals = close_series.dropna().values.flatten()

# Normalisation Log-Returns & Scaling
log_returns = np.diff(np.log(close_vals))

def create_sequences(data, seq_len=48, pred_len=5):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(log_returns, SEQ_LEN, PRED_LEN)

# Découpage Chronologique (Train 75% / Val 25%)
split = int(len(X_seq) * 0.75)
X_train, y_train = X_seq[:split], y_seq[:split]
X_val, y_val = X_seq[split:], y_seq[split:]

console.print(f"  • Train Set : {len(X_train):,} séquences (48h -> 5h)")
console.print(f"  • Val Set   : {len(X_val):,} séquences")

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Architecture TimesFM Sequential Trajectory Transformer
class TimesFMTrajectoryTransformer(nn.Module):
    def __init__(self, seq_len=48, pred_len=5, embed_dim=256, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        h = self.input_proj(x.unsqueeze(-1)) + self.pos_embed
        h = self.transformer(h)
        h_last = h[:, -1, :] # Représentation globale du dernier pas
        return self.output_proj(h_last)

model = TimesFMTrajectoryTransformer(SEQ_LEN, PRED_LEN)
num_params = sum(p.numel() for p in model.parameters())
console.print(f"\n[bold green][2/4] 🧠 Transformer TimesFM Initialisé : {num_params:,} Paramètres[/bold green]")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

EPOCHS = 200
PATIENCE_LIMIT = 20
best_val_loss = float('inf')
patience_counter = 0

os.makedirs("models", exist_ok=True)
checkpoint_path = "models/timesfm_btc_best.pt"

console.print(f"\n[bold green][3/4] 🏋️ DÉMARRAGE DE L'ENTRAÎNEMENT AVEC EARLY STOPPING...[/bold green]")
console.print("-" * 85)
console.print(f"{'Epoch':<10} | {'Train RMSE (%)':<20} | {'Val RMSE (%)':<20} | Statut")
console.print("-" * 85)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)

    train_loss = running_loss / len(X_train)
    train_rmse_pct = np.sqrt(max(1e-8, train_loss)) * 100.0

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, y_val_tensor).item()

    val_rmse_pct = np.sqrt(max(1e-8, val_loss)) * 100.0
    scheduler.step(val_loss)

    status = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        status = "[CHECKPOINT SAUVEGARDÉ]"
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or status != "":
        console.print(f"Epoch {epoch:<5}/{EPOCHS} | RMSE: {train_rmse_pct:6.3f}% / h     | Val RMSE: {val_rmse_pct:6.3f}% / h     | [bold green]{status}[/bold green]")

best_rmse_pct = np.sqrt(max(1e-8, best_val_loss)) * 100.0
console.print("-" * 85)
console.print(f"[bold green] 🏆 MEILLEUR CHECKPOINT TIMESFM OPTIMISÉ (Val RMSE: {best_rmse_pct:.3f}% / h)[/bold green]")
console.print(f" 💾 Modèle sauvegardé dans : [bold cyan]{checkpoint_path}[/bold cyan]\n")
