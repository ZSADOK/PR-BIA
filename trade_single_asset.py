#!/usr/bin/env python3
"""
Bot de Trading Automatisé Single-Asset (BTC-USD)
Intègre le Modèle TimesFM Ré-entraîné, le Risk Manager et le Terminal Live Rich Sans Scrolling.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

import torch
import torch.nn as nn

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.trading_config import console, trading_client, map_symbol_to_alpaca, load_persistent_state, save_persistent_state
from src.execution import check_instant_safety_limits, execute_sell_all, execute_trade_signals
from src.data import FinancialDataFetcher

SINGLE_TICKER = "BTC-USD"
ALPACA_SYMBOL = "BTC/USD"
SEQ_LEN = 48
PRED_LEN = 5
CHECKPOINT_PATH = "models/timesfm_btc_best.pt"

# Architecture Transformer TimesFM Trajectory Predictor
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
        h = self.input_proj(x.unsqueeze(-1)) + self.pos_embed
        h = self.transformer(h)
        h_last = h[:, -1, :]
        return self.output_proj(h_last)

def load_trained_timesfm_model():
    model = TimesFMTrajectoryTransformer(SEQ_LEN, PRED_LEN)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model.eval()
        return model, True
    return model, False

timesfm_model, is_trained = load_trained_timesfm_model()

def get_single_asset_live_panel(remaining_sec: int = 0) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=14),
        Layout(name="footer", size=3)
    )

    # Header
    layout["header"].update(
        Panel(
            Text(f"🏛️ PR-BIA INSTITUTIONAL SINGLE-ASSET TERMINAL | ACTIF MAÎTRE : {SINGLE_TICKER}", style="bold yellow center"),
            style="gold1"
        )
    )

    # Récupération Statut Alpaca
    account = trading_client.get_account()
    total_equity = float(account.equity)
    cash = float(account.cash)
    pnl = total_equity - 100000.0
    pnl_pct = (pnl / 100000.0) * 100.0

    # Données Temps Réel BTC-USD
    raw_data = yf.download(SINGLE_TICKER, period="5d", interval="1h", progress=False)
    close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
    close_vals = close_series.dropna().values.flatten()
    current_price = close_vals[-1]
    
    log_ret = np.diff(np.log(close_vals))
    seq_input = torch.tensor(log_ret[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_returns = timesfm_model(seq_input).squeeze(0).numpy()

    # Reconstruction de la trajectoire
    cum_returns = np.cumsum(pred_returns)
    predicted_prices = current_price * np.exp(cum_returns)
    expected_gain_pct = ((predicted_prices[-1] - current_price) / current_price) * 100.0

    # Signal & Confiance
    if expected_gain_pct > 0.5 and is_trained:
        signal_text = "[bold green]ORDER BUY (ACHAT)[/bold green]"
        confidence = min(95.0, 60.0 + expected_gain_pct * 10.0)
    elif expected_gain_pct < -0.5:
        signal_text = "[bold red]ORDER SELL (VENTE)[/bold red]"
        confidence = 80.0
    else:
        signal_text = "[bold yellow]HOLD (NEUTRE / CASH)[/bold yellow]"
        confidence = 50.0

    # Récupération Position Ouverte
    positions = trading_client.get_all_positions()
    btc_pos = [p for p in positions if p.symbol in ["BTCUSD", "BTC/USD"]]

    if btc_pos:
        pos_qty = float(btc_pos[0].qty)
        pos_val = float(btc_pos[0].market_value)
        pos_pnl = float(btc_pos[0].unrealized_pl)
        pos_str = f"[bold green]OUVERTE[/bold green] | Qte: {pos_qty:.4f} | Valeur: ${pos_val:,.2f} | P/L: ${pos_pnl:+.2f}"
    else:
        pos_str = "[yellow]AUCUNE (100% CASH LIQUIDE)[/yellow]"

    # Construction de la Table
    table = Table(title=f"📊 MODÈLE TIMESFM FINETUNED | STATUT APPRENTISSAGE: {'[GREEN]ENTRAÎNÉ[/GREEN]' if is_trained else '[YELLOW]ZERO-SHOT[/YELLOW]'}", expand=True)
    table.add_column("Métrique Financière", style="cyan", justify="left")
    table.add_column("Valeur / Signal", style="bold white", justify="right")

    table.add_row("Prix Actuel Bitcoin (BTC-USD)", f"${current_price:,.2f}")
    table.add_row("Trajectoire Prédite TimesFM (+5h)", f"${predicted_prices[-1]:,.2f} ({expected_gain_pct:+.2f}%)")
    table.add_row("Signal de Décision IA", signal_text)
    table.add_row("Confiance IA", f"{confidence:.1f}%")
    table.add_row("Position Alpaca Paper", pos_str)
    table.add_row("Capital Total Portefeuille", f"${total_equity:,.2f}")
    table.add_row("Cash Liquide Disponible", f"${cash:,.2f}")
    table.add_row("Profit / Perte Cumulé", f"${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    layout["main"].update(Panel(table, style="blue"))

    # Footer
    layout["footer"].update(
        Panel(
            Text(f"[OK] Moteur TimesFM Opérationnel | Prochain Scan dans : {remaining_sec}s | (Ctrl+C pour quitter)", style="bold cyan center"),
            style="green"
        )
    )

    return layout

def main():
    parser = argparse.ArgumentParser(description="Bot Single-Asset BTC-USD TimesFM")
    parser.add_argument("--continuous", action="store_true", help="Lancer la boucle continue")
    parser.add_argument("--interval_sec", type=int, default=60, help="Intervalle entre 2 scans")
    parser.add_argument("--sell_all", action="store_true", help="Liquider le portefeuille")
    args = parser.parse_args()

    if args.sell_all:
        execute_sell_all()
        return

    console.print(get_single_asset_live_panel(0))

    if args.continuous:
        with Live(get_single_asset_live_panel(args.interval_sec), refresh_per_second=1, console=console) as live:
            while True:
                start_t = time.time()
                # Execution des signaux
                raw_data = yf.download(SINGLE_TICKER, period="5d", interval="1h", progress=False)
                close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
                close_vals = close_series.dropna().values.flatten()
                current_price = close_vals[-1]
                log_ret = np.diff(np.log(close_vals))
                seq_input = torch.tensor(log_ret[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    pred_returns = timesfm_model(seq_input).squeeze(0).numpy()

                cum_returns = np.cumsum(pred_returns)
                expected_gain_pct = ((current_price * np.exp(cum_returns[-1]) - current_price) / current_price) * 100.0

                if expected_gain_pct > 0.5:
                    signal_data = [{
                        "ticker": SINGLE_TICKER,
                        "action": "BUY",
                        "confidence": min(0.95, 0.60 + expected_gain_pct * 0.1),
                        "price": current_price,
                        "horizon": 1
                    }]
                    execute_trade_signals(signal_data)

                for elapsed in range(args.interval_sec):
                    rem = args.interval_sec - elapsed
                    live.update(get_single_asset_live_panel(rem))
                    time.sleep(1.0)

if __name__ == "__main__":
    main()
