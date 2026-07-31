#!/usr/bin/env python3
"""
Bot de Trading Automatisé Continu Single-Asset SOTA (BTC-USD)
Intègre le Transformer Résiduel SOTA (25.8M de paramètres), le Pre-Screening Volume & Momentum, et le Terminal Live Rich 24/7.
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
from scripts.train_tabfm_residual_multi_asset import MultiAssetResidualTransformer, apply_triple_barrier_and_features

SINGLE_TICKER = "BTC-USD"
ALPACA_SYMBOL = "BTC/USD"
CHECKPOINT_PATH = "models/tabfm_residual_BTC_USD.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_trained_residual_transformer():
    if os.path.exists(CHECKPOINT_PATH):
        # Sample features to get input_dim
        raw_sample = yf.download(SINGLE_TICKER, period="30d", interval="1h", progress=False)
        feat_df = apply_triple_barrier_and_features(raw_sample, apply_prescreen=False)
        feat_cols = [c for c in feat_df.columns if c not in ["target_triple_barrier", "future_return", "sma50", "sma200"]]
        
        model = MultiAssetResidualTransformer(len(feat_cols)).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()
        return model, feat_cols, True
    else:
        return None, [], False

residual_model, feature_cols, is_trained = load_trained_residual_transformer()

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
    try:
        account = trading_client.get_account()
        total_equity = float(account.equity)
        cash = float(account.cash)
        pnl = total_equity - 100000.0
        pnl_pct = (pnl / 100000.0) * 100.0
    except Exception:
        total_equity, cash, pnl, pnl_pct = 100000.0, 100000.0, 0.0, 0.0

    # Données Temps Réel BTC-USD (historique 30 jours pour indicateurs 1h)
    raw_data = yf.download(SINGLE_TICKER, period="30d", interval="1h", progress=False)
    close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
    close_vals = close_series.dropna().values.flatten()
    current_price = close_vals[-1]
    
    confidence = 0.50
    signal_text = "[bold yellow]HOLD (NEUTRE / CASH)[/bold yellow]"

    if is_trained and residual_model is not None:
        feat_df = apply_triple_barrier_and_features(raw_data, apply_prescreen=False)
        if not feat_df.empty:
            last_feat = feat_df[feature_cols].iloc[-1:]
            
            # Normalisation basée sur l'historique récent
            mean = feat_df[feature_cols].mean()
            std = feat_df[feature_cols].std() + 1e-8
            last_norm = (last_feat - mean) / std
            
            x_input = torch.tensor(last_norm.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                prob_val = float(residual_model(x_input).cpu().numpy()[0])
            
            confidence = prob_val * 100.0
            
            if prob_val >= 0.58:
                signal_text = "[bold green]ORDER BUY (ACHAT SOTA)[/bold green]"
            elif prob_val <= 0.42:
                signal_text = "[bold red]ORDER SELL (VENTE / CASH)[/bold red]"
            else:
                signal_text = "[bold yellow]HOLD (NEUTRE / CONFUSION)[/bold yellow]"

    # Récupération Position Ouverte
    try:
        positions = trading_client.get_all_positions()
        btc_pos = [p for p in positions if p.symbol in ["BTCUSD", "BTC/USD"]]

        if btc_pos:
            pos_qty = float(btc_pos[0].qty)
            pos_val = float(btc_pos[0].market_value)
            pos_pnl = float(btc_pos[0].unrealized_pl)
            pos_str = f"[bold green]OUVERTE[/bold green] | Qte: {pos_qty:.4f} | Valeur: ${pos_val:,.2f} | P/L: ${pos_pnl:+.2f}"
        else:
            pos_str = "[yellow]AUCUNE (100% CASH LIQUIDE)[/yellow]"
    except Exception:
        pos_str = "[yellow]NON CONNECTÉ ALPACA[/yellow]"

    # Construction de la Table
    table = Table(title=f"📊 MODÈLE TRANSFORMER RÉSIDUEL (25.8M) | STATUT: {'[green]ENTRAÎNÉ (CHECKPOINT CAUM 242.11)[/green]' if is_trained else '[yellow]NON DÉTECTÉ[/yellow]'}", expand=True)
    table.add_column("Métrique Financière", style="cyan", justify="left")
    table.add_column("Valeur / Signal", style="bold white", justify="right")

    table.add_row("Prix Actuel Bitcoin (BTC-USD)", f"${current_price:,.2f}")
    table.add_row("Probabilité Inférence IA (Breakout 1h)", f"{confidence:.1f}%")
    table.add_row("Signal de Décision SOTA", signal_text)
    table.add_row("Seuil Déclenchement Achat (Threshold)", "58.0%")
    table.add_row("Position Alpaca Paper", pos_str)
    table.add_row("Capital Total Portefeuille", f"${total_equity:,.2f}")
    table.add_row("Cash Liquide Disponible", f"${cash:,.2f}")
    table.add_row("Profit / Perte Cumulé", f"${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    layout["main"].update(Panel(table, style="blue"))

    # Footer
    layout["footer"].update(
        Panel(
            Text(f"[OK] Moteur 25.8M Transformer Opérationnel | Prochain Scan dans : {remaining_sec}s | (Ctrl+C pour quitter)", style="bold cyan center"),
            style="green"
        )
    )

    return layout

def main():
    parser = argparse.ArgumentParser(description="Bot Single-Asset BTC-USD Transformer Résiduel 25.8M")
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
                raw_data = yf.download(SINGLE_TICKER, period="30d", interval="1h", progress=False)
                close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
                close_vals = close_series.dropna().values.flatten()
                current_price = close_vals[-1]

                if is_trained and residual_model is not None:
                    feat_df = apply_triple_barrier_and_features(raw_data, apply_prescreen=False)
                    if not feat_df.empty:
                        last_feat = feat_df[feature_cols].iloc[-1:]
                        mean = feat_df[feature_cols].mean()
                        std = feat_df[feature_cols].std() + 1e-8
                        last_norm = (last_feat - mean) / std
                        
                        x_input = torch.tensor(last_norm.values, dtype=torch.float32).to(device)
                        with torch.no_grad():
                            prob_val = float(residual_model(x_input).cpu().numpy()[0])

                        if prob_val >= 0.58:
                            signal_data = [{
                                "ticker": SINGLE_TICKER,
                                "action": "BUY",
                                "confidence": prob_val,
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
