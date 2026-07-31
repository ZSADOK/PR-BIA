#!/usr/bin/env python3
"""
Bot de Trading Automatisé Continu Single-Asset SOTA (BTC-USD) avec Journal de Suivi Dynamique
Intègre le Transformer Résiduel 25.8M, le Pre-Screening Volume/Momentum, et un Suivi Dynamique 1h
(Historique des trades, vérification si le modèle a eu raison/tort, P&L $ et %, WinRate en direct).
"""

import os
import sys
import time
import json
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
from src.trading_config import console, trading_client, map_symbol_to_alpaca
from src.execution import check_instant_safety_limits, execute_sell_all, execute_trade_signals
from scripts.train_tabfm_residual_multi_asset import MultiAssetResidualTransformer, apply_triple_barrier_and_features

SINGLE_TICKER = "BTC-USD"
ALPACA_SYMBOL = "BTC/USD"
CHECKPOINT_PATH = "models/tabfm_residual_BTC_USD.pt"
HISTORY_FILE_PATH = "data/btc_hourly_trades_history.json"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_history():
    if os.path.exists(HISTORY_FILE_PATH):
        try:
            with open(HISTORY_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    os.makedirs("data", exist_ok=True)
    try:
        with open(HISTORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def load_trained_residual_transformer():
    if os.path.exists(CHECKPOINT_PATH):
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

def update_and_verify_hourly_history(current_price: float, current_confidence: float, action: str):
    """
    Vérifie les prédictions passées (1h plus tard) et enregistre le résultat (Raisons/Erreurs & PnL).
    """
    history = load_history()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Mettre à jour la dernière prédiction en attente si 1h s'est écoulée
    for entry in history:
        if entry.get("status") == "PENDING":
            entry_price = entry["entry_price"]
            price_change_pct = ((current_price - entry_price) / entry_price) * 100.0
            
            entry["exit_price"] = current_price
            entry["exit_time"] = now_str
            entry["change_pct"] = price_change_pct
            
            # Vérifier si l'IA a eu raison
            predicted_up = entry["confidence"] >= 58.0
            actual_up = current_price > entry_price
            
            if (predicted_up and actual_up) or (not predicted_up and not actual_up):
                entry["outcome"] = "🏆 RAISON"
            else:
                entry["outcome"] = "❌ ERREUR"
                
            entry["status"] = "COMPLETED"

    # 2. Ajouter la nouvelle prédiction de l'heure courante
    new_entry = {
        "timestamp": now_str,
        "entry_price": current_price,
        "confidence": current_confidence,
        "action": action,
        "exit_price": None,
        "exit_time": None,
        "change_pct": 0.0,
        "outcome": "⌛ EN COURS (+1h)",
        "status": "PENDING"
    }
    history.append(new_entry)
    
    # Conserver les 20 derniers trades pour la clarté
    if len(history) > 20:
        history = history[-20:]
        
    save_history(history)
    return history

def get_single_asset_live_panel(remaining_sec: int = 0) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", size=11),
        Layout(name="journal", size=10),
        Layout(name="footer", size=3)
    )

    # Header
    layout["header"].update(
        Panel(
            Text(f"🏛️ PR-BIA INSTITUTIONAL TERMINAL 24/7 | SPÉCIALISTE BTC-USD (TRANSFORMER 25.8M)", style="bold yellow center"),
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

    # Données Temps Réel BTC-USD
    raw_data = yf.download(SINGLE_TICKER, period="30d", interval="1h", progress=False)
    close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
    close_vals = close_series.dropna().values.flatten()
    current_price = close_vals[-1]
    
    confidence = 50.0
    signal_text = "[bold yellow]HOLD (NEUTRE / CASH)[/bold yellow]"

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
            
            confidence = prob_val * 100.0
            
            if prob_val >= 0.58:
                signal_text = "[bold green]ORDER BUY (ACHAT SOTA)[/bold green]"
            elif prob_val <= 0.42:
                signal_text = "[bold red]ORDER SELL (VENTE / CASH)[/bold red]"
            else:
                signal_text = "[bold yellow]HOLD (NEUTRE / CONFUSION)[/bold yellow]"

    # Position Alpaca
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

    # Table Principale Statut
    table = Table(title=f"📊 ÉTAT DU MODÈLE ET COMPTE ALPACA PAPER | STATUT: {'[green]OPTIMISÉ (CAUM 242.11)[/green]' if is_trained else '[yellow]NON ENTRENÉ[/yellow]'}", expand=True)
    table.add_column("Métrique Financière", style="cyan", justify="left")
    table.add_column("Valeur Temps Réel", style="bold white", justify="right")

    table.add_row("Prix Actuel Bitcoin (BTC-USD)", f"${current_price:,.2f}")
    table.add_row("Probabilité Inférence IA (1h)", f"{confidence:.1f}%")
    table.add_row("Signal de Décision SOTA", signal_text)
    table.add_row("Position Alpaca Paper", pos_str)
    table.add_row("Capital Total Portefeuille", f"${total_equity:,.2f}")
    table.add_row("Cash Liquide Disponible", f"${cash:,.2f}")
    table.add_row("Profit / Perte Cumulé Portefeuille", f"${pnl:+,.2f} ({pnl_pct:+.2f}%)")

    layout["main"].update(Panel(table, style="blue"))

    # Table Journal de Suivi Horaire
    history = load_history()
    j_table = Table(title="📜 JOURNAL DE SUIVI DYNAMIQUE : VÉRIFICATION DES PRÉDICTIONS PAR HEURE", expand=True)
    j_table.add_column("Horodatage", style="dim", justify="left")
    j_table.add_column("Prix Entrée", style="white", justify="right")
    j_table.add_column("Confiance IA", style="cyan", justify="right")
    j_table.add_column("Action", style="bold white", justify="center")
    j_table.add_column("Prix Clôture (+1h)", style="white", justify="right")
    j_table.add_column("Variation 1h", style="bold white", justify="right")
    j_table.add_column("Résultat IA", style="bold yellow", justify="center")

    completed_wins = 0
    completed_total = 0

    for item in history[-7:]:  # Afficher les 7 derniers événements
        t_str = item.get("timestamp", "").split(" ")[1] if " " in item.get("timestamp", "") else item.get("timestamp", "")
        p_in = item.get("entry_price", 0.0)
        conf = item.get("confidence", 50.0)
        act = item.get("action", "HOLD")
        p_out = item.get("exit_price")
        chg = item.get("change_pct", 0.0)
        out = item.get("outcome", "⌛ EN COURS")

        p_out_str = f"${p_out:,.2f}" if p_out else "---"
        chg_str = f"{chg:+.2f}%" if p_out else "---"

        if "RAISON" in out:
            out_style = "[bold green]🏆 RAISON[/bold green]"
            completed_wins += 1
            completed_total += 1
        elif "ERREUR" in out:
            out_style = "[bold red]❌ ERREUR[/bold red]"
            completed_total += 1
        else:
            out_style = "[bold yellow]⌛ EN COURS (+1h)[/bold yellow]"

        j_table.add_row(t_str, f"${p_in:,.2f}", f"{conf:.1f}%", act, p_out_str, chg_str, out_style)

    realized_wr = (completed_wins / completed_total * 100.0) if completed_total > 0 else 0.0
    layout["journal"].update(Panel(j_table, title=f"[bold green]WinRate Historique Réel Enregistré : {realized_wr:.1f}% ({completed_wins}/{completed_total})[/bold green]", style="magenta"))

    # Footer
    rem_min = remaining_sec // 60
    rem_s = remaining_sec % 60
    layout["footer"].update(
        Panel(
            Text(f"[OK] Moteur 25.8M Transformer Opérationnel | Prochaine Clôture Horaire dans : {rem_min}m {rem_s}s | (Ctrl+C pour quitter)", style="bold cyan center"),
            style="green"
        )
    )

    return layout

def main():
    parser = argparse.ArgumentParser(description="Bot Single-Asset BTC-USD Transformer Résiduel 25.8M avec Suivi Dynamique")
    parser.add_argument("--continuous", action="store_true", help="Lancer la boucle continue")
    parser.add_argument("--interval_sec", type=int, default=3600, help="Intervalle en secondes (défaut: 3600s = 1h)")
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
                
                confidence = 50.0
                action = "HOLD"

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

                        confidence = prob_val * 100.0

                        if prob_val >= 0.58:
                            action = "BUY"
                            signal_data = [{
                                "ticker": SINGLE_TICKER,
                                "action": "BUY",
                                "confidence": prob_val,
                                "price": current_price,
                                "horizon": 1
                            }]
                            execute_trade_signals(signal_data)
                        elif prob_val <= 0.42:
                            action = "SELL"
                        else:
                            action = "HOLD"

                # Mise à jour et vérification du journal de suivi
                update_and_verify_hourly_history(current_price, confidence, action)

                # Compte à rebours 1 heure (3600s)
                for elapsed in range(args.interval_sec):
                    rem = args.interval_sec - elapsed
                    live.update(get_single_asset_live_panel(rem))
                    time.sleep(1.0)

if __name__ == "__main__":
    main()
