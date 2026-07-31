#!/usr/bin/env python3
"""
Bot de Trading Automatisé Multi-Horizon Temporel Multi-Thématique SOTA (BTC-USD)
Exécute en parallèle 3 modèles Transformer Résiduels 25.8M (1h, 5m, 1m) avec répartition stricte du budget :
- 1 Heure  (1h) : 30% du Cash Portfolio (Modèle Swing)
- 5 Minutes (5m) : 20% du Cash Portfolio (Modèle Intraday - 91.9% WinRate)
- 1 Minute  (1m) : 10% du Cash Portfolio (Modèle Scalping)
- Buffer Cash   : 40% préservé en sécurité constante.
"""

import os
import sys
import time
import json
import threading
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

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.trading_config import console, trading_client, map_symbol_to_alpaca
from src.execution import check_instant_safety_limits, execute_sell_all, execute_trade_signals
from scripts.train_tabfm_residual_multi_asset import MultiAssetResidualTransformer, apply_triple_barrier_and_features

SINGLE_TICKER = "BTC-USD"
ALPACA_SYMBOL = "BTC/USD"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Configuration des budgets et des checkpoints par horizon temporel
TIMEFRAME_CONFIGS = {
    "1h": {
        "name": "1 Heure (Swing)",
        "budget_pct": 0.30,
        "checkpoint": "models/tabfm_residual_BTC_USD.pt",
        "interval": "1h",
        "period": "30d",
        "interval_sec": 3600,
        "min_elapsed_sec": 3000,
        "history_file": "data/history_1h.json"
    },
    "5m": {
        "name": "5 Minutes (Intraday)",
        "budget_pct": 0.20,
        "checkpoint": "models/tabfm_residual_BTC_USD_5m.pt",
        "interval": "5m",
        "period": "30d",
        "interval_sec": 300,
        "min_elapsed_sec": 250,
        "history_file": "data/history_5m.json"
    },
    "1m": {
        "name": "1 Minute (Scalping)",
        "budget_pct": 0.10,
        "checkpoint": "models/tabfm_residual_BTC_USD_1m.pt",
        "interval": "1m",
        "period": "7d",
        "interval_sec": 60,
        "min_elapsed_sec": 50,
        "history_file": "data/history_1m.json"
    }
}

def make_sparkline(vals, length=8):
    if len(vals) < 2:
        return "---"
    sub_vals = vals[-length:]
    min_v, max_v = min(sub_vals), max(sub_vals)
    if max_v == min_v:
        return "▅" * len(sub_vals)
    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    res = ""
    for v in sub_vals:
        idx = int(((v - min_v) / (max_v - min_v)) * 7)
        res += bars[max(0, min(7, idx))]
    return res

# État global partagé entre les threads d'exécution
LIVE_STATES = {
    tf: {
        "confidence": 50.0,
        "action": "HOLD",
        "signal_text": "[yellow]INITIALISATION...[/yellow]",
        "allocated_notional": 0.0,
        "last_update": "Attente...",
        "next_countdown": cfg["interval_sec"],
        "price": 0.0,
        "price_change_pct": 0.0,
        "sparkline": "---"
    }
    for tf, cfg in TIMEFRAME_CONFIGS.items()
}

STATE_LOCK = threading.Lock()

def load_tf_history(history_file: str):
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_tf_history(history_file: str, history):
    os.makedirs("data", exist_ok=True)
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def load_residual_model(tf: str):
    cfg = TIMEFRAME_CONFIGS[tf]
    ckpt = cfg["checkpoint"]
    if not os.path.exists(ckpt):
        ckpt = "models/tabfm_residual_BTC_USD.pt"
    
    if os.path.exists(ckpt):
        try:
            raw_sample = yf.download(SINGLE_TICKER, period=cfg["period"], interval=cfg["interval"], progress=False)
            feat_df = apply_triple_barrier_and_features(raw_sample, apply_prescreen=False, interval=cfg["interval"])
            feat_cols = [c for c in feat_df.columns if c not in ["target_triple_barrier", "future_return", "sma50", "sma200"]]
            
            model = MultiAssetResidualTransformer(len(feat_cols)).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            return model, feat_cols, ckpt, True
        except Exception as e:
            return None, [], ckpt, False
    return None, [], ckpt, False

def update_tf_history(tf: str, current_price: float, confidence: float, action: str):
    cfg = TIMEFRAME_CONFIGS[tf]
    history = load_tf_history(cfg["history_file"])
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    current_key = now_dt.strftime("%Y-%m-%d %H:%M") if tf in ["1m", "5m"] else now_dt.strftime("%Y-%m-%d %H:00")

    # 1. Vérification à H+delta des prédictions précédentes
    for entry in history:
        if entry.get("status") == "PENDING":
            entry_time_str = entry.get("timestamp", "")
            try:
                entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                elapsed_sec = (now_dt - entry_dt).total_seconds()
            except Exception:
                elapsed_sec = cfg["min_elapsed_sec"] + 1.0

            if elapsed_sec >= cfg["min_elapsed_sec"]:
                entry_price = entry["entry_price"]
                chg_pct = ((current_price - entry_price) / entry_price) * 100.0
                entry["exit_price"] = current_price
                entry["exit_time"] = now_str
                entry["change_pct"] = chg_pct
                
                predicted_up = entry["confidence"] >= 58.0
                actual_up = current_price > entry_price
                
                if (predicted_up and actual_up) or (not predicted_up and not actual_up):
                    entry["outcome"] = "🏆 RAISON"
                else:
                    entry["outcome"] = "❌ ERREUR"
                entry["status"] = "COMPLETED"

    # 2. Dédoublonnage sur le même créneau
    already_exists = False
    for entry in history:
        if entry.get("timestamp", "").startswith(current_key[:15]):
            already_exists = True
            entry["confidence"] = confidence
            entry["action"] = action
            break

    if not already_exists:
        new_entry = {
            "timeframe": tf,
            "timestamp": now_str,
            "entry_price": current_price,
            "confidence": confidence,
            "action": action,
            "exit_price": None,
            "exit_time": None,
            "change_pct": 0.0,
            "outcome": f"⌛ EN COURS (+{tf})",
            "status": "PENDING"
        }
        history.append(new_entry)

    if len(history) > 15:
        history = history[-15:]

    save_tf_history(cfg["history_file"], history)

def timeframe_worker(tf: str):
    """
    Worker indépendant par horizon temporel (1h, 5m, 1m).
    """
    cfg = TIMEFRAME_CONFIGS[tf]
    model, feature_cols, ckpt_path, is_trained = load_residual_model(tf)

    while True:
        start_t = time.time()
        try:
            # 1. Obtenir les données Alpaca & Cash disponible
            try:
                account = trading_client.get_account()
                total_cash = float(account.cash)
            except Exception:
                total_cash = 100000.0

            max_tf_budget = total_cash * cfg["budget_pct"]

            # 2. Téléchargement des bougies avec fallback de sécurité
            raw_data = yf.download(SINGLE_TICKER, period=cfg["period"], interval=cfg["interval"], progress=False)
            if raw_data.empty or len(raw_data) < 10:
                raw_data = yf.download(SINGLE_TICKER, period="5d", interval=cfg["interval"], progress=False)

            if raw_data.empty:
                with STATE_LOCK:
                    LIVE_STATES[tf]["signal_text"] = "[yellow]ATTENTE FLUX FLOTTANT...[/yellow]"
                time.sleep(3.0)
                continue

            close_series = raw_data["Close"].iloc[:, 0] if isinstance(raw_data["Close"], pd.DataFrame) else raw_data["Close"]
            close_vals = close_series.dropna().values.flatten()
            if len(close_vals) == 0:
                time.sleep(3.0)
                continue
                
            current_price = close_vals[-1]

            confidence = 50.0
            action = "HOLD"
            allocated_notional = 0.0
            signal_text = "[yellow]NEUTRE / CASH[/yellow]"

            # Position Alpaca
            has_open_pos = False
            try:
                positions = trading_client.get_all_positions()
                btc_pos = [p for p in positions if p.symbol in ["BTCUSD", "BTC/USD"]]
                if btc_pos:
                    has_open_pos = True
            except Exception:
                has_open_pos = False

            # 3. Inférence PyTorch
            if is_trained and model is not None:
                feat_df = apply_triple_barrier_and_features(raw_data, apply_prescreen=False, interval=cfg["interval"])
                if not feat_df.empty:
                    last_feat = feat_df[feature_cols].iloc[-1:]
                    mean = feat_df[feature_cols].mean()
                    std = feat_df[feature_cols].std() + 1e-8
                    last_norm = (last_feat - mean) / std

                    x_input = torch.tensor(last_norm.values, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        prob_val = float(model(x_input).cpu().numpy()[0])

                    confidence = prob_val * 100.0

                    # Allocation Dynamique de Kelly appliquée au plafond du budget (30%, 20% ou 10%)
                    if prob_val >= 0.58:
                        action = "BUY"
                        confidence_ratio = (prob_val - 0.50) / 0.50
                        allocated_notional = min(max_tf_budget, max(500.0, max_tf_budget * confidence_ratio))
                        signal_text = f"[bold green]ACHAT (${allocated_notional:,.0f})[/bold green]"
                        
                        # Exécution Alpaca
                        signal_data = [{
                            "ticker": SINGLE_TICKER,
                            "action": "BUY",
                            "confidence": prob_val,
                            "price": current_price,
                            "horizon": tf
                        }]
                        execute_trade_signals(signal_data, threshold=0.58, notional=allocated_notional, max_budget=max_tf_budget, max_trade_cap=max_tf_budget)
                    elif prob_val <= 0.42:
                        action = "SELL"
                        if has_open_pos:
                            signal_text = "[bold red]ORDER SELL (LIQUIDATION POSITION)[/bold red]"
                        else:
                            signal_text = "[bold yellow]HOLD (100% CASH - IA BAISSIÈRE)[/bold yellow]"
                    else:
                        action = "HOLD"
                        signal_text = "[bold yellow]HOLD (100% CASH - IA NEUTRE)[/bold yellow]"

            # 4. Calcul de l'évolution du prix et graphique sparkline
            prev_p = close_vals[-2] if len(close_vals) >= 2 else current_price
            price_chg_pct = ((current_price - prev_p) / prev_p) * 100.0
            sparkline_str = make_sparkline(close_vals, length=10)

            # 5. Mise à jour de l'historique
            update_tf_history(tf, current_price, confidence, action)

            # 6. Mise à jour de l'état partagé pour la UI Rich
            with STATE_LOCK:
                LIVE_STATES[tf]["confidence"] = confidence
                LIVE_STATES[tf]["action"] = action
                LIVE_STATES[tf]["signal_text"] = signal_text
                LIVE_STATES[tf]["allocated_notional"] = allocated_notional
                LIVE_STATES[tf]["last_update"] = datetime.now().strftime("%H:%M:%S")
                LIVE_STATES[tf]["price"] = current_price
                LIVE_STATES[tf]["price_change_pct"] = price_chg_pct
                LIVE_STATES[tf]["sparkline"] = sparkline_str

        except Exception as e:
            with STATE_LOCK:
                LIVE_STATES[tf]["signal_text"] = f"[red]Erreur: {e}[/red]"

        # Compte à rebours précis sans dérive
        calc_duration = time.time() - start_t
        remaining_wait = max(1, cfg["interval_sec"] - calc_duration)
        
        for elapsed in range(int(remaining_wait)):
            with STATE_LOCK:
                LIVE_STATES[tf]["next_countdown"] = int(remaining_wait - elapsed)
            time.sleep(1.0)

def render_multi_tf_dashboard():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="main", size=14),
        Layout(name="journal", size=12),
        Layout(name="footer", size=3)
    )

    # 1. Compte Alpaca & Prix Temps Réel BTC
    try:
        account = trading_client.get_account()
        total_equity = float(account.equity)
        cash = float(account.cash)
        pnl = total_equity - 100000.0
        pnl_pct = (pnl / 100000.0) * 100.0
    except Exception:
        total_equity, cash, pnl, pnl_pct = 100000.0, 100000.0, 0.0, 0.0

    # Récupérer les données de prix récentes pour les variations 5m et 1h
    latest_price = 0.0
    var_5m_pct = 0.0
    var_1h_pct = 0.0
    with STATE_LOCK:
        if LIVE_STATES["5m"]["price"] > 0:
            latest_price = LIVE_STATES["5m"]["price"]
        elif LIVE_STATES["1m"]["price"] > 0:
            latest_price = LIVE_STATES["1m"]["price"]
        elif LIVE_STATES["1h"]["price"] > 0:
            latest_price = LIVE_STATES["1h"]["price"]

    if latest_price == 0.0:
        try:
            raw_sample = yf.download(SINGLE_TICKER, period="5d", interval="5m", progress=False)
            c_series = raw_sample["Close"].iloc[:, 0] if isinstance(raw_sample["Close"], pd.DataFrame) else raw_sample["Close"]
            c_vals = c_series.dropna().values.flatten()
            latest_price = c_vals[-1]
            p5m = c_vals[-2] if len(c_vals) >= 2 else latest_price
            p1h = c_vals[-13] if len(c_vals) >= 13 else c_vals[0]
            var_5m_pct = ((latest_price - p5m) / p5m) * 100.0
            var_1h_pct = ((latest_price - p1h) / p1h) * 100.0
        except Exception:
            latest_price = 0.0

    style_5m = "green" if var_5m_pct >= 0 else "red"
    style_1h = "green" if var_1h_pct >= 0 else "red"

    header_text = (
        f"[bold white]🏛️ PR-BIA MULTI-TIMEFRAME ENSEMBLE SYSTEM | ALLOCATION STRICTE DE CASH PORTFOLIO[/bold white]\n"
        f"Prix BTC-USD: [bold yellow]${latest_price:,.2f}[/bold yellow] | "
        f"Var 5m: [{style_5m}]{var_5m_pct:+.2f}%[/{style_5m}] | "
        f"Var 1h: [{style_1h}]{var_1h_pct:+.2f}%[/{style_1h}] | "
        f"Capital: [cyan]${total_equity:,.2f}[/cyan] | Cash: [green]${cash:,.2f}[/green] | PnL: [{'green' if pnl>=0 else 'red'}]${pnl:+,.2f} ({pnl_pct:+.2f}%)[/{'green' if pnl>=0 else 'red'}]"
    )

    layout["header"].update(Panel(header_text, style="bold white on blue"))

    # 2. Table Synthèse des 3 Horizons Temporels
    table = Table(title="📊 PORTFEUILLE ENSEMBLE MULTI-HORIZON (1H: 30% | 5M: 20% | 1M: 10% | CASH SAFETY: 40%)", expand=True)
    table.add_column("Horizon", style="cyan", justify="left")
    table.add_column("Prix BTC", style="bold white", justify="right")
    table.add_column("Variation", style="bold white", justify="center")
    table.add_column("Tendance Sparkline", style="bold yellow", justify="center")
    table.add_column("Part Cash", style="bold yellow", justify="center")
    table.add_column("Plafond ($)", style="bold white", justify="right")
    table.add_column("Confiance IA", style="bold cyan", justify="right")
    table.add_column("Signal SOTA", style="bold white", justify="center")
    table.add_column("Allocation Kelly ($)", style="bold green", justify="right")
    table.add_column("Prochain Scan", style="dim", justify="right")

    with STATE_LOCK:
        for tf, cfg in TIMEFRAME_CONFIGS.items():
            st = LIVE_STATES[tf]
            max_b = cash * cfg["budget_pct"]
            price_str = f"${st['price']:,.2f}" if st['price'] > 0 else "---"
            var_pct = st["price_change_pct"]
            var_style = "bold green" if var_pct >= 0 else "bold red"
            var_str = f"[{var_style}]{var_pct:+.2f}%[/{var_style}]" if st['price'] > 0 else "---"
            spark_str = f"[cyan]{st['sparkline']}[/cyan]"
            conf_str = f"{st['confidence']:.1f}%"
            sig_str = st["signal_text"]
            notional_str = f"${st['allocated_notional']:,.2f}" if st['allocated_notional'] > 0 else "0.00$ (CASH)"
            cd_str = f"{st['next_countdown']}s"
            
            table.add_row(
                cfg["name"],
                price_str,
                var_str,
                spark_str,
                f"{cfg['budget_pct']*100:.0f}%",
                f"${max_b:,.2f}",
                conf_str,
                sig_str,
                notional_str,
                cd_str
            )

    layout["main"].update(Panel(table, style="blue"))

    # 3. Journal de Suivi Combiné
    combined_history = []
    for tf, cfg in TIMEFRAME_CONFIGS.items():
        h = load_tf_history(cfg["history_file"])
        for item in h:
            item["tf_name"] = tf
            combined_history.append(item)

    combined_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    j_table = Table(title="📜 JOURNAL COMBINÉ DES PRÉDICTIONS INTER-TIMEFRAMES", expand=True)
    j_table.add_column("Horizon", style="bold cyan", justify="center")
    j_table.add_column("Horodatage", style="dim", justify="left")
    j_table.add_column("Prix Entrée", style="white", justify="right")
    j_table.add_column("Confiance", style="cyan", justify="right")
    j_table.add_column("Action", style="bold white", justify="center")
    j_table.add_column("Prix Clôture", style="white", justify="right")
    j_table.add_column("Variation", style="bold white", justify="right")
    j_table.add_column("Résultat IA", style="bold yellow", justify="center")

    for item in combined_history[:7]:
        tf_code = item.get("tf_name", "").upper()
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
        elif "ERREUR" in out:
            out_style = "[bold red]❌ ERREUR[/bold red]"
        else:
            out_style = f"[bold yellow]⌛ EN COURS (+{tf_code})[/bold yellow]"

        j_table.add_row(tf_code, t_str, f"${p_in:,.2f}", f"{conf:.1f}%", act, p_out_str, chg_str, out_style)

    layout["journal"].update(Panel(j_table, style="blue"))

    layout["footer"].update(Panel(
        "[bold green]✔ Moteur Multi-Horizon 25.8M Actif[/bold green] | [yellow]Threads 1h (30%), 5m (20%), 1m (10%) synchronisés[/yellow] | [white]Appuyez sur Ctrl+C pour quitter[/white]",
        style="bold white on black"
    ))

    return layout

def main():
    parser = argparse.ArgumentParser(description="Bot Multi-Horizon Temporel Ensemble SOTA (BTC-USD)")
    parser.add_argument("--sell_all", action="store_true", help="Fermer immédiatement toutes les positions")
    args = parser.parse_args()

    if args.sell_all:
        execute_sell_all()
        return

    # Lancer les 3 workers indépendants dans des threads d'arrière-plan (échelonnés de 1s)
    for tf in TIMEFRAME_CONFIGS.keys():
        t = threading.Thread(target=timeframe_worker, args=(tf,), daemon=True)
        t.start()
        time.sleep(1.0)

    # Boucle de rendu UI Rich en direct
    with Live(render_multi_tf_dashboard(), refresh_per_second=1, console=console) as live:
        try:
            while True:
                live.update(render_multi_tf_dashboard())
                time.sleep(1.0)
        except KeyboardInterrupt:
            console.print("\n[yellow]Arrêt du Bot Multi-Horizon...[/yellow]")

if __name__ == "__main__":
    main()
