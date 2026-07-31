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
        "period": "2d",
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

def update_tf_history(tf: str, current_price: float, confidence: float, action: str, allocated_notional: float = 0.0):
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
                entry_price = entry.get("entry_price", entry.get("price", current_price))
                if entry_price > 0:
                    chg_pct = ((current_price - entry_price) / entry_price) * 100.0
                else:
                    chg_pct = 0.0

                entry["exit_price"] = current_price
                entry["exit_time"] = now_str
                entry["change_pct"] = chg_pct
                
                entry_conf = float(entry.get("confidence", entry.get("Confiance", 50.0)))
                entry_act = entry.get("action", "HOLD")
                entry_notional = float(entry.get("allocated_notional", 0.0))
                
                # Calcul du Gain/Perte Réalisé en $
                if entry_act == "BUY":
                    trade_pnl_dollar = entry_notional * (chg_pct / 100.0) if entry_notional > 0 else 0.0
                    actual_up = current_price > entry_price
                    entry["outcome"] = "🏆 RAISON" if actual_up else "❌ ERREUR"
                else:
                    trade_pnl_dollar = 0.0
                    actual_down_or_flat = current_price <= entry_price
                    entry["outcome"] = "🏆 RAISON" if actual_down_or_flat else "❌ ERREUR"

                entry["pnl_dollar"] = trade_pnl_dollar
                entry["status"] = "COMPLETED"

    # 2. Dédoublonnage strict : verrouiller la première prédiction de la bougie
    already_exists = False
    for entry in history:
        if entry.get("timestamp", "").startswith(current_key[:15]):
            already_exists = True
            break

    if not already_exists:
        new_entry = {
            "timeframe": tf,
            "timestamp": now_str,
            "entry_price": current_price,
            "confidence": confidence,
            "action": action,
            "allocated_notional": allocated_notional,
            "exit_price": None,
            "exit_time": None,
            "change_pct": 0.0,
            "pnl_dollar": 0.0,
            "outcome": f"⌛ EN COURS (+{tf})",
            "status": "PENDING"
        }
        history.append(new_entry)

    if len(history) > 15:
        history = history[-15:]

    save_tf_history(cfg["history_file"], history)

# État global partagé du compte Alpaca pour la UI (mis à jour en arrière-plan)
ACCOUNT_STATE = {
    "equity": 100000.0,
    "cash": 100000.0,
    "pnl": 0.0,
    "pnl_pct": 0.0,
    "latest_price": 0.0,
    "var_5m_pct": 0.0,
    "var_1h_pct": 0.0
}

def account_updater_worker():
    """
    Thread d'arrière-plan dédié pour la mise à jour sans blocage du compte Alpaca et des prix.
    """
    while True:
        try:
            account = trading_client.get_account()
            eq = float(account.equity)
            cs = float(account.cash)
            pnl_val = eq - 100000.0
            pnl_p = (pnl_val / 100000.0) * 100.0

            with STATE_LOCK:
                ACCOUNT_STATE["equity"] = eq
                ACCOUNT_STATE["cash"] = cs
                ACCOUNT_STATE["pnl"] = pnl_val
                ACCOUNT_STATE["pnl_pct"] = pnl_p
        except Exception:
            pass

        time.sleep(5.0)

def timeframe_worker(tf: str):
    """
    Worker indépendant par horizon temporel (1h, 5m, 1m).
    """
    cfg = TIMEFRAME_CONFIGS[tf]
    model, feature_cols, ckpt_path, is_trained = load_residual_model(tf)

    while True:
        start_t = time.time()
        try:
            # 1. Obtenir le Cash disponible
            with STATE_LOCK:
                total_cash = ACCOUNT_STATE["cash"]

            max_tf_budget = total_cash * cfg["budget_pct"]

            # 2. Téléchargement rapide des bougies avec fallback de sécurité
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

            # 3. Inférence PyTorch ultra-rapide sur tail(1000)
            if is_trained and model is not None:
                fast_data = raw_data.tail(1000)
                feat_df = apply_triple_barrier_and_features(fast_data, apply_prescreen=False, interval=cfg["interval"])
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
                            signal_text = "[bold red]ORDRE VENTE[/bold red]"
                        else:
                            signal_text = "[bold yellow]CASH (BAISSIÈRE)[/bold yellow]"
                    else:
                        action = "HOLD"
                        signal_text = "[bold yellow]CASH (NEUTRE)[/bold yellow]"

            # 4. Calcul de l'évolution du prix et graphique sparkline
            prev_p = close_vals[-2] if len(close_vals) >= 2 else current_price
            price_chg_pct = ((current_price - prev_p) / prev_p) * 100.0
            sparkline_str = make_sparkline(close_vals, length=8)

            # 5. Mise à jour de l'historique
            update_tf_history(tf, current_price, confidence, action, allocated_notional)

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
        Layout(name="header", size=3),
        Layout(name="main", size=6),
        Layout(name="journal", size=18),
        Layout(name="footer", size=2)
    )

    # 1. Calcul des statistiques cumulées de session (1h, 5m, 1m & Global)
    global_pnl = 0.0
    global_wins = 0
    global_completed = 0
    tf_stats = {}

    for tf_key in ["1h", "5m", "1m"]:
        cfg = TIMEFRAME_CONFIGS[tf_key]
        history = load_tf_history(cfg["history_file"])
        
        completed_items = [x for x in history if x.get("status") == "COMPLETED"]
        tf_pnl = sum(x.get("pnl_dollar", 0.0) for x in completed_items)
        tf_wins = sum(1 for x in completed_items if "RAISON" in x.get("outcome", ""))
        tf_comp = len(completed_items)
        tf_winrate = (tf_wins / tf_comp * 100.0) if tf_comp > 0 else 0.0

        tf_stats[tf_key] = {
            "pnl": tf_pnl,
            "wins": tf_wins,
            "completed": tf_comp,
            "winrate": tf_winrate
        }

        global_pnl += tf_pnl
        global_wins += tf_wins
        global_completed += tf_comp

    global_winrate = (global_wins / global_completed * 100.0) if global_completed > 0 else 0.0

    # 2. État du compte en mémoire (0ms, aucun appel réseau dans le rendu UI)
    with STATE_LOCK:
        total_equity = ACCOUNT_STATE["equity"]
        cash = ACCOUNT_STATE["cash"]

        latest_price = 0.0
        if LIVE_STATES["1m"]["price"] > 0:
            latest_price = LIVE_STATES["1m"]["price"]
        elif LIVE_STATES["5m"]["price"] > 0:
            latest_price = LIVE_STATES["5m"]["price"]
        elif LIVE_STATES["1h"]["price"] > 0:
            latest_price = LIVE_STATES["1h"]["price"]

        var_5m_pct = LIVE_STATES["5m"]["price_change_pct"]
        var_1h_pct = LIVE_STATES["1h"]["price_change_pct"]

    style_5m = "green" if var_5m_pct >= 0 else "red"
    style_1h = "green" if var_1h_pct >= 0 else "red"
    g_pnl_style = "bold green" if global_pnl >= 0 else "bold red"

    header_text = (
        f"[bold white]🏛️ PR-BIA MULTI-TIMEFRAME ENSEMBLE SYSTEM | BILAN DE SESSION CONTINU[/bold white]\n"
        f"Prix BTC: [bold yellow]${latest_price:,.2f}[/bold yellow] | "
        f"Var 5m: [{style_5m}]{var_5m_pct:+.2f}%[/{style_5m}] | Var 1h: [{style_1h}]{var_1h_pct:+.2f}%[/{style_1h}] | "
        f"Capital: [cyan]${total_equity:,.2f}[/cyan] | Cash: [green]${cash:,.2f}[/green] | "
        f"PnL Session: [{g_pnl_style}]${global_pnl:+,.2f}[/{g_pnl_style}] | "
        f"Ratio Global: [bold yellow]{global_winrate:.0f}% ({global_wins}/{global_completed} Gagnés)[/bold yellow]"
    )

    layout["header"].update(Panel(header_text, style="bold white on blue"))

    # 3. Table Synthèse des 3 Horizons Temporels (Directement dans layout sans Panel lourd)
    table = Table(title="📊 PORTFEUILLE ENSEMBLE MULTI-HORIZON (1H: 30% | 5M: 20% | 1M: 10% | CASH SAFETY: 40%)", expand=True)
    table.add_column("Horizon", style="cyan", justify="left", no_wrap=True)
    table.add_column("Prix BTC", style="bold white", justify="right", no_wrap=True)
    table.add_column("Var %", style="bold white", justify="center", no_wrap=True)
    table.add_column("Part (Plafond)", style="bold yellow", justify="center", no_wrap=True)
    table.add_column("PnL Session", style="bold white", justify="right", no_wrap=True)
    table.add_column("WinRate", style="bold yellow", justify="center", no_wrap=True)
    table.add_column("Confiance", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Signal SOTA", style="bold white", justify="center", no_wrap=True)
    table.add_column("Scan", style="dim", justify="right", no_wrap=True)

    with STATE_LOCK:
        for tf, cfg in TIMEFRAME_CONFIGS.items():
            st = LIVE_STATES[tf]
            max_b = cash * cfg["budget_pct"]
            price_str = f"${st['price']:,.2f}" if st['price'] > 0 else "---"
            var_pct = st["price_change_pct"]
            var_style = "bold green" if var_pct >= 0 else "bold red"
            var_str = f"[{var_style}]{var_pct:+.2f}%[/{var_style}]" if st['price'] > 0 else "---"
            conf_str = f"{st['confidence']:.1f}%"
            sig_str = st["signal_text"]
            cd_str = f"{st['next_countdown']}s"
            
            st_tf = tf_stats[tf]
            pnl_tf_val = st_tf["pnl"]
            pnl_tf_style = "bold green" if pnl_tf_val >= 0 else "bold red"
            pnl_tf_str = f"[{pnl_tf_style}]${pnl_tf_val:+,.2f}[/{pnl_tf_style}]"
            wr_tf_str = f"{st_tf['winrate']:.0f}% ({st_tf['wins']}/{st_tf['completed']})"

            tf_short_name = "1h (Swing)" if tf == "1h" else "5m (Intraday)" if tf == "5m" else "1m (Scalping)"
            part_str = f"{cfg['budget_pct']*100:.0f}% (${max_b/1000:.1f}k)"
            
            table.add_row(
                tf_short_name,
                price_str,
                var_str,
                part_str,
                pnl_tf_str,
                wr_tf_str,
                conf_str,
                sig_str,
                cd_str
            )

    layout["main"].update(table)

    # 4. 3 Tableaux Séparés et Empilés Verticalement pour 1h, 5m, et 1m
    journal_layout = Layout()
    journal_layout.split_column(
        Layout(name="j_1h", ratio=1),
        Layout(name="j_5m", ratio=1),
        Layout(name="j_1m", ratio=1)
    )

    for tf_key, tf_name in [("1h", "j_1h"), ("5m", "j_5m"), ("1m", "j_1m")]:
        cfg = TIMEFRAME_CONFIGS[tf_key]
        history = load_tf_history(cfg["history_file"])
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        st_tf = tf_stats[tf_key]
        p_style = "bold green" if st_tf["pnl"] >= 0 else "bold red"
        pnl_title_str = f"[{p_style}]${st_tf['pnl']:+,.2f}[/{p_style}]"
        wr_title_str = f"{st_tf['winrate']:.0f}% ({st_tf['wins']}/{st_tf['completed']})"

        tf_label = "1H (SWING)" if tf_key == "1h" else "5M (INTRADAY)" if tf_key == "5m" else "1M (SCALPING)"
        title_text = f"📜 JOURNAL {tf_label} | PnL Session: {pnl_title_str} | WinRate: {wr_title_str}"

        j_table = Table(title=title_text, expand=True, show_header=True)
        j_table.add_column("Horodatage", style="dim", justify="left", no_wrap=True)
        j_table.add_column("Prix Entrée", style="white", justify="right", no_wrap=True)
        j_table.add_column("Confiance", style="cyan", justify="right", no_wrap=True)
        j_table.add_column("Signal SOTA", style="bold white", justify="center", no_wrap=True)
        j_table.add_column("Prix Clôture", style="white", justify="right", no_wrap=True)
        j_table.add_column("Variation", style="bold white", justify="right", no_wrap=True)
        j_table.add_column("Gain/Perte ($)", style="bold white", justify="right", no_wrap=True)
        j_table.add_column("Résultat IA", style="bold yellow", justify="center", no_wrap=True)

        for item in history[:3]:
            t_str = item.get("timestamp", "")
            p_in = item.get("entry_price", 0.0)
            conf = item.get("confidence", 50.0)
            act = item.get("action", "HOLD")
            p_out = item.get("exit_price")
            chg = item.get("change_pct", 0.0)
            pnl_dlr = item.get("pnl_dollar", 0.0)
            out = item.get("outcome", "⌛ EN COURS")

            p_out_str = f"${p_out:,.2f}" if p_out else "---"
            chg_style = "bold green" if chg >= 0 else "bold red"
            chg_str = f"[{chg_style}]{chg:+.2f}%[/{chg_style}]" if p_out else "---"

            if p_out:
                pnl_style = "bold green" if pnl_dlr >= 0 else "bold red"
                pnl_str = f"[{pnl_style}]${pnl_dlr:+,.2f}[/{pnl_style}]"
            else:
                pnl_str = "---"

            if "RAISON" in out:
                out_style = "[bold green]🏆 RAISON[/bold green]"
            elif "ERREUR" in out:
                out_style = "[bold red]❌ ERREUR[/bold red]"
            else:
                out_style = f"[bold yellow]⌛ EN COURS[/bold yellow]"

            act_str = f"[bold green]ACHAT ({conf:.0f}%)[/bold green]" if act == "BUY" else f"[bold yellow]{act}[/bold yellow]"

            j_table.add_row(t_str, f"${p_in:,.2f}", f"{conf:.1f}%", act_str, p_out_str, chg_str, pnl_str, out_style)

        journal_layout[tf_name].update(j_table)

    layout["journal"].update(journal_layout)

    layout["footer"].update(Panel(
        "[bold green]✔ Moteur Multi-Horizon 25.8M Actif (Rafraîchissement 4Hz)[/bold green] | [yellow]Threads 1h (30%), 5m (20%), 1m (10%) synchronisés[/yellow] | [white]Appuyez sur Ctrl+C pour quitter[/white]",
        style="bold white on black"
    ))

    return layout

    layout["footer"].update(Panel(
        "[bold green]✔ Moteur Multi-Horizon 25.8M Actif (Rafraîchissement 4Hz)[/bold green] | [yellow]Threads 1h (30%), 5m (20%), 1m (10%) synchronisés[/yellow] | [white]Appuyez sur Ctrl+C pour quitter[/white]",
        style="bold white on black"
    ))

    return layout

def main():
    parser = argparse.ArgumentParser(description="Bot Multi-Horizon Temporel Ensemble SOTA (BTC-USD)")
    parser.add_argument("--sell_all", action="store_true", help="Fermer immédiatement toutes les positions")
    parser.add_argument("--reset", "-r", action="store_true", help="Réinitialiser et effacer tout l'historique avant de démarrer")
    args = parser.parse_args()

    if args.sell_all:
        execute_sell_all()
        return

    if args.reset:
        console.print("[yellow]🧹 Option --reset détectée: Réinitialisation complète des fichiers d'historique...[/yellow]")
        for tf, cfg in TIMEFRAME_CONFIGS.items():
            if os.path.exists(cfg["history_file"]):
                try:
                    os.remove(cfg["history_file"])
                except Exception:
                    pass
        for extra_f in ["data/btc_hourly_trades_history.json", "data/bot_persistent_state.json"]:
            if os.path.exists(extra_f):
                try:
                    os.remove(extra_f)
                except Exception:
                    pass
        console.print("[bold green]✔ Historiques effacés. Démarrage vierge et synchronisé des 3 horizons![/bold green]\n")

    # Lancer le worker d'actualisation de compte Alpaca sans blocage
    t_acc = threading.Thread(target=account_updater_worker, daemon=True)
    t_acc.start()

    # Lancer les 3 workers indépendants dans des threads d'arrière-plan (échelonnés de 1s)
    for tf in TIMEFRAME_CONFIGS.keys():
        t = threading.Thread(target=timeframe_worker, args=(tf,), daemon=True)
        t.start()
        time.sleep(1.0)

    # Boucle de rendu UI Rich en direct (Rafraîchissement ultra-rapide 4Hz)
    with Live(render_multi_tf_dashboard(), refresh_per_second=4, console=console) as live:
        try:
            while True:
                live.update(render_multi_tf_dashboard())
                time.sleep(0.25)
        except KeyboardInterrupt:
            console.print("\n[yellow]Arrêt du Bot Multi-Horizon...[/yellow]")

if __name__ == "__main__":
    main()
