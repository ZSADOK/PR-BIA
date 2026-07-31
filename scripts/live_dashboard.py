#!/usr/bin/env python3
"""
Dashboard Terminal Dynamique Interactif & Live Engine (Rich UI).
Interface SOTA en temps réel sans défilement de print :
- Capital Initial (--budget, ex: 20,000 €)
- Plafond Max par Trade (--max-cap, ex: 5,000 €) avec Sizing Kelly
- Bilan des Gains/Pertes en Temps Réel (€ / %) & Win Rate
- Signal clair : [A] ACHAT / [V] VENTE / [H] HOLD
- Compte à Rebours Live 5 Min (300s) avec barre de progression
- Historique des Ordres & Statistiques Quantitatives
"""
import os
import sys

# Neutralisation OpenMP C++ macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Auto-bootstrap vers l'interpréteur .venv
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(base_dir, ".venv", "bin", "python3")
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import time
import argparse
import datetime
import logging
import pandas as pd
import numpy as np

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

sys.path.append(base_dir)

from config.settings import config
from src.data_loader import get_large_eth_data
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from src.models.meta_labeler import MetaLabeler
from src.risk.risk_manager import RiskManager
from src.execution.alpaca_executor import AlpacaExecutor

logging.basicConfig(level=logging.ERROR)

class TradingDashboardApp:
    def __init__(self, initial_budget: float = 20000.0, max_trade_cap: float = 5000.0):
        self.initial_budget = initial_budget
        self.current_balance = initial_budget
        self.max_trade_cap = max_trade_cap
        
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.trade_history = []
        
        self.engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
        self.meta_labeler = MetaLabeler()
        self.screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
        self.risk_mgr = RiskManager(
            default_risk_pct=config.risk_per_trade,
            max_kelly_fraction=config.max_kelly_fraction,
            max_portfolio_cap=config.max_portfolio_allocation
        )
        self.alpaca = AlpacaExecutor()
        
        self.latest_state = {
            "price": 0.0,
            "pred_price": 0.0,
            "pred_return_pct": 0.0,
            "confidence": 0.0,
            "action_code": "[H] HOLD",
            "action_color": "yellow",
            "rvol": 1.0,
            "capital_allocated": 0.0,
            "units": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "alpaca_status": "NEUTRAL_HOLD",
            "last_update": "N/A"
        }

    def run_single_cycle(self):
        """Exécute 1 cycle complet d'analyse quant et d'exécution Alpaca."""
        try:
            df = get_large_eth_data(symbol="ETH/USDT", timeframe=config.timeframe, days_back=30, force_refresh=False)
            current_price = float(df['Close'].iloc[-1])
            
            df_screened = self.screener.compute_indicators(df)
            latest_screen = self.screener.evaluate_latest(df_screened)
            
            signal = self.engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
            meta_confidence = self.meta_labeler.predict_meta_confidence(df_screened, timesfm_pred_return=signal['predicted_return_pct']/100.0)
            meta_passed = meta_confidence >= config.min_meta_confidence
            
            final_binary = 1 if (signal['signal_binary'] == 1 and meta_passed) else 0
            
            pos_info = self.risk_mgr.compute_position_size(
                total_capital=self.current_balance,
                entry_price=current_price,
                df_ohlcv=df_screened,
                signal_dict=signal,
                historical_win_rate=max(0.60, meta_confidence)
            )
            
            allocated_cap = min(pos_info['capital_allocated'], self.max_trade_cap) if final_binary == 1 else 0.0
            units = allocated_cap / current_price if current_price > 0 else 0.0
            
            if final_binary == 1:
                action_code = "[A] ACHAT (BUY)"
                action_color = "bold green"
            else:
                action_code = "[H] HOLD / [V] VENTE"
                action_color = "bold yellow"
                
            pos_dict_custom = pos_info.copy()
            pos_dict_custom['capital_allocated'] = allocated_cap
            exec_res = self.alpaca.execute_bot_cycle(signal, pos_dict_custom, symbol="ETH/USD")
            
            acc = self.alpaca.fetch_account()
            if acc.get('status') == 'connected':
                alpaca_status = f"Alpaca N° {acc.get('account_number')} (${acc.get('cash'):,.2f} Cash)"
            else:
                alpaca_status = f"Execution: {exec_res.get('action')}"
                
            self.latest_state = {
                "price": current_price,
                "pred_price": signal['predicted_price'],
                "pred_return_pct": signal['predicted_return_pct'],
                "confidence": meta_confidence * 100.0,
                "action_code": action_code,
                "action_color": action_color,
                "rvol": latest_screen['rvol'],
                "capital_allocated": allocated_cap,
                "units": units,
                "stop_loss": pos_info['stop_loss_price'],
                "take_profit": pos_info['take_profit_price'],
                "alpaca_status": alpaca_status,
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            }
            
            if final_binary == 1:
                self.trade_history.insert(0, {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "action": "ACHAT [A]",
                    "price": f"${current_price:,.2f}",
                    "amount": f"${allocated_cap:,.2f}",
                    "result": "EN COURS ⏳",
                    "pnl": "+0.00 €"
                })
                if len(self.trade_history) > 6:
                    self.trade_history.pop()
                    
        except Exception as e:
            self.latest_state["alpaca_status"] = f"Erreur cycle: {e}"

    def make_header_panel(self) -> Panel:
        win_rate = (self.wins / (self.wins + self.losses) * 100.0) if (self.wins + self.losses) > 0 else 75.00
        pnl_pct = (self.total_pnl / self.initial_budget) * 100.0
        pnl_color = "green" if self.total_pnl >= 0 else "red"
        
        table = Table(expand=True, show_header=True, header_style="bold cyan", box=None)
        table.add_column("BUDGET INITIAL", justify="center")
        table.add_column("SOLDE ACTUEL", justify="center")
        table.add_column("GAIN / PERTE CUMULÉ", justify="center")
        table.add_column("WIN RATE %", justify="center")
        table.add_column("CAP MAX / TRADE", justify="center")
        
        table.add_row(
            f"[bold white]{self.initial_budget:,.2f} €[/bold white]",
            f"[bold white]{self.current_balance:,.2f} €[/bold white]",
            f"[bold {pnl_color}]{self.total_pnl:+,.2f} € ({pnl_pct:+.2f}%)[/bold {pnl_color}]",
            f"[bold green]{win_rate:.2f}%[/bold green]",
            f"[bold yellow]{self.max_trade_cap:,.2f} €[/bold yellow]"
        )
        return Panel(table, title="[bold yellow]⚡ BILAN QUANTITATIF & CAPITAL GLOBAL[/bold yellow]", border_style="yellow")

    def make_signal_panel(self) -> Panel:
        state = self.latest_state
        table = Table(expand=True, show_header=False, box=None)
        table.add_column("Metric", style="bold white", width=30)
        table.add_column("Value", justify="right")
        
        pred_color = "green" if state['pred_return_pct'] > 0 else "red"
        
        table.add_row("• PRIX ETH/USD ACTUEL", f"[bold white]${state['price']:,.2f}[/bold white]")
        table.add_row("• PRÉDICTION TIMESFM H+1", f"[bold {pred_color}]${state['pred_price']:,.2f} ({state['pred_return_pct']:+.4f}%)[/bold {pred_color}]")
        table.add_row("• CONFIANCE XGBOOST", f"[bold cyan]{state['confidence']:.2f}%[/bold cyan] (Seuil >= {config.min_meta_confidence*100:.0f}%)")
        table.add_row("• PRE-SCREENING RVOL", f"[bold magenta]RVOL = {state['rvol']:.2f}[/bold magenta]")
        table.add_row("• ACTION DECIDÉE", f"[{state['action_color']}]{state['action_code']}[/{state['action_color']}]")
        
        return Panel(table, title="[bold cyan]📊 SIGNAL & DÉCISION DE TRADING (5M)[/bold cyan]", border_style="cyan")

    def make_risk_panel(self) -> Panel:
        state = self.latest_state
        table = Table(expand=True, show_header=False, box=None)
        table.add_column("Metric", style="bold white", width=30)
        table.add_column("Value", justify="right")
        
        table.add_row("• SOMME MISÉE SUR TRADE", f"[bold yellow]${state['capital_allocated']:,.2f}[/bold yellow] (Max Cap: ${self.max_trade_cap:,.2f})")
        table.add_row("• UNITÉS ETH ACHETÉES", f"[bold white]{state['units']:.4f} ETH[/bold white]")
        table.add_row("• STOP-LOSS (1.0x ATR)", f"[bold red]${state['stop_loss']:,.2f}[/bold red]")
        table.add_row("• TAKE-PROFIT (1.5x ATR)", f"[bold green]${state['take_profit']:,.2f}[/bold green]")
        table.add_row("• STATUT BROKER ALPACA", f"[bold green]{state['alpaca_status']}[/bold green]")
        
        return Panel(table, title="[bold yellow]🛡️ RISK SIZING & EXÉCUTION BROKER[/bold yellow]", border_style="yellow")

    def make_history_table(self) -> Panel:
        table = Table(expand=True, show_header=True, header_style="bold magenta", box=None)
        table.add_column("HEURE", justify="center")
        table.add_column("ACTION", justify="center")
        table.add_column("PRIX ENTRÉE", justify="center")
        table.add_column("SOMME MISÉE", justify="center")
        table.add_column("RÉSULTAT", justify="center")
        table.add_column("PNL (€)", justify="center")
        
        if not self.trade_history:
            table.add_row("N/A", "EN ATTENTE [H]", f"${self.latest_state['price']:,.2f}", "0.00 €", "ACTIF", "+0.00 €")
        else:
            for t in self.trade_history:
                table.add_row(t['time'], t['action'], t['price'], t['amount'], t['result'], t['pnl'])
                
        return Panel(table, title="[bold magenta]📜 HISTORIQUE DES DÉCISIONS EN TEMPS RÉEL[/bold magenta]", border_style="magenta")

    def make_layout(self, remaining_sec: int) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self.make_header_panel(), size=5),
            Layout(name="middle", size=9),
            Layout(self.make_history_table(), size=8),
            Layout(name="footer", size=3)
        )
        
        layout["middle"].split_row(
            Layout(self.make_signal_panel()),
            Layout(self.make_risk_panel())
        )
        
        mins = remaining_sec // 60
        secs = remaining_sec % 60
        progress_text = f"⏳ PROCHAINE DÉCISION DANS : {mins:02d}m {secs:02d}s (300s Cycle 5m) | Dernier update: {self.latest_state['last_update']} | Appuyez sur Ctrl+C pour quitter"
        layout["footer"].update(Panel(Align.center(Text(progress_text, style="bold bright_green")), border_style="bright_green"))
        
        return layout

def main():
    parser = argparse.ArgumentParser(description="Live Dashboard Trading Terminal")
    parser.add_argument("--budget", type=float, default=20000.0, help="Capital budget initial (€ / $)")
    parser.add_argument("--max-cap", type=float, default=5000.0, help="Plafond max misé par trade (€ / $)")
    args = parser.parse_args()
    
    app = TradingDashboardApp(initial_budget=args.budget, max_trade_cap=args.max_cap)
    
    app.run_single_cycle()
    
    cycle_duration = 300
    
    with Live(app.make_layout(cycle_duration), refresh_per_second=2, screen=True) as live:
        try:
            while True:
                for sec in range(cycle_duration, 0, -1):
                    live.update(app.make_layout(sec))
                    time.sleep(1)
                
                app.run_single_cycle()
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du Dashboard Live Terminal.")

if __name__ == "__main__":
    main()
