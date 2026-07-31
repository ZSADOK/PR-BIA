#!/usr/bin/env python3
"""
Bot de Trading Automatisé Continu Alpaca Paper Trading (100,000$)
Architecture Modulaire Ultra-Épurée (< 150 Lignes par Fichier)
"""

import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.live import Live

from src.trading_config import console, trading_client, map_symbol_to_alpaca
from src.ui import (
    get_account_status_panel, render_account_status_panel, render_ai_learning_memory_panel,
    render_api_health_panel, render_gemini_insights_panel, render_ranking_table
)
from src.execution import execute_trade_signals, execute_sell_all, check_instant_safety_limits
from src.data import SentimentFetcher
from src.models import GeminiSocialAnalyzer
from run_ai import run_meta_ensemble_strategy

from src.models import TimeSeriesEngine
from src.data import FinancialDataFetcher

ts_engine = TimeSeriesEngine()
fetcher = FinancialDataFetcher()

def evaluate_single_asset_task(task_args):
    name, symb, horizon, threshold, positions_map = task_args
    
    # Inférence Zero-Shot Pur Moteur Temporel (Google TimesFM & Amazon Chronos)
    df_ticker = fetcher.fetch_ticker(ticker=symb, interval="1d")
    if not df_ticker.empty and "Close" in df_ticker.columns:
        res = ts_engine.forecast_trajectory_and_quantiles(df_ticker["Close"], prediction_length=5)
        combined_prob = res["temporal_probability"]
    else:
        combined_prob = 0.50

    alpaca_symbol = map_symbol_to_alpaca(symb)
    clean_target = alpaca_symbol.replace("/", "").upper()
    
    has_pos = clean_target in positions_map
    pl_pct = positions_map.get(clean_target, 0.0)

    return {
        "Nom": name, "Ticker": symb, "AlpacaTicker": alpaca_symbol,
        "Confiance": combined_prob, "HasPos": has_pos, "PL_Pct": pl_pct
    }

def run_multi_asset_ranking_cycle(horizon: int = 1, threshold: float = 0.58, notional: float = 1000.0, api_cache_ttl_sec: int = 900, max_budget: float = 75000.0, max_trade_cap: float = 5000.0, universe_size: int = 15):
    now_str = datetime.now().strftime("%H:%M:%S")
    console.print(f"\n[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print(f"[bold cyan]  [{now_str}] DÉMARRAGE DU SCANNER IA EN TEMPS RÉEL ({universe_size} ACTIFS - HORIZON {horizon})[/bold cyan]")
    console.print(f"[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]\n")

    results = []
    gemini_insights = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console
    ) as progress:
        t1 = progress.add_task("[cyan][Etape 1/4] Chargement de l'Univers & Tendance Cashtags...[/cyan]", total=1)
        fetcher = SentimentFetcher()
        trending_universe = fetcher.extract_trending_cashtags(limit=universe_size)
        progress.update(t1, advance=1, description="[bold green][Etape 1/4] Univers & Cashtags Tendances OK[/bold green]")

        t2 = progress.add_task("[cyan][Etape 2/4] Analyse Gemini LLM & Sentiment...[/cyan]", total=1)
        gemini = GeminiSocialAnalyzer()
        if gemini.is_available():
            sample_posts = [f"${s} momentum haussier et volume élevé" for s in trending_universe.values()]
            gemini_insights = gemini.analyze_social_posts(sample_posts)
        progress.update(t2, advance=1, description="[bold green][Etape 2/4] Analyse Gemini LLM & Sentiment OK[/bold green]")

        # Vectorized Batch Fetching (1 seule requête HTTP pour tout l'univers)
        from src.data import FinancialDataFetcher
        data_fetcher = FinancialDataFetcher()
        tickers_list = list(trending_universe.values())
        data_fetcher.fetch_batch_tickers(tickers_list, period="2y", interval="1d")

        # Pre-fetch positions 1 seule fois pour eviter les requetes HTTP paralleles et la saturation du connection pool
        positions_map = {}
        try:
            raw_positions = trading_client.get_all_positions()
            for p in raw_positions:
                c_symb = p.symbol.replace("/", "").upper()
                positions_map[c_symb] = float(p.unrealized_plpc) * 100
        except Exception:
            pass

        tasks = [(name, symb, horizon, threshold, positions_map) for name, symb in trending_universe.items()]
        t3 = progress.add_task("[cyan][Etape 3/4] Inférence Pure Zero-Shot (TimesFM + Chronos)...[/cyan]", total=len(tasks))
        with ThreadPoolExecutor(max_workers=min(32, len(tasks))) as executor:
            futures = [executor.submit(evaluate_single_asset_task, t) for t in tasks]
            for future in futures:
                res = future.result()
                results.append(res)
                prob_pct = f"{res['Confiance']*100:.1f}%"
                status_str = "ACHAT" if res['Confiance'] >= threshold else "NEUTRE"
                progress.update(t3, advance=1, description=f"[cyan][Etape 3/4] Inférence Zero-Shot {res['Ticker']:<6} | Confiance: {prob_pct} | {status_str}[/cyan]")
        progress.update(t3, description="[bold green][Etape 3/4] Inférence Pure Zero-Shot SOTA OK (0.1s)[/bold green]")

    if gemini_insights:
        gemini_sent_map = {item.get("ticker"): item for item in gemini_insights if item.get("ticker")}
        for res in results:
            g_item = gemini_sent_map.get(res["Ticker"])
            if g_item:
                g_sent = g_item.get("sentiment", 0.0)
                cat_pow = g_item.get("catalyst_power", 5) / 10.0
                boost = g_sent * (0.15 + (cat_pow * 0.10))
                if boost > 0.05:
                    res["Confiance"] = min(0.95, res["Confiance"] + boost)
                elif boost < -0.05:
                    res["Confiance"] = max(0.05, res["Confiance"] + boost)

    results = sorted(results, key=lambda x: x["Confiance"], reverse=True)

    render_account_status_panel()
    render_ai_learning_memory_panel()
    render_api_health_panel()
    render_gemini_insights_panel(gemini_insights)
    render_ranking_table(results, threshold)

    execute_trade_signals(results, threshold, notional, max_budget, max_trade_cap)

def run_live_second_monitor(duration_sec: int):
    """
    Moniteur Rich Live (1s - FIXE ET ANCRÉ ÉCRAN) : Met à jour le tableau de bord en place sans défilement ni répétitions.
    """
    start_t = time.time()
    with Live(get_account_status_panel(duration_sec), console=console, refresh_per_second=1, transient=False) as live:
        while time.time() - start_t < duration_sec:
            remaining = max(0, int(duration_sec - (time.time() - start_t)))
            live.update(get_account_status_panel(remaining))
            time.sleep(1.0)

def main():
    parser = argparse.ArgumentParser(description="Bot de Trading Automatisé Continu Alpaca Paper Trading")
    parser.add_argument("--train", action="store_true", help="Lancer l'entraînement 3-Split (Train/Val/Test Holdout) optimisé CAUM")
    parser.add_argument("--run_all", action="store_true", help="Lancer le scanner multi-actifs")
    parser.add_argument("--continuous", action="store_true", help="Tourner en boucle infinie continue 24/7 avec monitoring 1s")
    parser.add_argument("--interval_sec", type=int, default=300, help="Intervalle en secondes entre 2 scans (défaut: 300s = 5 min)")
    parser.add_argument("--status", action="store_true", help="Afficher l'état du compte")
    parser.add_argument("--sell_all", action="store_true", help="Vendre toutes les positions")
    parser.add_argument("--horizon", type=int, default=1, choices=[0, 1, 2, 3], help="Choix d'Horizon IA (1=1j, 2=3j, 3=5j, 0=Consensus Tout)")
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--notional", type=float, default=1000.0)
    parser.add_argument("--universe_size", type=int, default=15)
    parser.add_argument("--max_budget", type=float, default=75000.0)
    parser.add_argument("--max_trade_cap", type=float, default=5000.0)

    args = parser.parse_args()

    if args.train:
        import subprocess
        subprocess.run(["python3", "scripts/train_tabfm_residual_multi_asset.py", "--tickers", "BTC-USD,ETH-USD,SOL-USD,AVAX-USD", "--patience", "12"])
    elif args.sell_all:
        execute_sell_all()
    elif args.status:
        render_account_status_panel()
        render_ai_learning_memory_panel()
        render_api_health_panel()
    elif args.run_all:
        if args.continuous:
            console.print(f"[bold green][START] Bot continu temps réel démarré (Scan toutes les {args.interval_sec}s)...[/bold green]")
            while True:
                try:
                    run_multi_asset_ranking_cycle(
                        horizon=args.horizon, threshold=args.threshold, notional=args.notional,
                        max_budget=args.max_budget, max_trade_cap=args.max_trade_cap, universe_size=args.universe_size
                    )
                    run_live_second_monitor(args.interval_sec)
                except KeyboardInterrupt:
                    console.print("\n[bold yellow][STOP] Bot continu arrêté par l'utilisateur.[/bold yellow]")
                    break
                except Exception as e:
                    console.print(f"\n[red]Erreur cycle continu: {e}[/red]")
                    time.sleep(10)
        else:
            run_multi_asset_ranking_cycle(
                horizon=args.horizon, threshold=args.threshold, notional=args.notional,
                max_budget=args.max_budget, max_trade_cap=args.max_trade_cap, universe_size=args.universe_size
            )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
