#!/usr/bin/env python3
"""
Script Principal d'IA Financière (Bourse & Crypto)
=================================================
Moteur d'IA : Méta-Ensemble Consensus (XGBoost + LightGBM + Random Forest + TabFM / TabPFN).

Ce script exécute l'ensemble du pipeline IA avancé :
1. Téléchargement des prix de marché (yfinance)
2. Extraction des sentiments (Reddit, X/Twitter, News, Crypto Fear & Greed Index)
3. Calcul des indicateurs techniques stationnaires (RSI, MACD, Bollinger, ATR, Volatilité, Z-score Volume)
4. Entraînement multi-modèles & Méta-Ensemble Consensus
5. Filtrage par Seuil de Confiance (Seuil > 54% ou 58%) et Backtest avec gestion des risques
"""

import os
import argparse
import logging
import threading
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

PLOT_LOCK = threading.Lock()
matplotlib.use("Agg")

from src.data import FinancialDataFetcher, LabelGenerator, SentimentFetcher, FeatureEngineer
from src.models import ModelTrainer
from src.backtest import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORTFOLIO_TICKERS = {
    "Bitcoin USD": "BTC-USD",
    "Ethereum USD": "ETH-USD",
    "Solana USD": "SOL-USD",
    "Avalanche USD": "AVAX-USD",
    "Dogecoin USD": "DOGE-USD",
    "Chainlink USD": "LINK-USD",
    "Nvidia Corp": "NVDA",
    "Apple Inc": "AAPL",
    "Microsoft Corp": "MSFT",
    "Amazon Inc": "AMZN",
    "Tesla Inc": "TSLA",
    "S&P 500 Index": "^GSPC"
}

def run_meta_ensemble_strategy(
    ticker: str,
    start_date: str = "2021-01-01",
    interval: str = "1d",
    horizon: int = 1,
    prob_threshold: float = 0.54,
    initial_capital: float = 10000.0,
    verbose: bool = False
) -> pd.DataFrame:
    if verbose:
        print("\n" + "=" * 80)
        print(f" 🚀 PIPELINE IA : MÉTA-ENSEMBLE CONSENSUS (XGBoost + TimesFM + Chronos + LGBM + RF) pour {ticker}")
        print(f" (Fréquence: {interval}, Horizon: {horizon}d, Seuil de Confiance: {prob_threshold*100:.0f}%)")
        print("=" * 80)

    # 1. Données de marché
    if verbose:
        print(f"\n[1/5] 📈 Récupération des prix OHLCV pour {ticker} depuis {start_date}...")
    fetcher = FinancialDataFetcher(cache_dir="data/raw")
    df_raw = fetcher.fetch_ticker(ticker=ticker, start_date=start_date, interval=interval)
    
    if df_raw.empty:
        logger.error(f"Aucune donnée pour {ticker}")
        return pd.DataFrame()

    # 2. Sentiments
    if verbose:
        print(f"\n[2/5] 🌐 Extraction du sentiment multi-sources (Reddit, X/Twitter, News & Fear & Greed)...")
    sent_fetcher = SentimentFetcher()
    df_sentiment = sent_fetcher.get_aggregated_sentiment(ticker=ticker, dates_index=df_raw.index, freq=interval)

    # 3. Features & Labels
    if verbose:
        print(f"\n[3/5] ⚙️ Calcul des indicateurs techniques stationnaires & Fusion...")
    engineer = FeatureEngineer()
    df_tech = engineer.build_technical_features(df_raw)
    
    label_gen = LabelGenerator()
    df_labeled = label_gen.build_target_dataset(df_tech, ticker_name=ticker, horizons=[horizon])
    target_col = f"target_triple_barrier_{horizon}d" if f"target_triple_barrier_{horizon}d" in df_labeled.columns else f"target_direction_{horizon}d"

    df_merged, feature_cols = engineer.merge_market_and_sentiment(df_labeled, df_sentiment)

    # 4. Modélisation Multi-Modèles & Méta-Ensemble
    if verbose:
        print(f"\n[4/5] 🤖 Entraînement des modèles (XGBoost, TimesFM + Chronos, LightGBM, Random Forest)...")
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, df_train, df_test = trainer.prepare_train_test_split(
        df=df_merged, feature_cols=feature_cols, target_col=target_col, train_ratio=0.8
    )

    xgb_model, xgb_metrics = trainer.train_xgboost(X_train, y_train, X_test, y_test)
    rf_model, rf_metrics = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    lgb_model, lgb_metrics = trainer.train_lightgbm(X_train, y_train, X_test, y_test)
    close_col = df_merged["Close"] if "Close" in df_merged.columns else None
    ts_metrics = trainer.eval_timeseries_engine(X_train, y_train, X_test, y_test, close_series=close_col)

    # Méta-Ensemble Consensus
    ensemble_metrics = trainer.build_meta_ensemble([xgb_metrics, lgb_metrics, rf_metrics, ts_metrics], y_test)

    # 5. Backtest comparatif
    if verbose:
        print(f"\n[5/5] 📊 Simulation de Trading & Backtest (Seuil de Confiance > {prob_threshold*100:.0f}%)...")
    backtester = Backtester(initial_capital=initial_capital)
    
    res_ensemble = backtester.run_backtest(
        df_test, ensemble_metrics["preds"], ensemble_metrics["probs"],
        model_name="🔥 Méta-Ensemble Consensus", prob_threshold=prob_threshold
    )
    
    res_xgb = backtester.run_backtest(
        df_test, xgb_metrics["preds"], xgb_metrics["probs"],
        model_name="XGBoost", prob_threshold=prob_threshold
    )
    
    res_ts = backtester.run_backtest(
        df_test, ts_metrics["preds"], ts_metrics["probs"],
        model_name="TimeSeries (TimesFM + Chronos)", prob_threshold=prob_threshold
    )

    results_dict = {
        "🔥 Méta-Ensemble Consensus": res_ensemble,
        "XGBoost": res_xgb,
        "TimeSeries (TimesFM + Chronos)": res_ts
    }
    
    clean_ticker_name = ticker.replace('^', 'INDEX_').replace('=', '_').replace('-', '_')
    chart_file = f"data/backtest_{clean_ticker_name}.png"
    try:
        with PLOT_LOCK:
            backtester.plot_backtest_results(results_dict, save_path=chart_file)
    except Exception:
        pass

    # Tableau de synthèse
    summary_data = [
        {
            "Actif": ticker,
            "Modèle": "Buy & Hold (Marché)",
            "Accuracy": "-",
            "ROC-AUC": "-",
            "Rendement (%)": f"{res_ensemble['return_benchmark_pct']:.2f}%",
            "Sharpe Ratio": "-",
            "Max Drawdown": "-"
        },
        {
            "Actif": ticker,
            "Modèle": "🔥 Méta-Ensemble Consensus",
            "Accuracy": f"{ensemble_metrics['accuracy']*100:.2f}%",
            "ROC-AUC": f"{ensemble_metrics['roc_auc']:.3f}",
            "Rendement (%)": f"{res_ensemble['return_strategy_pct']:.2f}%",
            "Sharpe Ratio": f"{res_ensemble['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{res_ensemble['max_drawdown_pct']:.2f}%"
        },
        {
            "Actif": ticker,
            "Modèle": "XGBoost",
            "Accuracy": f"{xgb_metrics['accuracy']*100:.2f}%",
            "ROC-AUC": f"{xgb_metrics['roc_auc']:.3f}",
            "Rendement (%)": f"{res_xgb['return_strategy_pct']:.2f}%",
            "Sharpe Ratio": f"{res_xgb['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{res_xgb['max_drawdown_pct']:.2f}%"
        },
        {
            "Actif": ticker,
            "Modèle": "TimeSeries (TimesFM + Chronos)",
            "Accuracy": f"{ts_metrics['accuracy']*100:.2f}%",
            "ROC-AUC": f"{ts_metrics['roc_auc']:.3f}",
            "Rendement (%)": f"{res_ts['return_strategy_pct']:.2f}%",
            "Sharpe Ratio": f"{res_ts['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{res_ts['max_drawdown_pct']:.2f}%"
        }
    ]
    
    summary_df = pd.DataFrame(summary_data)
    if verbose:
        print("\n" + "=" * 80)
        print(f" 🏆 SYNTHÈSE DES PERFORMANCES : {ticker}")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
        print(f"🖼️ Graphique du Backtest sauvegardé dans : {chart_file}\n")
    
    latest_prob = float(ensemble_metrics["probs"][-1])
    return summary_df, latest_prob

def main():
    parser = argparse.ArgumentParser(description="IA Financière Méta-Ensemble (XGBoost + TabFM + LGBM + RF)")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Ticker actif (ex: BTC-USD, NVDA, AAPL)")
    parser.add_argument("--start_date", type=str, default="2021-01-01", help="Date de début YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="1d", help="Intervalle (1d ou 1h)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon de prédiction en jours (1d, 3d, 5d)")
    parser.add_argument("--threshold", type=float, default=0.54, help="Seuil de probabilité d'achat (ex: 0.54 pour 54%)")
    parser.add_argument("--run_all", action="store_true", help="Exécuter sur l'ensemble du portefeuille multi-actifs")

    args = parser.parse_args()

    if args.run_all:
        print("\n" + "🚀" * 40)
        print(" EXECUTION DU MÉTA-ENSEMBLE SUR LE PORTEFEUILLE MULTI-ACTIFS")
        print("🚀" * 40)
        all_summaries = []
        for name, symb in PORTFOLIO_TICKERS.items():
            df_res, _ = run_meta_ensemble_strategy(
                ticker=symb, start_date=args.start_date, interval=args.interval,
                horizon=args.horizon, prob_threshold=args.threshold
            )
            if not df_res.empty:
                all_summaries.append(df_res)
                
        if all_summaries:
            global_df = pd.concat(all_summaries, ignore_index=True)
            print("\n" + "=" * 80)
            print(" 🏆 SYNTHÈSE GLOBALE PORTEFEUILLE - MÉTA-ENSEMBLE CONSENSUS")
            print("=" * 80)
            print(global_df.to_string(index=False))
            print("=" * 80)
    else:
        run_meta_ensemble_strategy(
            ticker=args.ticker, start_date=args.start_date, interval=args.interval,
            horizon=args.horizon, prob_threshold=args.threshold
        )

if __name__ == "__main__":
    main()
