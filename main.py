"""
Point d'Entrée Principal du Bot de Trading Algorithmique TimesFM (ETH 1h).
Unifie :
- Étape 2 : Ingestion de données massives (CCXT), Pré-screening Volume/Momentum & Prédiction TimesFM H+1
- Étape 3 : Gestion du Risque, Dynamic Kelly Allocation & Envelope Safety
- Étape 4 : Exécution automatique des ordres via CCXT (Sandbox / Live)
"""
import os
import sys
import logging
import pandas as pd

from config.settings import config
from src.data_loader import get_large_eth_data
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from src.risk.risk_manager import RiskManager
from src.execution.ccxt_executor import CCXTExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TimesFMBot")

def run_trading_cycle(capital: float = 10000.0) -> dict:
    logger.info("=== NOUVEAU CYCLE DU BOT DE TRADING ETH (TIMESFM) ===")
    
    # 1. Ingestion des Données Historiques CCXT Binance (ETH/USDT 1h)
    logger.info(f"1. Ingestion des données historiques 1h pour {config.symbol}...")
    df = get_large_eth_data(symbol=config.symbol, timeframe=config.timeframe, days_back=90, force_refresh=False)
    
    if len(df) < 50:
        logger.error("Données insuffisantes.")
        return {"status": "error", "message": "Données insuffisantes"}

    current_price = float(df['Close'].iloc[-1])
    logger.info(f"Dernier prix ETH (1h) : {current_price:.2f} $ ({len(df)} bougies 1h en mémoire)")
    
    # 2. Pré-Screening (RVOL > 1.2, SMA 50/200, RSI 50-72)
    logger.info("2. Calcul du Pré-Screening Volume & Momentum...")
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    logger.info(f"Résultat Screener : RVOL={latest_screen['rvol']:.2f}, Trend_OK={latest_screen['trend_ok']}, Passed={latest_screen['passed']}")
    
    # 3. Prédiction Zero-Shot TimesFM & Signal Binaire
    logger.info("3. Inférence du modèle Zero-Shot TimesFM (H+1)...")
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    logger.info(f"Prédiction TimesFM Prix H+1 : {signal['predicted_price']:.2f} $ (Variation: {signal['predicted_return_pct']:+.4f}%)")
    logger.info(f"Signal Binaire Généré : {signal['signal_binary']} -> {signal['signal_label']}")
    
    # 4. Gestion du Risque & Position Sizing (Dynamic Kelly Allocation)
    logger.info("4. Calcul du Position Sizing & Protection du Capital...")
    risk_mgr = RiskManager(
        default_risk_pct=config.risk_per_trade,
        max_kelly_fraction=config.max_kelly_fraction,
        max_portfolio_cap=config.max_portfolio_allocation
    )
    position_info = risk_mgr.compute_position_size(
        total_capital=capital,
        entry_price=current_price,
        df_ohlcv=df_screened,
        signal_dict=signal
    )
    
    logger.info(f"Position Sizing : Capital Alloué = {position_info['capital_allocated']:.2f} € | Units = {position_info['quantity_units']:.4f} ETH")
    if position_info['capital_allocated'] > 0:
        logger.info(f"Stop-Loss = {position_info['stop_loss_price']:.2f} $ | Take-Profit = {position_info['take_profit_price']:.2f} $")
    
    # 5. Exécution Automatisée CCXT
    logger.info("5. Exécution automatique via CCXT Executor...")
    executor = CCXTExecutor(exchange_id=config.exchange_id, sandbox=config.sandbox_mode)
    execution_result = executor.execute_bot_cycle(
        signal_dict=signal,
        position_size_dict=position_info,
        symbol=config.symbol
    )
    
    logger.info(f"Résultat d'exécution : {execution_result['action']}")
    return {
        "timestamp": str(df.index[-1]),
        "current_price": current_price,
        "signal": signal,
        "position_info": position_info,
        "execution": execution_result
    }

if __name__ == "__main__":
    res = run_trading_cycle()
    print("\n=== FIN DU CYCLE ===")
    print(res)
