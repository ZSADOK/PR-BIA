"""
Point d'Entrée Principal du Bot de Trading Algorithmique TimesFM + Meta-Labeler (ETH 5m Alpaca Paper).
Unifie :
- Ingestion de données 5m massives (CCXT Binance / Alpaca)
- Pré-Screening Volume Relatif (RVOL > 1.2x) & Momentum 5m
- Inférence IA TimesFM (5m H+1 forecast)
- Méta-Filtre XGBoost Triple Barrière (Win Rate > 75%)
- Dynamic Kelly Allocation & Envelope Safety (Risk Manager)
- Exécution automatique des ordres sur Alpaca Paper Trading ($100k Cash)
"""
import os
import sys

# Neutralisation OpenMP C++ macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Auto-bootstrap vers l'interpréteur .venv
base_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(base_dir, ".venv", "bin", "python3")
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(base_dir)

from config.settings import config
from src.data_loader import get_large_eth_data
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from src.models.meta_labeler import MetaLabeler
from src.risk.risk_manager import RiskManager
from src.execution.ccxt_executor import CCXTExecutor
from src.execution.alpaca_executor import AlpacaExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TimesFMBot5m")

def run_trading_cycle(capital: float = 100000.0) -> dict:
    logger.info("=== NOUVEAU CYCLE DU BOT DE TRADING ETH 5M (TIMESFM + ALPACA PAPER TRADING) ===")
    
    # 1. Ingestion des Données 5m
    logger.info(f"1. Ingestion des données historiques {config.timeframe} pour ETH/USDT...")
    df = get_large_eth_data(symbol="ETH/USDT", timeframe=config.timeframe, days_back=30, force_refresh=False)
    
    if len(df) < 50:
        logger.error("Données insuffisantes.")
        return {"status": "error", "message": "Données insuffisantes"}

    current_price = float(df['Close'].iloc[-1])
    logger.info(f"Dernier prix ETH (5m) : ${current_price:,.2f} ({len(df)} bougies 5m en mémoire)")
    
    # 2. Pré-Screening (RVOL > 1.2, SMA 50/200, RSI 50-72)
    logger.info("2. Calcul du Pré-Screening Volume & Momentum 5m...")
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    logger.info(f"Résultat Screener : RVOL={latest_screen['rvol']:.2f}, Trend_OK={latest_screen['trend_ok']}, Passed={latest_screen['passed']}")
    
    # 3. Prédiction Zero-Shot / Fine-Tuned TimesFM (5m H+1)
    logger.info("3. Inférence du modèle TimesFM (5m H+1)...")
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    logger.info(f"Prédiction TimesFM Prix H+1 (5m) : ${signal['predicted_price']:,.2f} (Variation: {signal['predicted_return_pct']:+.4f}%)")
    
    # 3.5 ÉVALUATION DU MÉTA-LABELER XGBOOST (Triple Barrière)
    logger.info("3.5 Évaluation par le Méta-Filtre XGBoost...")
    meta_labeler = MetaLabeler()
    meta_confidence = meta_labeler.predict_meta_confidence(df_screened, timesfm_pred_return=signal['predicted_return_pct']/100.0)
    
    meta_passed = meta_confidence >= config.min_meta_confidence
    logger.info(f"Score de Confiance Méta-Model : {meta_confidence*100:.2f}% (Seuil requis >= {config.min_meta_confidence*100:.0f}%) -> Passed: {meta_passed}")
    
    final_signal_binary = 1 if (signal['signal_binary'] == 1 and meta_passed) else 0
    signal['signal_binary'] = final_signal_binary
    signal['signal_label'] = "BUY (LONG - META CONFIRMED)" if final_signal_binary == 1 else "SELL / NEUTRAL"
    signal['meta_confidence'] = meta_confidence
    
    logger.info(f"Signal Binaire Final après Méta-Filtre : {final_signal_binary} -> {signal['signal_label']}")
    
    # 4. Gestion du Risque & Position Sizing (Dynamic Kelly Allocation)
    logger.info("4. Calcul du Position Sizing & Dynamic Kelly Allocation...")
    risk_mgr = RiskManager(
        default_risk_pct=config.risk_per_trade,
        max_kelly_fraction=config.max_kelly_fraction,
        max_portfolio_cap=config.max_portfolio_allocation
    )
    position_info = risk_mgr.compute_position_size(
        total_capital=capital,
        entry_price=current_price,
        df_ohlcv=df_screened,
        signal_dict=signal,
        historical_win_rate=max(0.60, meta_confidence)
    )
    
    logger.info(f"Position Sizing : Capital Alloué = ${position_info['capital_allocated']:,.2f} | Units = {position_info['quantity_units']:.4f} ETH")
    
    # 5. Exécution Automatisée sur Alpaca Paper Trading
    logger.info(f"5. Exécution automatique via {config.exchange_id.upper()} Executor...")
    if config.exchange_id == "alpaca":
        executor = AlpacaExecutor()
        acc_info = executor.fetch_account()
        cash_val = float(acc_info.get('cash', 0.0))
        account_num = acc_info.get('account_number', 'PA3T5NINSLGS')
        logger.info(f"Compte Alpaca N° {account_num} connecté. Solde Cash: ${cash_val:,.2f}")
        execution_result = executor.execute_bot_cycle(
            signal_dict=signal,
            position_size_dict=position_info,
            symbol="ETH/USD"
        )
    else:
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
    print("\n" + "="*88)
    print("=== FIN DU CYCLE DE TRADING ETH 5M ALPACA ===")
    print("="*88)
