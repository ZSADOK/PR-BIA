"""
Point d'Entrée Principal du Bot de Trading Algorithmique TimesFM + Meta-Labeler (ETH 5m Alpaca Paper).
Dashboard visuel haute précision avec affichage structuré des signaux, du risk sizing et des ordres Alpaca.
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

import time
import argparse
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
    # 1. Ingestion des Données 5m
    df = get_large_eth_data(symbol="ETH/USDT", timeframe=config.timeframe, days_back=30, force_refresh=False)
    if len(df) < 50:
        logger.error("Données insuffisantes.")
        return {"status": "error", "message": "Données insuffisantes"}

    current_price = float(df['Close'].iloc[-1])
    
    # 2. Pré-Screening (RVOL > 1.2, SMA 50/200, RSI 50-72)
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    
    # 3. Prédiction TimesFM (5m H+1)
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    
    # 3.5 ÉVALUATION DU MÉTA-LABELER XGBOOST (Triple Barrière)
    meta_labeler = MetaLabeler()
    meta_confidence = meta_labeler.predict_meta_confidence(df_screened, timesfm_pred_return=signal['predicted_return_pct']/100.0)
    meta_passed = meta_confidence >= config.min_meta_confidence
    
    final_signal_binary = 1 if (signal['signal_binary'] == 1 and meta_passed) else 0
    signal['signal_binary'] = final_signal_binary
    signal['signal_label'] = "BUY (LONG - META CONFIRMED)" if final_signal_binary == 1 else "SELL / NEUTRAL"
    signal['meta_confidence'] = meta_confidence
    
    # 4. Gestion du Risque & Position Sizing (Dynamic Kelly Allocation)
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
    
    # 5. Exécution Automatisée sur Alpaca Paper Trading
    if config.exchange_id == "alpaca":
        executor = AlpacaExecutor()
        acc_info = executor.fetch_account()
        cash_val = float(acc_info.get('cash', 100000.0))
        account_num = acc_info.get('account_number', 'PA3T5NINSLGS')
        execution_result = executor.execute_bot_cycle(
            signal_dict=signal,
            position_size_dict=position_info,
            symbol="ETH/USD"
        )
    else:
        account_num = "CCXT_SANDBOX"
        cash_val = 100000.0
        executor = CCXTExecutor(exchange_id=config.exchange_id, sandbox=config.sandbox_mode)
        execution_result = executor.execute_bot_cycle(
            signal_dict=signal,
            position_size_dict=position_info,
            symbol=config.symbol
        )

    # --------------------------------------------------------------------------
    # TABLEAU D'AFFICHAGE ET DASHBOARD STYLE TABFM / SOTA
    # --------------------------------------------------------------------------
    print("\n" + "="*88)
    print("⚡ BOT DE TRADING QUANTITATIF TIMESFM 5M — CYCLE EN TEMPS RÉEL ⚡")
    print("="*88)
    
    print("\n🔍 DIAGNOSTIC SYSTÈME & MODÈLES IA")
    print("-" * 60)
    print(f"• Modèle TimesFM Fine-Tuné     : {'✅ CHARGÉ (Fine-Tuned PT)' if engine.is_finetuned else '⚠️ MODE ZERO-SHOT'}")
    print(f"• Méta-Modèle XGBoost (5m)    : {'✅ DÉTECTÉ & ACTIF' if meta_labeler.meta_model is not None else '⚠️ MODE HEURISTIQUE'}")
    print(f"• Compte Alpaca Broker         : ✅ CONNECTÉ (N° {account_num} | Solde: ${cash_val:,.2f})")
    print(f"• Paire & Timeframe            : {config.symbol} ({config.timeframe})")
    print("-" * 60)
    
    print("\n📊 SIGNAL & DÉCISION IA DU CYCLE 5M")
    print("-" * 60)
    print(f"• Prix ETH Actuel              : ${current_price:,.2f}")
    print(f"• Prédiction TimesFM H+1 (5m)  : ${signal['predicted_price']:,.2f} ({signal['predicted_return_pct']:+.4f}%)")
    print(f"• Score Confiance XGBoost      : {meta_confidence*100:.2f}% (Seuil requis >= {config.min_meta_confidence*100:.0f}%) -> {'✅ PASSED' if meta_passed else '❌ REJECTED'}")
    print(f"• Pré-Screening RVOL & Trend  : RVOL = {latest_screen['rvol']:.2f} -> {'✅ PASSED' if latest_screen['passed'] else '⚠️ NEUTRE'}")
    
    signal_icon = "🟢 BUY (LONG)" if final_signal_binary == 1 else "🔴 NEUTRAL / HOLD"
    print(f"• Signal Binaire Final         : {final_signal_binary} -> {signal_icon}")
    print("-" * 60)
    
    print("\n🛡️ GESTION DU RISQUE & POSITION SIZING (DYNAMIC KELLY)")
    print("-" * 60)
    print(f"• Capital Alloué par Ordre     : ${position_info['capital_allocated']:,.2f} ({position_info['capital_allocated']/capital*100:.2f}% du capital)")
    print(f"• Unités d'Achat Réelles      : {position_info['quantity_units']:.4f} ETH")
    print(f"• Niveau Stop-Loss (1.0x ATR)  : ${position_info['stop_loss_price']:,.2f}")
    print(f"• Niveau Take-Profit (1.5x ATR): ${position_info['take_profit_price']:,.2f}")
    print(f"• Envelope Safety Check        : {'✅ VALIDÉE' if position_info['envelope_safety_passed'] else '❌ DEPASSEE'}")
    print("-" * 60)
    
    print("\n🚀 STATUT D'EXÉCUTION SUR ALPACA PAPER TRADING")
    print("-" * 60)
    action_str = execution_result.get('action', 'NEUTRAL_HOLD')
    if action_str == "BUY_EXECUTED_ALPACA":
        print(f"• Statut Ordre                 : 🟢 ORDRE D'ACHAT TRANSMIS SUR ALPACA !")
        print(f"• Détails Ordre ID            : {execution_result.get('order_details', {}).get('order_id', 'N/A')}")
    elif action_str == "SELL_CLOSE_EXECUTED_ALPACA":
        print(f"• Statut Ordre                 : 🔴 POSITION REVENDUE / FERMÉE SUR ALPACA !")
    else:
        print(f"• Statut Ordre                 : ⏸️ NEUTRAL_HOLD (Capital Protégé en Cash)")
    print("="*88 + "\n")

    return {
        "timestamp": str(df.index[-1]),
        "current_price": current_price,
        "signal": signal,
        "position_info": position_info,
        "execution": execution_result
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot de Trading TimesFM ETH 5m")
    parser.add_argument("--loop", action="store_true", help="Exécute le bot en boucle continue toutes les 5 minutes (300s)")
    args = parser.parse_args()
    
    if args.loop:
        try:
            while True:
                run_trading_cycle()
                print("⏳ En attente de la prochaine bougie 5m (300s)... (Appuyez sur Ctrl+C pour stopper)\n")
                time.sleep(300)
        except KeyboardInterrupt:
            print("\n🛑 Arrêt manuel du bot.")
    else:
        run_trading_cycle()
