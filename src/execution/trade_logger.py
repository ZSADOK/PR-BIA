import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Système de Mémoire & Apprentissage Continu par Rétroaction de Trade.
    Enregistre tous les trades réels/papier exécutés par l'IA et calcule les métriques de performance.
    """
    
    def __init__(self, log_filepath: str = "data/trade_history.csv"):
        self.filepath = log_filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            df_empty = pd.DataFrame(columns=[
                "timestamp", "ticker", "action", "qty", "entry_price", 
                "exit_price", "pl_usd", "pl_pct", "confidence", "reason"
            ])
            df_empty.to_csv(self.filepath, index=False)

    def log_trade(
        self,
        ticker: str,
        action: str,
        qty: float,
        entry_price: float,
        exit_price: float = 0.0,
        pl_usd: float = 0.0,
        pl_pct: float = 0.0,
        confidence: float = 0.0,
        reason: str = "SIGNAL IA"
    ):
        """
        Enregistre une opération (ACHAT ou FERMETURE) dans l'historique permanent.
        """
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker": ticker,
            "action": action,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pl_usd": pl_usd,
            "pl_pct": pl_pct,
            "confidence": confidence,
            "reason": reason
        }
        df = pd.DataFrame([row])
        df.to_csv(self.filepath, mode="a", header=False, index=False)
        logger.info(f"💾 Trade enregistré dans la Mémoire IA : {action} {ticker} (P/L: {pl_pct:+.2f}%)")

    def get_performance_summary(self) -> Dict:
        """
        Renvoie les statistiques d'apprentissage de l'IA basées sur l'historique permanent.
        """
        if not os.path.exists(self.filepath):
            return {"total_trades": 0, "win_rate": 0.0, "total_pl_usd": 0.0}
        
        try:
            df = pd.read_csv(self.filepath)
            df_exits = df[df["action"].str.startswith("VENTE")]
            if df_exits.empty:
                return {"total_trades": 0, "win_rate": 0.0, "total_pl_usd": 0.0}
            
            total_trades = len(df_exits)
            wins = len(df_exits[df_exits["pl_pct"] > 0])
            win_rate = (wins / total_trades) * 100
            total_pl_usd = df_exits["pl_usd"].sum()
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "total_pl_usd": total_pl_usd
            }
        except Exception as e:
            logger.error(f"Erreur lecture mémoire trade_history : {e}")
            return {"total_trades": 0, "win_rate": 0.0, "total_pl_usd": 0.0}
