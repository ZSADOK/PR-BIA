import os
import logging
from typing import List
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LabelGenerator:
    """
    Générateur de labels (cibles) financiers avancés.
    Intègre la méthode des Triple Barrières (Profit Target vs Stop Loss vs Expiration)
    pour filtrer le bruit stochastique et booster la précision des modèles ML/IA.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def compute_triple_barrier_labels(
        df: pd.DataFrame,
        horizon: int = 3,
        vol_window: int = 20,
        tp_mult: float = 1.2,
        sl_mult: float = 1.0,
        price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Méthode des Triple Barrières (Marcos Lopez de Prado):
        - Barrière Haute (Take Profit) = P_t * (1 + tp_mult * sigma_t)
        - Barrière Basse (Stop Loss)  = P_t * (1 - sl_mult * sigma_t)
        - Barrière Temporelle (Horizon) = t + horizon
        
        Permet de filtrer le bruit des micro-variations (< 0.2%) et de ne labelliser
        que les mouvements statistiquement significatifs.
        """
        res = df.copy()
        
        # Volatilité quotidienne glissante
        past_returns = np.log(res[price_col] / res[price_col].shift(1))
        vol = past_returns.rolling(window=vol_window).std().fillna(0.01)
        
        target = np.zeros(len(res), dtype=int)
        
        close_prices = res[price_col].values
        vols = vol.values
        n = len(res)
        
        for i in range(n - horizon):
            p0 = close_prices[i]
            v = vols[i]
            
            upper_barrier = p0 * (1.0 + tp_mult * v)
            lower_barrier = p0 * (1.0 - sl_mult * v)
            
            # Prix futurs sur l'horizon [i+1, i+horizon]
            future_window = close_prices[i+1 : i+1+horizon]
            
            # Vérifier quelle barrière est touchée en premier
            hit_tp = np.where(future_window >= upper_barrier)[0]
            hit_sl = np.where(future_window <= lower_barrier)[0]
            
            first_tp = hit_tp[0] if len(hit_tp) > 0 else 999
            first_sl = hit_sl[0] if len(hit_sl) > 0 else 999
            
            if first_tp < first_sl:
                target[i] = 1 # Hausse nette significativement profitable
            elif first_sl < first_tp:
                target[i] = 0 # Baisse nette / Stop loss
            else:
                # Aucune barrière touchée (mouvement neutre/bruit) -> Exiger au moins +0.50% net de frais
                final_price = future_window[-1] if len(future_window) > 0 else p0
                target[i] = 1 if final_price >= (p0 * 1.005) else 0

        res[f"target_triple_barrier_{horizon}d"] = target
        return res

    @staticmethod
    def compute_directional_labels(
        df: pd.DataFrame, horizons: List[int] = [1, 5, 21], min_threshold: float = 0.005, price_col: str = "Close"
    ) -> pd.DataFrame:
        """
        Classification directionnelle nette de frais (min_threshold = 0.50%).
        Élimine les micro-variations neutres (< 0.50%) pour forcer l'IA à ne prédire QUE les vraies hausses rentables.
        """
        res = df.copy()
        for k in horizons:
            future_price = res[price_col].shift(-k)
            ret = (future_price - res[price_col]) / res[price_col]
            col_name = f"target_direction_{k}d"
            res[col_name] = (ret >= min_threshold).astype(int)
        return res

    def build_target_dataset(
        self,
        df: pd.DataFrame,
        ticker_name: str,
        horizons: List[int] = [1, 5, 21],
        save_parquet: bool = True,
    ) -> pd.DataFrame:
        df_processed = df.copy()
        df_processed = self.compute_directional_labels(df_processed, horizons=horizons)
        df_processed = self.compute_triple_barrier_labels(df_processed, horizon=3)
        return df_processed

if __name__ == "__main__":
    print("LabelGenerator initialisé.")
