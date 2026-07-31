"""
Module d'Optimisation des Gain & Alpha (Étape 6).
Intègre :
1. Trailing Stop-Loss Dynamique (basé sur l'ATR et le plus haut atteint).
2. Take-Profit Partiel avec Break-Even automatique (50% sorti à +1.5x ATR, 50% conservé pour la tendance).
3. Ajustement de Position selon le Régime de Volatilité (ATR Volatility Regime Scaling).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AdvancedAlphaManager:
    def __init__(
        self,
        tp1_atr_multiplier: float = 1.5,   # TP1 partiel à +1.5x ATR
        tp2_atr_multiplier: float = 3.5,   # TP2 final à +3.5x ATR
        trailing_atr_multiplier: float = 1.8 # Distance du Trailing Stop = 1.8x ATR
    ):
        self.tp1_atr_multiplier = tp1_atr_multiplier
        self.tp2_atr_multiplier = tp2_atr_multiplier
        self.trailing_atr_multiplier = trailing_atr_multiplier

    def update_position_state(
        self,
        entry_price: float,
        highest_price_seen: float,
        current_price: float,
        atr: float,
        initial_units: float,
        tp1_executed: bool = False
    ) -> Dict[str, Any]:
        """
        Met à jour l'état dynamique d'une position ouverte :
        - Calcule le Trailing Stop dynamique (qui ne peut que monter).
        - Détecte l'exécution d'un Take-Profit Partiel (TP1).
        - Ajuste le Stop-Loss au Break-Even une fois le TP1 atteint.
        """
        new_highest = max(highest_price_seen, current_price)
        
        # 1. Calcul du Trailing Stop dynamique basatif sur l'ATR
        raw_trailing_stop = new_highest - (self.trailing_atr_multiplier * atr)
        
        # Break-Even Safety : Si TP1 a déjà été exécuté, le Stop-Loss ne peut pas descendre en dessous du Prix d'Entrée
        if tp1_executed:
            effective_stop_loss = max(raw_trailing_stop, entry_price)
        else:
            initial_sl = entry_price - (2.0 * atr)
            effective_stop_loss = max(raw_trailing_stop, initial_sl)

        # 2. Détection du TP1 partiel (+1.5x ATR)
        tp1_target = entry_price + (self.tp1_atr_multiplier * atr)
        trigger_tp1 = (current_price >= tp1_target) and not tp1_executed
        
        # 3. Détection du TP2 final (+3.5x ATR)
        tp2_target = entry_price + (self.tp2_atr_multiplier * atr)
        trigger_tp2 = current_price >= tp2_target
        
        # 4. Détection du Stop Loss / Trailing Stop touché
        trigger_stop = current_price <= effective_stop_loss

        action = "HOLD"
        units_to_close = 0.0
        
        if trigger_stop:
            action = "CLOSE_FULL_STOP"
            units_to_close = initial_units if not tp1_executed else initial_units * 0.5
        elif trigger_tp2:
            action = "CLOSE_FULL_TP2"
            units_to_close = initial_units * 0.5 if tp1_executed else initial_units
        elif trigger_tp1:
            action = "PARTIAL_TP1_EXECUTE"
            units_to_close = initial_units * 0.5  # Vente de 50% de la position

        return {
            "current_price": current_price,
            "entry_price": entry_price,
            "highest_price_seen": new_highest,
            "effective_stop_loss": float(effective_stop_loss),
            "tp1_target": float(tp1_target),
            "tp2_target": float(tp2_target),
            "tp1_executed": tp1_executed or trigger_tp1,
            "action": action,
            "units_to_close": float(units_to_close)
        }

    def compute_volatility_scaling_factor(self, df_ohlcv: pd.DataFrame, atr: float) -> float:
        """
        Ajuste la taille de position en fonction du régime de volatilité relative (ATR / Price).
        Si la volatilité est extrême (>3%), on réduit le sizing de 30% à 50% pour prévenir les drawdowns.
        Si la volatilité est modérée (1.0% à 2.0%), on conserve 100% de l'allocation Kelly.
        """
        current_price = float(df_ohlcv['Close'].iloc[-1])
        volatility_ratio = atr / current_price  # % d'ATR par rapport au prix
        
        if volatility_ratio > 0.035: # Volatilité extrême (>3.5%)
            scaling_factor = 0.50
        elif volatility_ratio > 0.025: # Volatilité élevée (2.5% - 3.5%)
            scaling_factor = 0.75
        elif volatility_ratio < 0.008: # Volatilité anémique (<0.8%)
            scaling_factor = 0.80
        else: # Régime de volatilité optimale (1% - 2.5%)
            scaling_factor = 1.00
            
        return float(scaling_factor)
