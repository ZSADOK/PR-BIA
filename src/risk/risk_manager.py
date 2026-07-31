"""
Module de Gestion du Risque et Position Sizing.
Conforme aux exigences quantitatives et aux garde-fous AGENTS.md :
- Dynamic Kelly Allocation & Envelope Safety
- Calcul du montant fixe en Euros (€ / $) basé sur le pourcentage de risque (ex: 1% ou 2%)
- Stop-Loss & Take-Profit dynamiques basés sur la volatilité (ATR - Average True Range)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(
        self,
        default_risk_pct: float = 0.02,       # 2% de risque par trade par défaut
        max_kelly_fraction: float = 0.25,    # Quarter Kelly pour limiter la volatilité
        max_portfolio_cap: float = 0.50,     # Plafond d'enveloppe de sécurité (max 50% du capital total sur un trade)
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,      # Stop Loss à 2 * ATR
        atr_tp_multiplier: float = 3.5       # Take Profit à 3.5 * ATR (R/R > 1.75)
    ):
        self.default_risk_pct = default_risk_pct
        self.max_kelly_fraction = max_kelly_fraction
        self.max_portfolio_cap = max_portfolio_cap
        self.atr_period = atr_period
        self.atr_sl_multiplier = atr_sl_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calcule l'Average True Range (ATR) sur les données OHLCV."""
        df = df.copy()
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_series = true_range.rolling(window=period).mean()
        
        latest_atr = atr_series.iloc[-1]
        if pd.isna(latest_atr) or latest_atr <= 0:
            # Fallback à 1.5% du prix si ATR indisponible
            latest_atr = df['Close'].iloc[-1] * 0.015
            
        return float(latest_atr)

    def calculate_kelly_fraction(
        self,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.5,
        confidence: float = 0.60
    ) -> float:
        """
        Calcule la fraction d'allocation optimale de Kelly :
        f* = (p * b - q) / b
        où p = win_rate, q = 1 - p, b = win_loss_ratio.
        Ajustée par la fraction de Kelly conservatrice et la confiance du modèle TimesFM.
        """
        p = max(0.01, min(0.99, win_rate))
        q = 1.0 - p
        b = max(0.1, win_loss_ratio)
        
        full_kelly = (p * b - q) / b
        
        if full_kelly <= 0:
            return 0.0
        
        # Application de la fraction de Kelly conservatrice (Quarter Kelly) et pondération par la confiance TimesFM
        fractional_kelly = full_kelly * self.max_kelly_fraction * confidence
        
        # Plafond d'enveloppe de sécurité
        safe_kelly = min(fractional_kelly, self.max_portfolio_cap)
        return float(safe_kelly)

    def compute_position_size(
        self,
        total_capital: float,
        entry_price: float,
        df_ohlcv: pd.DataFrame,
        signal_dict: Dict[str, Any],
        risk_pct_override: Optional[float] = None,
        historical_win_rate: float = 0.56,
        historical_win_loss_ratio: float = 1.6
    ) -> Dict[str, Any]:
        """
        Calcule la taille exacte de position en capital (€ / $) et en unité d'actif (ETH).
        
        Retourne un dictionnaire complet incluant :
        - capital_allocated : Montant (€/$) engagé dans le trade
        - quantity_units : Quantité d'ETH à acheter
        - stop_loss_price : Niveau de Stop-Loss (€/$)
        - take_profit_price : Niveau de Take-Profit (€/$)
        - max_risk_amount : Montant maximal risqué en cas de SL
        - envelope_safety_passed : Booléen certifiant que le budget respecte le plafond strict
        """
        risk_pct = risk_pct_override if risk_pct_override is not None else self.default_risk_pct
        confidence = signal_dict.get('confidence', 0.60)
        signal_binary = signal_dict.get('signal_binary', 0)

        # Si pas de signal d'achat (signal == 0), allocation = 0
        if signal_binary == 0 or total_capital <= 0 or entry_price <= 0:
            return {
                "capital_allocated": 0.0,
                "quantity_units": 0.0,
                "stop_loss_price": 0.0,
                "take_profit_price": 0.0,
                "max_risk_amount": 0.0,
                "risk_pct_used": 0.0,
                "kelly_fraction": 0.0,
                "envelope_safety_passed": True,
                "reason": "Signal neutre/vente (0) ou capital nul"
            }

        # 1. Calcul de l'ATR et des niveaux SL / TP
        atr = self.calculate_atr(df_ohlcv, period=self.atr_period)
        stop_loss_dist = self.atr_sl_multiplier * atr
        take_profit_dist = self.atr_tp_multiplier * atr
        
        stop_loss_price = max(0.01, entry_price - stop_loss_dist)
        take_profit_price = entry_price + take_profit_dist
        
        loss_pct_per_unit = (entry_price - stop_loss_price) / entry_price
        
        # 2. Méthode A : Fixed Risk Position Sizing (Montant risqué = Capital * risk_pct)
        target_risk_amount = total_capital * risk_pct
        fixed_risk_capital = target_risk_amount / loss_pct_per_unit
        
        # 3. Méthode B : Dynamic Kelly Allocation Position Sizing
        kelly_fraction = self.calculate_kelly_fraction(
            win_rate=historical_win_rate,
            win_loss_ratio=historical_win_loss_ratio,
            confidence=confidence
        )
        kelly_capital = total_capital * kelly_fraction
        
        # 4. Synthese et Plafond d'Enveloppe de Sécurité (Envelope Safety)
        max_allowed_allocation = total_capital * self.max_portfolio_cap
        
        # On retient le minimum entre le sizing à risque fixe et le sizing Kelly pour la sécurité maximale
        capital_allocated = min(fixed_risk_capital, kelly_capital, max_allowed_allocation)
        capital_allocated = max(0.0, capital_allocated)
        
        # Quantité d'ETH
        quantity_units = capital_allocated / entry_price
        
        # Montant réel risqué si le SL est touché
        max_risk_amount = quantity_units * (entry_price - stop_loss_price)
        actual_risk_pct = (max_risk_amount / total_capital) * 100.0 if total_capital > 0 else 0.0
        
        return {
            "total_capital": total_capital,
            "entry_price": entry_price,
            "capital_allocated": float(capital_allocated),
            "quantity_units": float(quantity_units),
            "stop_loss_price": float(stop_loss_price),
            "take_profit_price": float(take_profit_price),
            "atr": float(atr),
            "max_risk_amount": float(max_risk_amount),
            "risk_pct_used": float(actual_risk_pct),
            "kelly_fraction": float(kelly_fraction),
            "envelope_safety_passed": capital_allocated <= max_allowed_allocation,
            "risk_reward_ratio": float(take_profit_dist / stop_loss_dist)
        }
