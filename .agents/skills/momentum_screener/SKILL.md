---
name: momentum_screener
description: Skill de Pré-Screening par Volume Relatif et Momentum Volatilité pour détection d'actifs à haute probabilité.
---

# Momentum Screener Skill

Ce skill définit la procédure de détection dynamique d'actifs en mouvement :

1. **Volume Relatif (RVOL)** : `RVOL = Volume_Actuel / Volume_Moyen_20J > 1.2`
2. **Tendance SMA** : `Prix > SMA50` et `Prix > SMA200`
3. **Poussée RSI** : `RSI(14) entre 50 et 72` (Momentum haussier sans surachat extrême).

Ce filtrage alimente le Meta-Ensemble ML avec des candidats pré-qualifiés à fort potentiel (> 58% de confiance).
