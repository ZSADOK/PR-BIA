"""
Script de Test de Connexion aux Identifiants Alpaca Paper Trading.
"""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.execution.alpaca_executor import AlpacaExecutor

def main():
    print("=== TEST DE CONNEXION ALPACA PAPER TRADING ===")
    executor = AlpacaExecutor()
    
    account_info = executor.fetch_account()
    
    print("\n--- RÉSULTATS DU COMPTE ALPACA ---")
    for k, v in account_info.items():
        print(f" • {k}: {v}")
        
    if account_info.get("status") == "connected":
        print(f"\n✅ CONNEXION RÉUSSIE ! Solde Cash virtuel : ${account_info.get('cash'):,.2f}")
        positions = executor.fetch_positions()
        print(f" • Positions Actives sur Alpaca : {len(positions)}")
    else:
        print("\n❌ ERREUR DE CONNEXION : Vérifiez les clés dans .env")

if __name__ == "__main__":
    main()
