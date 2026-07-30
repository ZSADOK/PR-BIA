"""
Package Principal PJ-IA-Bourse (Architecture Modulaire Quantitative)
"""

from src.data.data_fetcher import FinancialDataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.data.label_generator import LabelGenerator
from src.data.sentiment_fetcher import SentimentFetcher
from src.models.model_trainer import ModelTrainer
from src.models.onnx_engine import ONNXInferenceEngine
from src.models.gemini_analyzer import GeminiSocialAnalyzer
from src.models.timeseries_engine import TimeSeriesEngine
from src.models.crypto_utility_metric import CryptoCustomUtilityMetric
from src.execution.risk_manager import RiskManager, check_instant_safety_limits, compute_dynamic_kelly_notional, calculate_expected_value
from src.execution.trade_executor import execute_trade_signals, execute_sell_all, is_us_stock_market_open
from src.execution.trade_logger import TradeLogger
from src.ui.ui_renderer import render_account_status_panel, render_ai_learning_memory_panel, render_api_health_panel, render_gemini_insights_panel, render_ranking_table
from src.backtest.backtester import Backtester
from src.trading_config import console, trading_client, map_symbol_to_alpaca, is_crypto_asset, SECTOR_MAP, TICKER_MAP
