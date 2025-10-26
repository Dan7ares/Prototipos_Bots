"""
Configuración especializada para scalping en 1M con fallback a 5M si mejora el win rate.
"""

M1_SYMBOLS = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm"]

# Preferencia inicial (se autoajusta tras el warm-up)
TIMEFRAME_DEFAULT = "M1"

# Config dedicada 1M - OPTIMIZADA para Win Rate 70%+
M1_STRATEGY_CONFIG = {
    "timeframe": "M1",
    "ema_fast": 3,        # Optimizado: más reactivo a cambios
    "ema_medium": 8,      # Optimizado: mejor detección de tendencia
    "ema_slow": 18,       # Optimizado: equilibrio velocidad/estabilidad
    "rsi_period": 13,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "atr_period": 14,
    "min_atr_threshold": 0.00010,  # M1: volatilidad mínima
    "max_spread_threshold": 0.00018,  # ≈ 1.8 pips para EURUSD
    "min_confidence": 0.70,  # Optimizado: balance entre oportunidades y calidad
    "min_confirmations": 3,
    "adx_period": 12,
    "adx_min": 22,        # Optimizado: permite más trades en tendencias moderadas
    "cooldown_bars": 2,
    "take_profit_multiplier": 1.8,  # Optimizado: mejor ratio riesgo/beneficio
    "stop_loss_multiplier": 0.9,    # Optimizado: stops más ajustados
    "min_distance_pips": 2,
    "max_distance_pips": 12,
    "max_holding_bars": 15,  # Salida rápida típica de 1M
    "use_volatility_filter": True,
    "use_trend_filter": True,
    "use_momentum_filter": True,
}

# Fallback realista 5M cuando sea superior el win rate
M5_STRATEGY_CONFIG = {
    "timeframe": "M5",
    "ema_fast": 5,
    "ema_medium": 12,
    "ema_slow": 21,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bollinger_period": 20,
    "bollinger_std": 2.0,
    "atr_period": 14,
    "min_atr_threshold": 0.00012,  # 5M exige un poco más de ATR
    "max_spread_threshold": 0.00020,
    "min_confidence": 0.65,
    "min_confirmations": 3,
    "adx_period": 14,
    "adx_min": 25,
    "cooldown_bars": 3,
    "take_profit_multiplier": 2.2,
    "stop_loss_multiplier": 1.2,
    "min_distance_pips": 4,
    "max_distance_pips": 20,
    "max_holding_bars": 24,  # Dura más que 1M
    "use_volatility_filter": True,
    "use_trend_filter": True,
    "use_momentum_filter": True,
}

# Gestión de riesgo, slippage y costes
RISK_CONFIG_M1 = {
    "initial_capital": 10000.0,
    "risk_per_trade": 0.0020,        # 0.20% por trade
    "daily_loss_limit_pct": 0.015,   # 1.5% límite diario
    "max_consecutive_losses": 3,
    "commission_pct": 0.00010,
    "slippage_pips_base": 0.0,
    "slippage_pips_scale_atr": 0.0,
    "slippage_pips_floor": 0.0
}

PERFORMANCE_MONITOR_CONFIG = {
    "rolling_window_trades": 30,
    "switch_timeframe_if_winrate_below": 0.52,
    "minimum_trades_to_switch": 15,
    "report_every_trades": 10
}