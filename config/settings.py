# CONFIGURACION DE SCALPING OPTIMIZADA - ALTA RENTABILIDAD
SCALPING_CONFIG = {
    "symbol": "EURUSD",
    "timeframe": "M1",  # Scalping en 1 minuto
    "volume": 0.01,
    "max_daily_trades": 20,   # Reducido para optimización
    "max_simultaneous_trades": 2,  # Reducido para mejor control de riesgo
    "risk_per_trade": 0.004,  # 0.4% por operación (reducido)
    "max_daily_loss": 0.015,  # 1.5% máximo diario (más conservador)
    
    # Horarios de trading ajustados a tu horario (GMT-5 aproximado)
    "trading_hours": {
        "start": 8,   # 08:00 (optimizado)
        "end": 18     # 18:00 (optimizado)
    },
    
    # Condiciones de mercado adaptadas con configuración dinámica
    "max_spread": 1.5,  # Spread máximo base para mayor precisión
    "max_spread_dynamic": 3.0,  # Spread máximo durante alta volatilidad
    "min_volatility": 0.00008,  # Ajustado a la volatilidad actual del mercado
    "min_volatility_adaptive": 0.00005,  # Umbral reducido para mercados lentos
    
    # Configuración dinámica de condiciones de mercado
    "market_conditions": {
        "high_volatility_threshold": 0.0003,
        "low_volatility_threshold": 0.00005,
        "spread_tolerance_multiplier": 2.0
    },
    
    "max_consecutive_holds": 3,  # Reducido para optimización
    "cycle_interval": 120  # Aumentado para menos frecuencia
}

# Estrategia de Scalping ALTA RENTABILIDAD - AJUSTE ITERATIVO 1
STRATEGY_CONFIG = {
    # EMAs optimizadas para mejor rentabilidad - AJUSTE 1: Más sensibles
    "ema_fast": 5,           # Incrementado para mejor captura de tendencias
    "ema_medium": 12,        # Ajustado para mejor balance
    "ema_slow": 21,          # Mantenido para estabilidad

    # RSI más balanceado - AJUSTE 1: Menos restrictivo
    "rsi_period": 14,        # Período estándar más confiable
    "rsi_oversold": 30,      # Menos restrictivo para más señales
    "rsi_overbought": 70,    # Menos restrictivo para más señales

    # Bollinger optimizado - AJUSTE 1: Más sensible
    "bollinger_period": 20,  # Período estándar
    "bollinger_std": 2.0,    # Desviación estándar para mejor balance

    # ATR y gestión de riesgo mejorada - AJUSTE 1: Más conservador
    "atr_period": 14,        # ATR estándar más confiable
    "take_profit_multiplier": 2.5,  # TP más realista
    "stop_loss_multiplier": 1.2,    # SL más conservador
    "min_distance_pips": 5,         # Distancia mínima aumentada
    "max_distance_pips": 20,        # Distancia máxima reducida

    # Filtros de calidad - AJUSTE 1: Menos restrictivo
    "min_confidence": 0.65,         # Confianza mínima reducida para más señales

    # Indicadores adicionales - AJUSTE 1: Más balanceados
    "min_confirmations": 3,         # Confirmaciones reducidas
    "support_resistance_lookback": 20,  # Lookback aumentado para mejor precisión
    "fib_lookback": 50,             # Fibonacci más estable
    "mfi_period": 14,               # MFI estándar
    "vwap_enabled": True,

    # Tolerancias y filtros - AJUSTE 1: Más permisivos
    "sr_tolerance_atr_multiplier": 0.8,  # Tolerancia SR aumentada
    "adx_period": 14,               # ADX estándar
    "adx_min": 20,                  # ADX mínimo reducido
    "cooldown_bars": 3,             # Cooldown aumentado para mejor calidad
    "mfi_buy_min": 45.0,            # MFI buy menos agresivo
    "mfi_sell_max": 55.0,           # MFI sell menos agresivo
    "vwap_atr_multiplier": 0.5      # VWAP balanceado
}

# Capital inicial para backtesting y gestión de riesgo
INITIAL_CAPITAL = 10000.0  # $10,000 USD

# Configuración de múltiples timeframes para análisis avanzado
MULTI_TIMEFRAME_CONFIG = {
    "primary_timeframes": ["M1", "M5"],
    "analysis_timeframes": ["M1", "M5", "M15"],  # Análisis más completo
    "timeframe_weights": {
        "M1": 0.7,   # Mayor peso para M1 (más oportunidades de scalping)
        "M5": 0.9,   # Peso muy alto para M5 (mejor balance señal/ruido)
        "M15": 0.3   # Menor peso para M15 (confirmación de tendencia)
    },
    "min_timeframe_score": 0.25  # Umbral reducido para más oportunidades
}

# Configuración de múltiples mercados
MARKET_CONFIG = {
    "symbols": ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm"],
    "market_evaluation_period": 150,  # Período reducido para mayor reactividad
    "min_market_score": 0.10,         # Score mínimo reducido de 0.15 a 0.10
    "volatility_threshold": 0.15,     # Umbral de volatilidad reducido de 0.2 a 0.15
    "trend_strength_threshold": 0.10  # Umbral de tendencia reducido de 0.15 a 0.10
}

# Objetivos de rentabilidad más realistas
PROFITABILITY_TARGETS = {
    "initial_target": 0.15,  # 15% rentabilidad inicial más realista
    "final_target": 0.25,    # 25% rentabilidad final
    "min_win_rate": 0.55,    # 55% win rate mínimo
    "max_drawdown": 0.10,    # 10% drawdown máximo
    "min_profit_factor": 1.5, # Factor de ganancia mínimo
    "min_trades": 20         # Mínimo de trades para validez
}

# Configuración balanceada de estrategia para rentabilidad realista
STRATEGY_CONFIG_OPTIMIZED = {
    # RSI balanceado para mejor precisión
    "rsi_period": 14,        # Período estándar más confiable
    "rsi_oversold": 25,      # Menos agresivo, más preciso
    "rsi_overbought": 75,    # Menos agresivo, más preciso
    
    # EMA equilibrada para capturar movimientos
    "ema_fast": 8,           # Rápida pero no ultra sensible
    "ema_slow": 21,          # Más estable para tendencias
    
    # Gestión de riesgo realista
    "stop_loss_multiplier": 1.5,    # SL más conservador
    "take_profit_multiplier": 2.5,  # TP realista (R:R 1:1.67)
    
    # Filtros de calidad balanceados
    "min_atr_threshold": 0.00010,   # ATR mínimo para volatilidad adecuada
    "max_spread_threshold": 0.00020, # Spread máximo aceptable
    "min_confidence": 0.65,         # Confianza mínima balanceada
    
    # Parámetros de volumen y momentum estables
    "volume_sma_period": 20,        # Período estándar más estable
    "momentum_period": 10,          # Momentum balanceado
    "bollinger_period": 20,         # Período estándar
    "bollinger_std": 2.0,           # Desviación estándar
    
    # Indicadores con parámetros balanceados
    "adx_period": 14,
    "adx_threshold": 25,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "stoch_k": 14,
    "stoch_d": 3,
    "williams_r_period": 14,
    
    # Configuración de sesiones de trading optimizada
    "trading_sessions": {
        "london": {"start": "08:00", "end": "17:00"},
        "new_york": {"start": "13:00", "end": "22:00"},
        "overlap": {"start": "13:00", "end": "17:00"}  # Sesión de mayor liquidez
    },
    
    # Filtros adicionales para calidad
    "use_session_filter": True,
    "use_volatility_filter": True,
    "use_trend_filter": True,
    "use_momentum_filter": True
}

# Estrategia de Scalping ALTA RENTABILIDAD - AJUSTE ITERATIVO 2
STRATEGY_CONFIG = {
    # EMAs optimizadas para mejor rentabilidad - AJUSTE 2: Más agresivas
    "ema_fast": 8,           # Incrementado para mejor balance sensibilidad/estabilidad
    "ema_medium": 21,        # Ajustado para mejor confirmación
    "ema_slow": 34,          # Incrementado para mejor filtrado de tendencia

    # RSI más balanceado - AJUSTE 2: Más permisivo
    "rsi_period": 14,        # Período estándar más confiable
    "rsi_oversold": 35,      # Más permisivo para más señales de compra
    "rsi_overbought": 65,    # Más permisivo para más señales de venta

    # Bollinger optimizado - AJUSTE 2: Más sensible
    "bollinger_period": 20,  # Período estándar
    "bollinger_std": 1.8,    # Desviación reducida para más señales

    # ATR y gestión de riesgo mejorada - AJUSTE 2: Más agresivo
    "atr_period": 14,        # ATR estándar más confiable
    "take_profit_multiplier": 2.5,  # TP más realista
    "stop_loss_multiplier": 1.2,    # SL más conservador
    "min_distance_pips": 5,         # Distancia mínima aumentada
    "max_distance_pips": 20,        # Distancia máxima reducida

    # Filtros de calidad - AJUSTE 1: Menos restrictivo
    "min_confidence": 0.65,     # Elevar confianza mínima en horas pico
    "min_confirmations": 3,     # Confirmaciones mínimas para evitar laterales
    "adx_min": 25,              # ADX más estricto para tendencia clara
    "cooldown_bars": 3,         # Reducir overtrading en picos de volatilidad
    "mfi_buy_min": 45.0,            # MFI buy menos agresivo
    "mfi_sell_max": 55.0,           # MFI sell menos agresivo
    "vwap_atr_multiplier": 0.5      # VWAP balanceado
}

# Capital inicial para backtesting y gestión de riesgo
INITIAL_CAPITAL = 10000.0  # $10,000 USD

# Configuración de múltiples timeframes para análisis avanzado
MULTI_TIMEFRAME_CONFIG = {
    "primary_timeframes": ["M1", "M5"],
    "analysis_timeframes": ["M1", "M5", "M15"],  # Análisis más completo
    "timeframe_weights": {
        "M1": 0.7,   # Mayor peso para M1 (más oportunidades de scalping)
        "M5": 0.9,   # Peso muy alto para M5 (mejor balance señal/ruido)
        "M15": 0.3   # Menor peso para M15 (confirmación de tendencia)
    },
    "min_timeframe_score": 0.25  # Umbral reducido para más oportunidades
}

# Configuración de múltiples mercados
MARKET_CONFIG = {
    "symbols": ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm"],
    "market_evaluation_period": 200,  # Período aumentado para mejor evaluación
    "min_market_score": 0.10,         # Score mínimo reducido de 0.25 a 0.10
    "volatility_threshold": 0.15,     # Umbral de volatilidad reducido de 0.3 a 0.15
    "trend_strength_threshold": 0.10  # Umbral de tendencia reducido de 0.2 a 0.10
}

# Objetivos de rentabilidad ajustados para 70% de margen
PROFITABILITY_TARGETS = {
    "initial_target": 0.70,  # 70% objetivo principal
    "final_target": 0.80,    # 80% objetivo extendido
    "min_win_rate": 0.60,    # 60% win rate mínimo
    "max_drawdown": 0.12,    # 12% drawdown máximo
    "min_profit_factor": 1.8, # Factor de ganancia mínimo
    "min_trades": 20         # Mínimo de trades para validez
}

# Configuración balanceada de estrategia para rentabilidad realista
STRATEGY_CONFIG_OPTIMIZED = {
    # RSI balanceado para mejor precisión
    "rsi_period": 14,        # Período estándar más confiable
    "rsi_oversold": 25,      # Menos agresivo, más preciso
    "rsi_overbought": 75,    # Menos agresivo, más preciso
    
    # EMA equilibrada para capturar movimientos
    "ema_fast": 8,           # Rápida pero no ultra sensible
    "ema_slow": 21,          # Más estable para tendencias
    
    # Gestión de riesgo realista
    "stop_loss_multiplier": 1.5,    # SL más conservador
    "take_profit_multiplier": 2.5,  # TP realista (R:R 1:1.67)
    
    # Filtros de calidad balanceados
    "min_atr_threshold": 0.00010,   # ATR mínimo para volatilidad adecuada
    "max_spread_threshold": 0.00020, # Spread máximo aceptable
    "min_confidence": 0.65,         # Confianza mínima balanceada
    
    # Parámetros de volumen y momentum estables
    "volume_sma_period": 20,        # Período estándar más estable
    "momentum_period": 10,          # Momentum balanceado
    "bollinger_period": 20,         # Período estándar
    "bollinger_std": 2.0,           # Desviación estándar
    
    # Indicadores con parámetros balanceados
    "adx_period": 14,
    "adx_threshold": 25,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "stoch_k": 14,
    "stoch_d": 3,
    "williams_r_period": 14,
    
    # Configuración de sesiones de trading optimizada
    "trading_sessions": {
        "london": {"start": "08:00", "end": "17:00"},
        "new_york": {"start": "13:00", "end": "22:00"},
        "overlap": {"start": "13:00", "end": "17:00"}  # Sesión de mayor liquidez
    },
    
    # Filtros adicionales para calidad
    "use_session_filter": True,
    "use_volatility_filter": True,
    "use_trend_filter": True,
    "use_momentum_filter": True
}