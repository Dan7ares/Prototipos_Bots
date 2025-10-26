#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estrategia de Scalping - Módulo de Análisis Técnico y Generación de Señales
---------------------------------------------------------------------------
Este módulo implementa una estrategia de scalping avanzada basada en
múltiples indicadores técnicos para generar señales de trading de alta calidad.

Autor: Trading Bot Team
Versión: 2.0
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional, Union
from enum import Enum, auto

class SignalType(Enum):
    """Enumeración de los tipos de señales posibles"""
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    
    def __str__(self) -> str:
        return self.name

class ScalpingStrategy:
    """
    Estrategia de scalping basada en análisis técnico avanzado.
    
    Esta clase implementa una estrategia de trading que combina múltiples
    indicadores técnicos (EMAs, RSI, Bollinger Bands, ATR) para generar
    señales de compra/venta de alta calidad con gestión dinámica de riesgos.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa la estrategia de scalping con la configuración proporcionada.
        """
        self.config = config
        self.logger = logging.getLogger('ScalpingStrategy')
        self.last_signal = str(SignalType.HOLD)
        # Tracking de cooldown por número de barras
        self._last_trade_step = None

        # Validar configuración
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Valida que la configuración tenga todos los parámetros necesarios.
        """
        required_params = [
            'ema_fast', 'ema_slow',
            'rsi_period', 'rsi_oversold', 'rsi_overbought',
            'bollinger_period', 'bollinger_std', 
            'take_profit_multiplier', 'stop_loss_multiplier'
        ]
        defaults = {
            'ema_fast': 8, 'ema_slow': 21,
            'rsi_period': 14, 'rsi_oversold': 25, 'rsi_overbought': 75,
            'bollinger_period': 20, 'bollinger_std': 2.0, 
            'take_profit_multiplier': 2.5, 'stop_loss_multiplier': 1.5,
            'min_atr_threshold': 0.00010, 'max_spread_threshold': 0.00020,
            'min_confidence': 0.65, 'use_session_filter': True,
            'use_volatility_filter': True, 'use_trend_filter': True,
            'min_distance_pips': 5, 'max_distance_pips': 50
        }
        for param in required_params:
            if param not in self.config:
                self.logger.warning(f"Parámetro '{param}' no encontrado, usando valor por defecto")
                self.config[param] = defaults.get(param, 1.0)
        for param, default_value in defaults.items():
            if param not in self.config:
                self.config[param] = default_value
        # NUEVOS PARÁMETROS PARA ANÁLISIS AVANZADO
        advanced_defaults = {
            'fib_lookback': 40,
            'support_resistance_lookback': 15,
            'min_confirmations': 3,
            'mfi_period': 12,
            'vwap_enabled': True,
            'sr_tolerance_atr_multiplier': 0.6,
            'adx_period': 12,
            'adx_min': 18,
            'cooldown_bars': 2,
            'mfi_buy_min': 50.0,
            'mfi_sell_max': 50.0,
            'vwap_atr_multiplier': 0.4
        }
        for param, default_value in advanced_defaults.items():
            if param not in self.config:
                self.config[param] = default_value
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos optimizados, incluyendo ADX y niveles SR/fib.
        """
        df = df.copy()
        close, high, low = df['close'], df['high'], df['low']
        ema_fast_period = int(self.config.get('ema_fast', 8))
        ema_slow_period = int(self.config.get('ema_slow', 21))
        df['ema_fast'] = close.ewm(span=ema_fast_period).mean()
        df['ema_slow'] = close.ewm(span=ema_slow_period).mean()
        df['ema_medium'] = close.ewm(span=(ema_fast_period + ema_slow_period) // 2).mean()
        
        # Cruces de EMAs
        df['ema_cross'] = df['ema_fast'] - df['ema_medium']
        df['ema_trend'] = df['ema_medium'] - df['ema_slow']
        
        # === INDICADORES DE MOMENTUM OPTIMIZADOS ===
        # RSI para sobrecompra/sobreventa (más restrictivo)
        rsi_period = self.config.get('rsi_period', 14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        
        # Evitar división por cero
        loss = loss.replace(0, np.finfo(float).eps)
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum de precio (múltiples períodos para mejor análisis)
        df['momentum_3'] = close.pct_change(3)
        df['momentum_5'] = close.pct_change(5)
        df['price_change'] = close.diff()
        
        # === INDICADORES DE VOLATILIDAD OPTIMIZADOS ===
        # Bollinger Bands (usando configuración optimizada)
        bb_period = self.config.get('bollinger_period', 20)
        bb_std = self.config.get('bollinger_std', 2.0)
        
        df['bb_middle'] = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        
        # Posición relativa en las bandas (0 = banda inferior, 1 = banda superior)
        bb_range = df['bb_upper'] - df['bb_lower']
        bb_range = bb_range.replace(0, np.finfo(float).eps)
        df['bb_position'] = (close - df['bb_lower']) / bb_range
        
        # ATR para volatilidad y gestión de riesgos
        atr_period = self.config.get('atr_period', 12)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_period).mean()
        # === ADX simplificado ===
        adx_period = int(self.config.get('adx_period', 14))
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr_rolling = tr.rolling(adx_period).sum()
        plus_dm_rolling = plus_dm.rolling(adx_period).sum()
        minus_dm_rolling = minus_dm.rolling(adx_period).sum()
        eps = np.finfo(float).eps
        di_plus = 100.0 * (plus_dm_rolling / tr_rolling.replace(0, eps))
        di_minus = 100.0 * (minus_dm_rolling / tr_rolling.replace(0, eps))
        dx = 100.0 * (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, eps))
        df['adx'] = dx.rolling(adx_period).mean()
        
        # Volatilidad normalizada (más sensible)
        df['volatility'] = close.rolling(10).std() / close.rolling(10).mean()
        
        # === INDICADORES ADICIONALES PARA ALTA RENTABILIDAD ===
        
        # MACD para confirmación de tendencia
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic para momentum
        lowest_low = low.rolling(14).min()
        highest_high = high.rolling(14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        df['stoch_k'] = k_percent.rolling(3).mean()
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R para sobrecompra/sobreventa adicional
        df['williams_r'] = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        # === INDICADORES DE VOLUMEN (si está disponible) ===
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
            df['volume_ma'] = volume.rolling(20).mean()
            df['volume_ma'] = df['volume_ma'].replace(0, np.finfo(float).eps)
            df['volume_trend'] = volume / df['volume_ma']
            df['obv'] = (volume * close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
            price_direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            df['volume_price_trend'] = df['volume_trend'] * price_direction

        # === MFI (Money Flow Index) ===
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
        if vol_col is not None:
            typical_price = (high + low + close) / 3.0
            money_flow = typical_price * df[vol_col]
            tp_diff = typical_price.diff()
            mfi_period = int(self.config.get('mfi_period', 14))
            pos_flow = money_flow.where(tp_diff > 0, 0.0).rolling(mfi_period).sum()
            neg_flow = money_flow.where(tp_diff < 0, 0.0).rolling(mfi_period).sum()
            neg_flow = neg_flow.replace(0, np.finfo(float).eps)
            mfr = pos_flow / neg_flow
            df['mfi'] = 100.0 - (100.0 / (1.0 + mfr))

        # === VWAP (Volume Weighted Average Price) ===
        if vol_col is not None and self.config.get('vwap_enabled', True):
            typical_price = (high + low + close) / 3.0
            cum_v = df[vol_col].cumsum().replace(0, np.finfo(float).eps)
            cum_vp = (typical_price * df[vol_col]).cumsum()
            df['vwap'] = cum_vp / cum_v

        # === NIVELES DE FIBONACCI SOBRE LOOKBACK ===
        fib_lookback = int(self.config.get('fib_lookback', 50))
        window = df.iloc[-fib_lookback:] if len(df) >= fib_lookback else df
        swing_high = window['high'].max()
        swing_low = window['low'].min()
        range_eps = max(swing_high - swing_low, np.finfo(float).eps)
        df['fib_retracement'] = (close - swing_low) / range_eps  # 0..1
        # niveles clásicos como valores absolutos (para anotación/visualización)
        df['fib_23'] = swing_high - 0.236 * (swing_high - swing_low)
        df['fib_38'] = swing_high - 0.382 * (swing_high - swing_low)
        df['fib_50'] = swing_high - 0.500 * (swing_high - swing_low)
        df['fib_61'] = swing_high - 0.618 * (swing_high - swing_low)
        df['fib_78'] = swing_high - 0.786 * (swing_high - swing_low)

        # === SOPORTES/RESISTENCIAS (fractales/pivots) ===
        sr_lb = int(self.config.get('support_resistance_lookback', 20))
        ph = (high > high.shift(1)) & (high > high.shift(-1))
        pl = (low < low.shift(1)) & (low < low.shift(-1))
        df['pivot_high'] = ph.fillna(False)
        df['pivot_low'] = pl.fillna(False)
        # proximidad a SR con tolerancia basada en ATR
        tol_source = df['atr'].rolling(5).mean() if 'atr' in df.columns else close.rolling(5).std()
        tol_multiplier = float(self.config.get('sr_tolerance_atr_multiplier', 0.75))
        tol = (tol_source.bfill()) * tol_multiplier
        recent_highs = df['high'].where(df['pivot_high']).rolling(sr_lb, min_periods=1).max()
        recent_lows = df['low'].where(df['pivot_low']).rolling(sr_lb, min_periods=1).min()
        df['nearest_resistance'] = recent_highs.ffill()
        df['nearest_support'] = recent_lows.ffill()
        df['near_resistance'] = (abs(df['nearest_resistance'] - close) <= tol).fillna(False)
        df['near_support'] = (abs(close - df['nearest_support']) <= tol).fillna(False)

        # === PATRONES DE VELAS (confirmaciones) ===
        body = (close - df['open']).abs()
        upper_wick = df['high'] - np.maximum(df['open'], close)
        lower_wick = np.minimum(df['open'], close) - df['low']
        df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & (close > df['open']) & (df['open'] <= df['close'].shift(1)) & (close >= df['open'].shift(1))
        df['bearish_engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['open'] > close) & (close <= df['open'].shift(1)) & (df['open'] >= df['close'].shift(1))
        df['bullish_pinbar'] = (lower_wick > body * 2) & (lower_wick > upper_wick * 2)
        df['bearish_pinbar'] = (upper_wick > body * 2) & (upper_wick > lower_wick * 2)
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        df['doji'] = body < (df['high'] - df['low']) * 0.1

        # === FILTROS DE CALIDAD ===
        min_movement = df['atr'].rolling(5).mean() * 0.1
        df['significant_move'] = abs(df['price_change']) > min_movement
        df['trend_clarity'] = abs(df['ema_fast'] - df['ema_slow']) / df['atr']
        df['momentum_consistency'] = (
            (df['momentum_3'] > 0) & (df['momentum_5'] > 0) |
            (df['momentum_3'] < 0) & (df['momentum_5'] < 0)
        )
        return df.dropna()
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Genera señales de trading basadas en los indicadores calculados.
        """
        min_required = max(21, self.config['ema_slow'] + 5)
        if len(df) < min_required:
            self.logger.warning(f"Datos insuficientes: {len(df)} filas (mínimo {min_required})")
            return str(SignalType.HOLD), 0.0
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        # Cooldown operativo (evita sobretrading)
        cooldown_bars = int(self.config.get('cooldown_bars', 0))
        cooldown_active = (self._last_trade_step is not None and (len(df) - self._last_trade_step) < cooldown_bars)
        # Patrones
        pattern_bullish = bool(current.get('bullish_engulfing', False) or current.get('bullish_pinbar', False))
        pattern_bearish = bool(current.get('bearish_engulfing', False) or current.get('bearish_pinbar', False))
        # Zonas de Fibonacci
        fib_ret = float(current.get('fib_retracement', 0.5))
        fib_buy_zone = 0.35 <= fib_ret <= 0.65
        fib_sell_zone = 0.35 <= fib_ret <= 0.65
        # Cerca de soportes/resistencias
        near_sup = bool(current.get('near_support', False))
        near_res = bool(current.get('near_resistance', False))
        # MFI y VWAP con umbrales ajustados
        mfi_val = float(current.get('mfi', 50.0))
        mfi_buy = mfi_val >= float(self.config.get('mfi_buy_min', 55.0))
        mfi_sell = mfi_val <= float(self.config.get('mfi_sell_max', 45.0))
        vwap_val = float(current.get('vwap', current['close']))
        vwap_buy = current['close'] >= vwap_val
        vwap_sell = current['close'] <= vwap_val
        vwap_near = True
        if 'vwap' in df.columns and 'atr' in df.columns:
            vwap_near = abs(current['close'] - vwap_val) <= float(self.config.get('vwap_atr_multiplier', 0.5)) * float(current.get('atr', 0.0) or 0.0)
        # ADX gating
        adx_val = float(current.get('adx', 25.0))
        adx_ok = adx_val >= float(self.config.get('adx_min', 20))
        # === CONDICIONES PARA COMPRA - AJUSTE ITERATIVO 3: Más permisivas ===
        buy_conditions = [
            current['ema_fast'] > current['ema_medium'],  # Tendencia alcista básica
            current['ema_medium'] > current['ema_slow'],  # Confirmación de tendencia
            current['ema_cross'] > prev['ema_cross'],     # Momentum alcista
            current['rsi'] > self.config['rsi_oversold'], # RSI no sobrevendido
            current['momentum_3'] > -0.1,                 # Momentum ligeramente positivo (más permisivo)
            current['price_change'] > -0.0001,            # Cambio de precio no muy negativo (más permisivo)
            current['bb_position'] < 0.7,                 # No en zona extrema superior (más permisivo)
            current['volatility'] > 0.00010,              # Volatilidad mínima reducida
            # Condiciones opcionales (no todas requeridas)
            True,  # pattern_bullish (removido como requisito)
            True,  # fib_buy_zone and near_sup (removido como requisito)
            True,  # mfi_buy (removido como requisito)
            True,  # vwap_buy and vwap_near (removido como requisito)
            adx_ok if self.config.get('use_trend_filter', True) else True
        ]
        # === CONDICIONES PARA VENTA - AJUSTE ITERATIVO 3: Más permisivas ===
        sell_conditions = [
            current['ema_fast'] < current['ema_medium'],  # Tendencia bajista básica
            current['ema_medium'] < current['ema_slow'],  # Confirmación de tendencia
            current['ema_cross'] < prev['ema_cross'],     # Momentum bajista
            current['rsi'] < self.config['rsi_overbought'], # RSI no sobrecomprado
            current['momentum_3'] < 0.1,                  # Momentum ligeramente negativo (más permisivo)
            current['price_change'] < 0.0001,             # Cambio de precio no muy positivo (más permisivo)
            current['bb_position'] > 0.3,                 # No en zona extrema inferior (más permisivo)
            current['volatility'] > 0.00010,              # Volatilidad mínima reducida
            # Condiciones opcionales (no todas requeridas)
            True,  # pattern_bearish (removido como requisito)
            True,  # fib_sell_zone and near_res (removido como requisito)
            True,  # mfi_sell (removido como requisito)
            True,  # vwap_sell and vwap_near (removido como requisito)
            adx_ok if self.config.get('use_trend_filter', True) else True
        ]
        # Volumen
        if 'volume_trend' in current:
            buy_conditions.append(current['volume_trend'] > 1.0)
            sell_conditions.append(current['volume_trend'] > 1.0)
            if 'volume_price_trend' in current:
                buy_conditions.append(current['volume_price_trend'] > 0)
                sell_conditions.append(current['volume_price_trend'] < 0)
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        total_conditions = len(buy_conditions)
        
        # AJUSTE ITERATIVO 3: Score mínimo muy reducido para más señales
        min_score = max(4, int(total_conditions * 0.3))  # Reducido de 0.5 a 0.3
        min_confirmations = int(self.config.get('min_confirmations', 2))  # Reducido por defecto
        
        # Bonus por continuidad de señal (reducido)
        if self.last_signal == str(SignalType.BUY) and buy_score >= min_score - 2:
            buy_score += 0.3  # Reducido de 0.5
        if self.last_signal == str(SignalType.SELL) and sell_score >= min_score - 2:
            sell_score += 0.3  # Reducido de 0.5
        
        # Confirmaciones simplificadas - AJUSTE 3: Más permisivas
        buy_confirms = sum([
            current['ema_fast'] > current['ema_medium'],  # Simplificado
            current['momentum_3'] > -0.1,                 # Más permisivo
            current['rsi'] > self.config['rsi_oversold'], # Básico
            True,  # Siempre cuenta como confirmación adicional
        ])
        sell_confirms = sum([
            current['ema_fast'] < current['ema_medium'],  # Simplificado
            current['momentum_3'] < 0.1,                  # Más permisivo
            current['rsi'] < self.config['rsi_overbought'], # Básico
            True,  # Siempre cuenta como confirmación adicional
        ])
        min_conf = float(self.config.get('min_confidence', 0.5))  # Reducido de 0.7 a 0.5
        
        # Cooldown y gating por ADX (más permisivo) - AJUSTE 3
        if cooldown_active:
            self.logger.debug("Cooldown activo, manteniendo HOLD")
            confidence = max(buy_score, sell_score) / total_conditions
            return str(SignalType.HOLD), confidence
            
        # ADX gating más permisivo - AJUSTE 3
        if (not adx_ok) and max(buy_confirms, sell_confirms) < (min_confirmations):  # Removido +1
            confidence = max(buy_score, sell_score) / total_conditions
            self.logger.debug(f"Tendencia débil (ADX={adx_val:.1f}), HOLD con confianza {confidence:.2f}")
            return str(SignalType.HOLD), confidence
        # Señal final
        if buy_score >= min_score and buy_score > sell_score and buy_confirms >= min_confirmations:
            confidence = min(buy_score / total_conditions, 1.0)  # Limitar a 1.0
            if confidence < min_conf:
                return str(SignalType.HOLD), confidence
            self.last_signal = str(SignalType.BUY)
            self._last_trade_step = len(df)
            self.logger.info(f"Señal: BUY, Confianza: {confidence:.2f}, Score: {buy_score}/{total_conditions}, Confirms: {buy_confirms}")
            return str(SignalType.BUY), confidence
        elif sell_score >= min_score and sell_score > buy_score and sell_confirms >= min_confirmations:
            confidence = min(sell_score / total_conditions, 1.0)  # Limitar a 1.0
            if confidence < min_conf:
                return str(SignalType.HOLD), confidence
            self.last_signal = str(SignalType.SELL)
            self._last_trade_step = len(df)
            self.logger.info(f"Señal: SELL, Confianza: {confidence:.2f}, Score: {sell_score}/{total_conditions}, Confirms: {sell_confirms}")
            return str(SignalType.SELL), confidence
        else:
            confidence = min(max(buy_score, sell_score) / total_conditions, 1.0)  # Limitar a 1.0
            self.logger.debug(f"HOLD, Confianza: {confidence:.2f}, Buy: {buy_score}, Sell: {sell_score}, Confirms(B/S): {buy_confirms}/{sell_confirms}")
            return str(SignalType.HOLD), confidence
    
    def calculate_dynamic_exits(self, df: pd.DataFrame, signal: str, entry_price: float) -> Tuple[float, float]:
        """
        Calcula niveles dinámicos de stop loss y take profit basados en ATR.
        
        Args:
            df: DataFrame con indicadores calculados
            signal: Tipo de señal ("BUY" o "SELL")
            entry_price: Precio de entrada de la operación
            
        Returns:
            Tupla con (stop_loss, take_profit)
            
        Raises:
            ValueError: Si el DataFrame no contiene ATR o la señal es inválida
        """
        # Validar entrada
        if signal not in [str(SignalType.BUY), str(SignalType.SELL)]:
            raise ValueError(f"Señal inválida: {signal}. Debe ser 'BUY' o 'SELL'")
        
        if 'atr' not in df.columns:
            raise ValueError("DataFrame no contiene columna 'atr'")
        
        # Obtener ATR actual
        atr = df['atr'].iloc[-1]
        current_close = df['close'].iloc[-1]
        
        # Calcular relación riesgo/beneficio mejorada
        tp_multiplier = self.config['take_profit_multiplier']
        sl_multiplier = self.config['stop_loss_multiplier']
        
        # Calcular niveles base
        if signal == str(SignalType.BUY):
            stop_loss = entry_price - (atr * sl_multiplier)
            take_profit = entry_price + (atr * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr * sl_multiplier)
            take_profit = entry_price - (atr * tp_multiplier)
        
        # Convertir pips a precio para distancias mínimas y máximas
        min_distance = self.config['min_distance_pips'] * 0.0001
        max_distance = self.config['max_distance_pips'] * 0.0001
        
        # Ajustar niveles para asegurar distancias mínimas y máximas
        if signal == str(SignalType.BUY):
            # Para compras: SL debe estar por debajo del precio de entrada
            stop_loss = min(entry_price - min_distance, stop_loss)  # Asegurar distancia mínima
            stop_loss = max(entry_price - max_distance, stop_loss)  # Limitar distancia máxima
            
            # Para compras: TP debe estar por encima del precio de entrada
            take_profit = max(entry_price + min_distance, take_profit)  # Asegurar distancia mínima
            take_profit = min(entry_price + max_distance * 1.5, take_profit)  # Permitir TP más amplio
        else:  # SELL
            # Para ventas: SL debe estar por encima del precio de entrada
            stop_loss = max(entry_price + min_distance, stop_loss)  # Asegurar distancia mínima
            stop_loss = min(entry_price + max_distance, stop_loss)  # Limitar distancia máxima
            
            # Para ventas: TP debe estar por debajo del precio de entrada
            take_profit = min(entry_price - min_distance, take_profit)  # Asegurar distancia mínima
            take_profit = max(entry_price - max_distance * 1.5, take_profit)  # Permitir TP más amplio
        
        # Registrar niveles calculados
        self.logger.info(f"Niveles calculados - Entrada: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        
        return stop_loss, take_profit
    
    def calculate_profit_loss(self, entry_price, exit_price, signal_type, position_size=1.0):
        """
        Calcular profit/loss con mayor precisión
        """
        try:
            if signal_type == 'BUY':
                # Para posición larga: ganancia cuando precio sube
                profit_pips = (exit_price - entry_price) * 100000  # Convertir a pips
            else:  # SELL
                # Para posición corta: ganancia cuando precio baja
                profit_pips = (entry_price - exit_price) * 100000  # Convertir a pips
            
            # Calcular profit en USD (asumiendo $10 por pip para lote estándar)
            pip_value = 10.0  # USD por pip para EURUSD
            profit_usd = profit_pips * pip_value * position_size
            
            # Aplicar spread (costo de transacción)
            spread_cost = 1.5 * pip_value * position_size  # 1.5 pips de spread típico
            net_profit = profit_usd - spread_cost
            
            return {
                'profit_pips': profit_pips,
                'profit_usd': net_profit,
                'gross_profit': profit_usd,
                'spread_cost': spread_cost
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando P&L: {e}")
            return {
                'profit_pips': 0,
                'profit_usd': 0,
                'gross_profit': 0,
                'spread_cost': 0
            }
