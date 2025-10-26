#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estrategia especializada para scalping en 1M con gating de ATR, spread y confianza.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from enum import Enum, auto

class SignalType(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()
    def __str__(self) -> str:
        return self.name

class M1ScalpingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('M1ScalpingStrategy')
        self._last_signal = str(SignalType.HOLD)
        self._last_trade_step = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close, high, low = df['close'], df['high'], df['low']

        # EMAs
        df['ema_fast'] = close.ewm(span=int(self.config.get('ema_fast', 4))).mean()
        df['ema_medium'] = close.ewm(span=int(self.config.get('ema_medium', 9))).mean()
        df['ema_slow'] = close.ewm(span=int(self.config.get('ema_slow', 21))).mean()
        df['ema_cross'] = df['ema_fast'] - df['ema_medium']
        df['ema_trend'] = df['ema_medium'] - df['ema_slow']

        # RSI
        rsi_period = int(self.config.get('rsi_period', 13))
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        loss = loss.replace(0, np.finfo(float).eps)
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger
        bb_period = int(self.config.get('bollinger_period', 20))
        bb_std = float(self.config.get('bollinger_std', 2.0))
        mid = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        df['bb_upper'] = mid + std * bb_std
        df['bb_lower'] = mid - std * bb_std
        rng = (df['bb_upper'] - df['bb_lower']).replace(0, np.finfo(float).eps)
        df['bb_position'] = (close - df['bb_lower']) / rng

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(int(self.config.get('atr_period', 14))).mean()

        # ADX simplificado
        adx_period = int(self.config.get('adx_period', 12))
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr_roll = tr.rolling(adx_period).sum().replace(0, np.finfo(float).eps)
        df['adx'] = (100.0 * (abs(plus_dm.rolling(adx_period).sum() - minus_dm.rolling(adx_period).sum()) / tr_roll)).rolling(adx_period).mean()

        # Momentum micro (M1)
        df['momentum_3'] = close.pct_change(3)
        df['momentum_5'] = close.pct_change(5)
        df['price_change'] = close.diff()

        # Micro-pullback heurístico
        df['micro_pullback_ok'] = (
            (df['ema_fast'] > df['ema_medium']) &
            (df['ema_medium'] > df['ema_slow']) &
            (df['bb_position'] < 0.65) &
            (df['price_change'] >= -0.00008)
        )
        df['micro_breakout_ok'] = (
            (df['ema_fast'] < df['ema_medium']) &
            (df['ema_medium'] < df['ema_slow']) &
            (df['bb_position'] > 0.35) &
            (df['price_change'] <= 0.00008)
        )

        return df.dropna()

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        if len(df) < max(21, int(self.config.get('ema_slow', 21)) + 5):
            return str(SignalType.HOLD), 0.0

        current = df.iloc[-1]
        prev = df.iloc[-2]
        cooldown_bars = int(self.config.get('cooldown_bars', 0))
        cooldown_active = (self._last_trade_step is not None and (len(df) - self._last_trade_step) < cooldown_bars)

        # Gating por ATR y spread
        atr_ok = float(current.get('atr', 0.0)) >= float(self.config.get('min_atr_threshold', 0.00010))
        spread = float(df['spread'].iloc[-1]) if 'spread' in df.columns else 0.0
        spread_ok = spread <= float(self.config.get('max_spread_threshold', 0.00018))

        if self.config.get('use_volatility_filter', True) and not atr_ok:
            return str(SignalType.HOLD), 0.0
        if not spread_ok:
            return str(SignalType.HOLD), 0.0
        if cooldown_active:
            conf = 0.5
            return str(SignalType.HOLD), conf

        adx_ok = float(current.get('adx', 0.0)) >= float(self.config.get('adx_min', 22))

        buy_conditions = [
            current['ema_fast'] > current['ema_medium'],
            current['ema_medium'] > current['ema_slow'],
            current['ema_cross'] > prev['ema_cross'],
            current['rsi'] > float(self.config.get('rsi_oversold', 30)),
            current['bb_position'] < 0.75,
            current['momentum_3'] > -0.08,
            current['micro_pullback_ok'],
            atr_ok,
            adx_ok if self.config.get('use_trend_filter', True) else True,
        ]
        sell_conditions = [
            current['ema_fast'] < current['ema_medium'],
            current['ema_medium'] < current['ema_slow'],
            current['ema_cross'] < prev['ema_cross'],
            current['rsi'] < float(self.config.get('rsi_overbought', 70)),
            current['bb_position'] > 0.25,
            current['momentum_3'] < 0.08,
            current['micro_breakout_ok'],
            atr_ok,
            adx_ok if self.config.get('use_trend_filter', True) else True,
        ]

        total = len(buy_conditions)
        buy_score = sum(bool(c) for c in buy_conditions)
        sell_score = sum(bool(c) for c in sell_conditions)
        min_score = max(4, int(total * 0.35))  # un poco más estricto que 0.3

        min_conf = float(self.config.get('min_confidence', 0.70))
        if buy_score >= min_score and buy_score > sell_score:
            confidence = min(buy_score / total, 1.0)
            if confidence < min_conf: return str(SignalType.HOLD), confidence
            self._last_signal = str(SignalType.BUY)
            self._last_trade_step = len(df)
            return str(SignalType.BUY), confidence
        if sell_score >= min_score and sell_score > buy_score:
            confidence = min(sell_score / total, 1.0)
            if confidence < min_conf: return str(SignalType.HOLD), confidence
            self._last_signal = str(SignalType.SELL)
            self._last_trade_step = len(df)
            return str(SignalType.SELL), confidence

        confidence = max(buy_score, sell_score) / total
        return str(SignalType.HOLD), confidence

    def calculate_dynamic_exits(self, df: pd.DataFrame, signal: str, entry_price: float) -> Tuple[float, float]:
        atr = float(df['atr'].iloc[-1])
        tp_mult = float(self.config.get('take_profit_multiplier', 1.6))
        sl_mult = float(self.config.get('stop_loss_multiplier', 1.0))
        min_d = float(self.config.get('min_distance_pips', 2)) * 0.0001
        max_d = float(self.config.get('max_distance_pips', 12)) * 0.0001

        base_sl = atr * sl_mult
        base_tp = atr * tp_mult
        if signal == str(SignalType.BUY):
            sl = max(entry_price - max_d, min(entry_price - min_d, entry_price - base_sl))
            tp = min(entry_price + max_d * 1.5, max(entry_price + min_d, entry_price + base_tp))
        else:
            sl = min(entry_price + max_d, max(entry_price + min_d, entry_price + base_sl))
            tp = max(entry_price - max_d * 1.5, min(entry_price - min_d, entry_price - base_tp))
        return sl, tp