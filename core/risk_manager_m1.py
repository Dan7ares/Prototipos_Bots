#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gestión de riesgo especializada para 1M con escalado por volatilidad y límites diarios.
"""

import logging
from dataclasses import dataclass
from typing import Dict

@dataclass
class RiskState:
    equity: float
    trades_today: int
    consecutive_losses: int
    daily_pnl: float

class RiskManagerM1:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('RiskManagerM1')
        self.state = RiskState(equity=self.config.get('initial_capital', 10000.0),
                               trades_today=0, consecutive_losses=0, daily_pnl=0.0)

    def compute_position_size(self, atr: float, entry_price: float, sl_price: float) -> float:
        """
        Calcula el tamaño en lotes para que la pérdida en SL ≈ risk_per_trade * equity.
        """
        risk_pct = float(self.config.get('risk_per_trade', 0.0025))
        equity = float(self.state.equity)
        risk_usd = equity * risk_pct

        # Distancia a SL en pips
        sl_distance_price = abs(entry_price - sl_price)
        sl_pips = max(sl_distance_price * 100000, 0.1)
        pip_value_usd_per_lot = 10.0  # pip value estándar

        lots = max(risk_usd / (sl_pips * pip_value_usd_per_lot), 0.01)
        return round(lots, 2)

    def should_stop_trading(self) -> bool:
        max_dd = float(self.config.get('daily_loss_limit_pct', 0.02)) * self.state.equity
        if self.state.daily_pnl <= -max_dd:
            self.logger.warning("⚠️ Límite de pérdida diaria alcanzado. Deteniendo trading.")
            return True
        if self.state.consecutive_losses >= int(self.config.get('max_consecutive_losses', 4)):
            self.logger.warning("⚠️ Máximo de pérdidas consecutivas alcanzado. Pausa.")
            return True
        return False

    def update_after_trade(self, profit_usd: float) -> None:
        self.state.trades_today += 1
        self.state.daily_pnl += profit_usd
        self.state.equity += profit_usd
        if profit_usd <= 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def slippage_pips(self, atr: float, confidence: float) -> float:
        """
        Slippage dinámico basado en configuración y ATR.
        Permite 0 si 'slippage_pips_floor' == 0.0.
        """
        base = float(self.config.get('slippage_pips_base', 0.4))
        scale = float(self.config.get('slippage_pips_scale_atr', 0.10))
        floor = float(self.config.get('slippage_pips_floor', 0.1))
        atr_pips = atr * 100000
        value = base + scale * atr_pips * (1.0 - max(min(confidence, 1.0), 0.0))
        return max(floor, value)