import logging
import time
import MetaTrader5 as mt5
import datetime as _dt
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass
class PositionParameters:
    """Parámetros calculados para una posición de trading"""
    volume: float
    stop_loss: float
    take_profit: float
    stop_loss_usd: float
    take_profit_usd: float
    risk_reward_ratio: float
    confidence_score: float

@dataclass
class MarketOpportunity:
    """Evaluación de oportunidad de mercado"""
    signal_strength: float  # 0-1
    technical_score: float  # 0-1
    market_conditions_score: float  # 0-1
    risk_reward_score: float  # 0-1
    overall_score: float  # 0-1
    recommendation: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0-1
    reasons: list

class IntelligentRiskManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('IntelligentRiskManager')
        self.daily_stats = {
            'trades_today': 0,
            'loss_today': 0,
            'profit_today': 0,
            'daily_start_balance': 0,
            'last_trade_time': None,
            'max_drawdown_today': 0,
            'consecutive_losses': 0,
            'risk_alerts': []
        }
        # Configuración de capital utilizada por varios métodos
        self.capital_config = {
            'max_risk_per_trade': 0.02,         # 2% riesgo por operación
            'max_daily_drawdown': 0.05,         # 5% drawdown diario
            'min_risk_reward_ratio': 1.5        # R:R mínimo
        }
        # Configuración de trailing stop usada por implement_trailing_stop
        @dataclass
        class TrailingConfig:
            activation_profit: float
            trail_distance: float
        self.trailing_config = TrailingConfig(
            activation_profit=1.0,  # Activar trailing al ganar $1
            trail_distance=0.5      # Distancia de trailing $0.5
        )
        # Estados que usan otros métodos
        self.active_positions = {}
        self.risk_alerts_history = []
        
        # Configuración de evaluación de oportunidad (throttling/caché)
        self._opportunity_eval_cooldown = self.config.get('opportunity_eval_cooldown', 10)
        self._last_opportunity = {
            'symbol': None,
            'timestamp': 0,
            'overall_score': None,
            'recommendation': None,
            'result': None
        }
    
    def initialize_daily_stats(self, balance: float):
        """Reinicia estadísticas diarias con idempotencia diaria"""
        today = _dt.date.today().isoformat()
        # Evita doble inicialización/log si ya se inicializó en la misma fecha
        if getattr(self, "daily_stats", None) and self.daily_stats.get("date") == today:
            return
        self.daily_stats = {
            'date': today,
            'trades_today': 0,
            'loss_today': 0,
            'profit_today': 0,
            'daily_start_balance': balance,
            'last_trade_time': None,
            'max_drawdown_today': 0,
            'consecutive_losses': 0,
            'risk_alerts': []
        }
        self.logger.info(f"Estadísticas diarias inicializadas - Capital: ${balance:.2f}")
    
    def calculate_intelligent_position(self, current_balance: float, entry_price: float, 
                                     market_conditions: dict = None) -> PositionParameters:
        """
        Calcula parámetros de posición inteligente con gestión de capital de $50
        
        Args:
            current_balance: Balance actual de la cuenta
            entry_price: Precio de entrada propuesto
            market_conditions: Condiciones actuales del mercado
            
        Returns:
            PositionParameters: Parámetros calculados para la posición
        """
        try:
            # Capital base de $50 para gestión de riesgo
            base_capital = 50.0
            effective_capital = min(current_balance, base_capital)
            
            # Riesgo máximo por operación (2% del capital efectivo)
            max_risk_usd = effective_capital * self.capital_config['max_risk_per_trade']  # $1 máximo
            
            # Stop loss base de $2 (4% del capital de $50)
            base_stop_loss_usd = 2.0
            
            # Ajustar stop loss según volatilidad del mercado
            volatility_multiplier = 1.0
            if market_conditions:
                volatility = market_conditions.get('volatility', 0)
                spread = market_conditions.get('spread', 0)
                
                # Ajustar según volatilidad
                if volatility > 0.0003:  # Alta volatilidad
                    volatility_multiplier = 1.5
                    self.logger.info("Alta volatilidad detectada - Stop loss ampliado")
                elif volatility < 0.0001:  # Baja volatilidad
                    volatility_multiplier = 0.8
                    self.logger.info("Baja volatilidad detectada - Stop loss reducido")
            
            # Calcular stop loss dinámico
            dynamic_stop_loss_usd = min(base_stop_loss_usd * volatility_multiplier, max_risk_usd)
            
            # Convertir USD a pips (para EURUSD)
            pip_value = 0.0001
            account_currency_pip_value = 0.1  # $0.1 por pip para lote de 0.01
            
            # Calcular stop loss en pips
            stop_loss_pips = dynamic_stop_loss_usd / account_currency_pip_value
            
            # Calcular tamaño de posición basado en el riesgo
            # Volume = Risk_USD / (Stop_Loss_Pips * Pip_Value_USD)
            volume = dynamic_stop_loss_usd / (stop_loss_pips * account_currency_pip_value)
            
            # Aplicar límites de volumen
            min_volume = 0.01
            max_volume = min(effective_capital / 1000, 0.1)  # Máximo 0.1 lotes
            volume = max(min_volume, min(volume, max_volume))
            volume = round(volume, 2)
            
            # Calcular take profit dinámico (ratio 1.5:1 mínimo)
            min_reward_ratio = self.capital_config['min_risk_reward_ratio']
            take_profit_pips = stop_loss_pips * min_reward_ratio
            
            # Ajustar take profit según condiciones de mercado
            if market_conditions:
                # En alta volatilidad, objetivo más ambicioso
                if market_conditions.get('volatility', 0) > 0.0003:
                    take_profit_pips *= 1.3
                # En mercados laterales, objetivo más conservador
                elif market_conditions.get('trend_strength', 0.5) < 0.3:
                    take_profit_pips *= 0.8
            
            # Calcular precios de stop loss y take profit
            symbol_info = mt5.symbol_info(self.config.get('symbol', 'EURUSD'))
            if not symbol_info:
                symbol_info = mt5.symbol_info('EURUSDm')
            if symbol_info:
                point = symbol_info.point
                stop_loss_price = entry_price - (stop_loss_pips * point)
                take_profit_price = entry_price + (take_profit_pips * point)
            else:
                # Valores por defecto si no se puede obtener info del símbolo
                stop_loss_price = entry_price - (stop_loss_pips * 0.0001)
                take_profit_price = entry_price + (take_profit_pips * 0.0001)
            
            # Confianza basada en condiciones de mercado
            confidence_score = 0.5
            if market_conditions:
                vol = market_conditions.get('volatility', 0)
                spread = market_conditions.get('spread', 0)
                trend = market_conditions.get('trend_strength', 0)
                if 0.0001 <= vol <= 0.0005:
                    confidence_score += 0.2
                if spread <= 1.5:
                    confidence_score += 0.1
                # Ajuste moderado por fuerza de tendencia
                confidence_score += min(max(trend - 0.5, -0.5), 0.5) * 0.4
                confidence_score = max(0.0, min(confidence_score, 1.0))
            
            # Crear parámetros de posición compatibles con PositionParameters
            position_params = PositionParameters(
                volume=volume,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                stop_loss_usd=dynamic_stop_loss_usd,
                take_profit_usd=dynamic_stop_loss_usd * min_reward_ratio,
                risk_reward_ratio=min_reward_ratio,
                confidence_score=confidence_score
            )
            self.logger.info(
                f"Posición calculada - Volume: {volume}, SL: ${dynamic_stop_loss_usd:.2f} "
                f"({stop_loss_pips:.1f} pips), TP: {take_profit_pips:.1f} pips"
            )
            return position_params
            
        except Exception as e:
            self.logger.error(f"Error calculando posición inteligente: {e}")
            # Retornar parámetros seguros por defecto (coinciden con PositionParameters)
            return PositionParameters(
                volume=0.01,
                stop_loss=entry_price - 0.0020,  # 20 pips
                take_profit=entry_price + 0.0030,  # 30 pips
                stop_loss_usd=2.0,
                take_profit_usd=3.0,
                risk_reward_ratio=1.5,
                confidence_score=0.5
            )
    
    def can_open_trade(self, current_balance: float) -> Tuple[bool, str]:
        """
        Verifica si se puede abrir nueva operación con gestión de riesgo inteligente
        
        Args:
            current_balance: Balance actual de la cuenta
            
        Returns:
            Tuple[bool, str]: (Puede operar, Razón si no puede)
        """
        try:
            # Verificar límite diario de operaciones
            if self.daily_stats['trades_today'] >= self.config.get('max_daily_trades', 30):
                return False, "Límite diario de operaciones alcanzado"
            
            # Calcular drawdown diario actual
            daily_pnl = current_balance - self.daily_stats['daily_start_balance']
            daily_drawdown_pct = abs(daily_pnl) / self.daily_stats['daily_start_balance'] if daily_pnl < 0 else 0
            
            # Verificar drawdown máximo diario (5%)
            if daily_drawdown_pct > self.capital_config['max_daily_drawdown']:
                alert_msg = f"Drawdown diario excedido: {daily_drawdown_pct:.2%} > 5%"
                self._add_risk_alert(alert_msg, "CRITICAL")
                return False, alert_msg
            
            # Verificar pérdidas consecutivas (máximo 3)
            if self.daily_stats['consecutive_losses'] >= 3:
                return False, "Máximo de 3 pérdidas consecutivas alcanzado"
            
            # Verificar tiempo entre operaciones (mínimo 2 minutos para reflexión)
            if self.daily_stats['last_trade_time']:
                time_since_last = (datetime.now() - self.daily_stats['last_trade_time']).total_seconds()
                if time_since_last < 120:  # 2 minutos
                    return False, f"Esperando {120 - int(time_since_last)}s entre operaciones"
            
            # Verificar que el capital mínimo esté disponible
            if current_balance < 10.0:  # Capital mínimo para operar
                return False, "Capital insuficiente para operar de forma segura"
            
            return True, "Condiciones de riesgo OK"
            
        except Exception as e:
            self.logger.error(f"Error verificando condiciones de trading: {e}")
            return False, f"Error en verificación de riesgo: {str(e)}"
    
    def implement_trailing_stop(self, ticket: int, current_price: float, 
                               position_type: str, original_sl: float) -> bool:
        """
        Implementa trailing stop dinámico
        
        Args:
            ticket: Ticket de la posición
            current_price: Precio actual
            position_type: "BUY" o "SELL"
            original_sl: Stop loss original
            
        Returns:
            bool: True si se actualizó el trailing stop
        """
        try:
            if ticket not in self.active_positions:
                # Inicializar tracking de la posición
                self.active_positions[ticket] = {
                    'max_profit': 0,
                    'trailing_active': False,
                    'last_sl_update': original_sl
                }
            
            position_data = self.active_positions[ticket]
            
            # Obtener información de la posición
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            position = positions[0]
            entry_price = position.price_open
            current_profit = position.profit
            
            # Actualizar máximo profit
            if current_profit > position_data['max_profit']:
                position_data['max_profit'] = current_profit
            
            # Activar trailing stop después de $1 de ganancia
            if not position_data['trailing_active'] and current_profit >= self.trailing_config.activation_profit:
                position_data['trailing_active'] = True
                self.logger.info(f"Trailing stop activado para posición {ticket} - Profit: ${current_profit:.2f}")
            
            # Aplicar trailing stop si está activo
            if position_data['trailing_active']:
                trail_distance_usd = self.trailing_config.trail_distance
                
                if position_type == "BUY":
                    # Para posiciones de compra, mover SL hacia arriba
                    pip_value = 0.0001
                    trail_distance_pips = trail_distance_usd / 0.1  # $0.1 por pip
                    new_sl = current_price - (trail_distance_pips * pip_value)
                    
                    # Solo mover SL si es mejor que el actual
                    if new_sl > position_data['last_sl_update']:
                        if self._modify_position_sl(ticket, new_sl):
                            position_data['last_sl_update'] = new_sl
                            self.logger.info(f"Trailing stop actualizado para {ticket}: SL = {new_sl:.5f}")
                            return True
                
                elif position_type == "SELL":
                    # Para posiciones de venta, mover SL hacia abajo
                    pip_value = 0.0001
                    trail_distance_pips = trail_distance_usd / 0.1
                    new_sl = current_price + (trail_distance_pips * pip_value)
                    
                    # Solo mover SL si es mejor que el actual
                    if new_sl < position_data['last_sl_update']:
                        if self._modify_position_sl(ticket, new_sl):
                            position_data['last_sl_update'] = new_sl
                            self.logger.info(f"Trailing stop actualizado para {ticket}: SL = {new_sl:.5f}")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error implementando trailing stop: {e}")
            return False
    
    def _modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modifica el stop loss de una posición"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            position = positions[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": position.tp,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                self.logger.warning(f"Error modificando SL: {result.comment if result else 'Sin respuesta'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en _modify_position_sl: {e}")
            return False
    
    def implement_partial_scaling(self, ticket: int, current_profit: float) -> bool:
        """
        Implementa escalado parcial de posiciones para asegurar ganancias
        
        Args:
            ticket: Ticket de la posición
            current_profit: Ganancia actual en USD
            
        Returns:
            bool: True si se ejecutó escalado parcial
        """
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            position = positions[0]
            current_volume = position.volume
            
            # Definir niveles de escalado parcial
            scaling_levels = [
                {'profit_threshold': 1.0, 'close_percentage': 0.25},  # Cerrar 25% con $1 ganancia
                {'profit_threshold': 1.5, 'close_percentage': 0.50},  # Cerrar 50% con $1.5 ganancia
                {'profit_threshold': 2.0, 'close_percentage': 0.75},  # Cerrar 75% con $2 ganancia
            ]
            
            for level in scaling_levels:
                if current_profit >= level['profit_threshold']:
                    # Calcular volumen a cerrar
                    volume_to_close = current_volume * level['close_percentage']
                    volume_to_close = round(volume_to_close, 2)
                    
                    # Verificar que el volumen sea válido
                    if volume_to_close >= 0.01 and volume_to_close <= current_volume:
                        if self._close_partial_position(ticket, volume_to_close):
                            self.logger.info(f"Escalado parcial ejecutado: {volume_to_close} lotes "
                                           f"cerrados con ${current_profit:.2f} ganancia")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error en escalado parcial: {e}")
            return False
    
    def _close_partial_position(self, ticket: int, volume: float) -> bool:
        """Cierra parcialmente una posición"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            
            position = positions[0]
            
            # Determinar tipo de orden de cierre
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "Escalado parcial",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                self.logger.warning(f"Error cerrando posición parcial: {result.comment if result else 'Sin respuesta'}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en _close_partial_position: {e}")
            return False
    
    def monitor_active_positions(self):
        """
        Monitoreo continuo de posiciones activas con trailing stops y escalado parcial
        """
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            for position in positions:
                ticket = position.ticket
                current_price = position.price_current
                position_type = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
                current_profit = position.profit
                
                # Aplicar trailing stop
                self.implement_trailing_stop(ticket, current_price, position_type, position.sl)
                
                # Aplicar escalado parcial si hay ganancia
                if current_profit > 0:
                    self.implement_partial_scaling(ticket, current_profit)
                
                # Generar alerta si la pérdida supera el límite
                if current_profit < -2.5:  # Alerta si pérdida > $2.5
                    self._add_risk_alert(f"Posición {ticket} con pérdida elevada: ${current_profit:.2f}", "WARNING")
            
        except Exception as e:
            self.logger.error(f"Error monitoreando posiciones: {e}")
    
    def _add_risk_alert(self, message: str, level: str):
        """Agrega una alerta de riesgo al historial"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        
        self.daily_stats['risk_alerts'].append(alert)
        self.risk_alerts_history.append(alert)
        
        # Mantener solo las últimas 100 alertas
        if len(self.risk_alerts_history) > 100:
            self.risk_alerts_history = self.risk_alerts_history[-100:]
        
        # Log según el nivel
        if level == "CRITICAL":
            self.logger.critical(f"ALERTA CRÍTICA: {message}")
        elif level == "WARNING":
            self.logger.warning(f"ALERTA: {message}")
        else:
            self.logger.info(f"INFO: {message}")
    
    def evaluate_market_opportunity(self, df, current_price: float, market_conditions: dict = None) -> MarketOpportunity:
        """
        Evalúa la oportunidad de mercado basada en múltiples factores
        
        Args:
            df: DataFrame con datos de mercado e indicadores
            current_price: Precio actual del mercado
            market_conditions: Condiciones actuales del mercado
            
        Returns:
            MarketOpportunity: Evaluación completa de la oportunidad
        """
        try:
            reasons = []
            
            # Throttling/caché por símbolo para reducir ruido y CPU
            symbol = self.config.get('symbol', 'EURUSD')
            now = time.time()
            if (
                self._last_opportunity.get('result') is not None
                and self._last_opportunity.get('symbol') == symbol
                and now - self._last_opportunity.get('timestamp', 0) < self._opportunity_eval_cooldown
            ):
                return self._last_opportunity.get('result')
            
            # 1. ANÁLISIS TÉCNICO EN TIEMPO REAL
            technical_score = self._analyze_technical_indicators(df, reasons)
            
            # 2. CONDICIONES DEL MERCADO
            market_score = self._analyze_market_conditions(market_conditions, reasons)
            
            # 3. PARÁMETROS DE RIESGO/RECOMPENSA
            risk_reward_score = self._analyze_risk_reward(df, current_price, reasons)
            
            # 4. FUERZA DE LA SEÑAL
            signal_strength = self._calculate_signal_strength(df, reasons)
            
            # Calcular puntuación general (promedio ponderado reequilibrado)
            weights = {
                'technical': 0.40,
                'market': 0.25,
                'risk_reward': 0.15,
                'signal': 0.20
            }
            
            overall_score = (
                technical_score * weights['technical'] +
                market_score * weights['market'] +
                risk_reward_score * weights['risk_reward'] +
                signal_strength * weights['signal']
            )
            
            # Determinar recomendación con umbrales más adaptativos
            recommendation, confidence = self._determine_recommendation(
                overall_score, technical_score, signal_strength, risk_reward_score, reasons
            )
            
            opportunity = MarketOpportunity(
                signal_strength=signal_strength,
                technical_score=technical_score,
                market_conditions_score=market_score,
                risk_reward_score=risk_reward_score,
                overall_score=overall_score,
                recommendation=recommendation,
                confidence=confidence,
                reasons=reasons[:5]  # Acotar a las 5 razones más relevantes
            )
            
            # Logging condicional solo ante cambios relevantes
            prev_overall = self._last_opportunity.get('overall_score')
            prev_reco = self._last_opportunity.get('recommendation')
            score_delta = abs(overall_score - prev_overall) if prev_overall is not None else None
            should_log = prev_reco != recommendation or (score_delta is not None and score_delta >= 0.05)
            if should_log:
                self.logger.info(f"🎯 EVALUACIÓN [{symbol}]: Score={overall_score:.2f} Conf={confidence:.2f} Rec={recommendation}")
                self.logger.info(f"   📊 Técnico: {technical_score:.2f} | 🌍 Mercado: {market_score:.2f} | ⚖️ R:R: {risk_reward_score:.2f} | 📈 Señal: {signal_strength:.2f}")
            
            # Actualizar caché
            self._last_opportunity = {
                'symbol': symbol,
                'timestamp': now,
                'overall_score': overall_score,
                'recommendation': recommendation,
                'result': opportunity
            }
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error evaluando oportunidad de mercado: {e}")
            return MarketOpportunity(
                signal_strength=0.0,
                technical_score=0.0,
                market_conditions_score=0.0,
                risk_reward_score=0.0,
                overall_score=0.0,
                recommendation="HOLD",
                confidence=0.0,
                reasons=["Error en evaluación"]
            )
    
    def _analyze_technical_indicators(self, df, reasons: list) -> float:
        """Analiza indicadores técnicos en tiempo real"""
        try:
            score = 0.0
            factors = 0
            
            if len(df) < 5:
                reasons.append("❌ Datos insuficientes para análisis técnico")
                return 0.0
            
            # EMA Crossover
            if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
                ema_fast = df['ema_fast'].iloc[-1]
                ema_slow = df['ema_slow'].iloc[-1]
                ema_fast_prev = df['ema_fast'].iloc[-2]
                ema_slow_prev = df['ema_slow'].iloc[-2]
                
                if ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev:
                    score += 0.8
                    reasons.append("✅ Cruce alcista de EMAs detectado")
                elif ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev:
                    score += 0.8
                    reasons.append("✅ Cruce bajista de EMAs detectado")
                elif ema_fast > ema_slow:
                    score += 0.4
                    reasons.append("📈 Tendencia alcista en EMAs")
                elif ema_fast < ema_slow:
                    score += 0.4
                    reasons.append("📉 Tendencia bajista en EMAs")
                factors += 1
            
            # RSI
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 30:
                    score += 0.7
                    reasons.append(f"✅ RSI sobreventa: {rsi:.1f}")
                elif rsi > 70:
                    score += 0.7
                    reasons.append(f"✅ RSI sobrecompra: {rsi:.1f}")
                elif 40 <= rsi <= 60:
                    score += 0.3
                    reasons.append(f"📊 RSI neutral: {rsi:.1f}")
                factors += 1
            
            # Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
                close = df['close'].iloc[-1]
                bb_upper = df['bb_upper'].iloc[-1]
                bb_lower = df['bb_lower'].iloc[-1]
                
                if close <= bb_lower:
                    score += 0.6
                    reasons.append("✅ Precio en banda inferior de Bollinger")
                elif close >= bb_upper:
                    score += 0.6
                    reasons.append("✅ Precio en banda superior de Bollinger")
                factors += 1
            
            # Volumen (si está disponible)
            if 'volume' in df.columns and len(df) >= 10:
                avg_volume = df['volume'].tail(10).mean()
                current_volume = df['volume'].iloc[-1]
                
                if current_volume > avg_volume * 1.5:
                    score += 0.3
                    reasons.append("✅ Volumen elevado detectado")
                factors += 1
            
            return min(score / max(factors, 1), 1.0) if factors > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error en análisis técnico: {e}")
            reasons.append(f"❌ Error en análisis técnico: {str(e)}")
            return 0.0
    
    def _analyze_market_conditions(self, market_conditions: dict, reasons: list) -> float:
        """Analiza las condiciones actuales del mercado"""
        try:
            if not market_conditions:
                reasons.append("⚠️ Sin datos de condiciones de mercado")
                return 0.5
            
            score = 0.0
            factors = 0
            
            # Volatilidad
            volatility = market_conditions.get('volatility', 0)
            if volatility:
                if 0.0001 <= volatility <= 0.0005:  # Volatilidad óptima para scalping
                    score += 0.8
                    reasons.append(f"✅ Volatilidad óptima: {volatility:.6f}")
                elif volatility > 0.0005:
                    score += 0.4
                    reasons.append(f"⚠️ Alta volatilidad: {volatility:.6f}")
                else:
                    score += 0.2
                    reasons.append(f"⚠️ Baja volatilidad: {volatility:.6f}")
                factors += 1
            
            # Spread
            spread = market_conditions.get('spread', 0)
            if spread:
                if spread <= 1.5:
                    score += 0.8
                    reasons.append(f"✅ Spread favorable: {spread:.1f} pips")
                elif spread <= 3.0:
                    score += 0.4
                    reasons.append(f"⚠️ Spread moderado: {spread:.1f} pips")
                else:
                    score += 0.1
                    reasons.append(f"❌ Spread alto: {spread:.1f} pips")
                factors += 1
            
            # Fuerza de tendencia
            trend_strength = market_conditions.get('trend_strength', 0)
            if trend_strength:
                if trend_strength >= 0.7:
                    score += 0.7
                    reasons.append(f"✅ Tendencia fuerte: {trend_strength:.2f}")
                elif trend_strength >= 0.4:
                    score += 0.4
                    reasons.append(f"📊 Tendencia moderada: {trend_strength:.2f}")
                else:
                    score += 0.2
                    reasons.append(f"⚠️ Tendencia débil: {trend_strength:.2f}")
                factors += 1
            
            return min(score / max(factors, 1), 1.0) if factors > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error analizando condiciones de mercado: {e}")
            reasons.append(f"❌ Error en condiciones de mercado: {str(e)}")
            return 0.0
    
    def _analyze_risk_reward(self, df, current_price: float, reasons: list) -> float:
        """Analiza los parámetros de riesgo/recompensa"""
        try:
            if len(df) < 3:
                reasons.append("❌ Datos insuficientes para análisis R:R")
                return 0.0
            
            # Calcular ATR para stop loss dinámico
            if 'atr' in df.columns:
                atr = df['atr'].iloc[-1]
                
                # Stop loss basado en ATR (1.5x ATR)
                sl_distance = atr * 1.5
                
                # Take profit basado en ratio 2:1 mínimo
                tp_distance = sl_distance * 2.0
                
                # Calcular ratio riesgo/recompensa
                risk_reward_ratio = tp_distance / sl_distance
                
                if risk_reward_ratio >= 2.0:
                    score = 0.8
                    reasons.append(f"✅ Excelente R:R ratio: {risk_reward_ratio:.1f}:1")
                elif risk_reward_ratio >= 1.5:
                    score = 0.6
                    reasons.append(f"✅ Buen R:R ratio: {risk_reward_ratio:.1f}:1")
                else:
                    score = 0.3
                    reasons.append(f"⚠️ R:R ratio bajo: {risk_reward_ratio:.1f}:1")
                
                # Verificar que el stop loss no sea demasiado grande
                sl_usd = sl_distance * 100000 * 0.01  # Aproximación para EURUSD
                if sl_usd <= 2.5:  # Dentro del límite de $2.50
                    score += 0.2
                    reasons.append(f"✅ Stop loss apropiado: ${sl_usd:.2f}")
                else:
                    score -= 0.1
                    reasons.append(f"⚠️ Stop loss elevado: ${sl_usd:.2f}")
                
                return min(score, 1.0)
            else:
                reasons.append("⚠️ Sin datos de ATR para análisis R:R")
                return 0.4
                
        except Exception as e:
            self.logger.error(f"Error en análisis riesgo/recompensa: {e}")
            reasons.append(f"❌ Error en análisis R:R: {str(e)}")
            return 0.0
    
    def _calculate_signal_strength(self, df, reasons: list) -> float:
        """Calcula la fuerza de la señal de trading"""
        try:
            if len(df) < 5:
                reasons.append("❌ Datos insuficientes para señal")
                return 0.0
            
            strength = 0.0
            
            # Momentum
            if 'close' in df.columns:
                recent_closes = df['close'].tail(5)
                momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
                
                if abs(momentum) > 0.001:  # Momentum fuerte
                    strength += 0.4
                    direction = "alcista" if momentum > 0 else "bajista"
                    reasons.append(f"✅ Momentum {direction} fuerte: {momentum:.4f}")
                elif abs(momentum) > 0.0005:  # Momentum moderado
                    strength += 0.2
                    direction = "alcista" if momentum > 0 else "bajista"
                    reasons.append(f"📊 Momentum {direction} moderado: {momentum:.4f}")
            
            # Consistencia de la tendencia
            if 'ema_fast' in df.columns:
                ema_trend = df['ema_fast'].tail(3)
                if all(ema_trend.iloc[i] > ema_trend.iloc[i-1] for i in range(1, len(ema_trend))):
                    strength += 0.3
                    reasons.append("✅ Tendencia alcista consistente")
                elif all(ema_trend.iloc[i] < ema_trend.iloc[i-1] for i in range(1, len(ema_trend))):
                    strength += 0.3
                    reasons.append("✅ Tendencia bajista consistente")
            
            # Convergencia de indicadores
            convergence_count = 0
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                if rsi < 35 or rsi > 65:  # RSI en zona de acción
                    convergence_count += 1
            
            if 'ema_fast' in df.columns and 'ema_slow' in df.columns:
                if abs(df['ema_fast'].iloc[-1] - df['ema_slow'].iloc[-1]) > 0.0001:
                    convergence_count += 1
            
            if convergence_count >= 2:
                strength += 0.3
                reasons.append("✅ Convergencia de indicadores")
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculando fuerza de señal: {e}")
            reasons.append(f"❌ Error en fuerza de señal: {str(e)}")
            return 0.0
    
    def _determine_recommendation(self, overall_score: float, technical_score: float, 
                                signal_strength: float, risk_reward_score: float, reasons: list) -> Tuple[str, float]:
        """Determina la recomendación final y confianza (gating más permisivo y robusto)"""
        try:
            # Umbrales de decisión más adaptativos
            base_overall = 0.58
            base_technical = 0.50
            base_signal = 0.45
            base_rr = 0.60  # ≈ R:R >= 1.5 según normalización de _analyze_risk_reward
            
            # Confianza derivada
            confidence = min(
                max(0.0, 0.6 * signal_strength + 0.4 * technical_score + (0.05 if risk_reward_score >= 0.7 else 0.0)),
                1.0
            )
            
            if (overall_score >= base_overall and
                technical_score >= base_technical and
                signal_strength >= base_signal and
                risk_reward_score >= base_rr):
                
                # Determinar dirección basada en indicadores (reasons)
                buy_signals = sum(1 for reason in reasons if any(word in reason.lower() 
                                for word in ['alcista', 'sobreventa', 'cruce alcista', 'inferior']))
                sell_signals = sum(1 for reason in reasons if any(word in reason.lower() 
                                 for word in ['bajista', 'sobrecompra', 'cruce bajista', 'superior']))
                
                if buy_signals > sell_signals:
                    return "BUY", min(max(confidence, overall_score + 0.1), 1.0)
                elif sell_signals > buy_signals:
                    return "SELL", min(max(confidence, overall_score + 0.1), 1.0)
                else:
                    # Si no hay dirección clara, preferir espera
                    return "HOLD", confidence * 0.9
            
            elif overall_score >= 0.5:
                reasons.append("⚠️ Oportunidad moderada - Esperar mejor momento")
                return "HOLD", confidence * 0.8
            
            else:
                reasons.append("❌ Condiciones no favorables para trading")
                return "HOLD", confidence * 0.6
                
        except Exception as e:
            self.logger.error(f"Error determinando recomendación: {e}")
            return "HOLD", 0.0
    
    def get_risk_summary(self, current_balance: Optional[float] = None) -> Dict:
        """
        Obtiene un resumen completo del estado de riesgo

        Args:
            current_balance: Balance actual opcional para override (útil en pruebas/offline)

        Returns:
            Dict: Resumen de riesgo con métricas clave
        """
        try:
            total_profit = 0
            active_positions_count = 0

            # Obtener información de cuenta si no se suministra balance
            if current_balance is None:
                account_info = mt5.account_info()
                if account_info:
                    current_balance = account_info.balance
                    total_profit = account_info.profit
                else:
                    # Fallback en modo offline
                    current_balance = 0.0
                    total_profit = 0.0

            # Contar posiciones activas
            positions = mt5.positions_get()
            if positions:
                active_positions_count = len(positions)

            # Calcular drawdown diario con protección de divisor
            start_balance = self.daily_stats.get('daily_start_balance', 0.0)
            daily_pnl = current_balance - start_balance
            if start_balance > 0 and daily_pnl < 0:
                daily_drawdown_pct = abs(daily_pnl) / start_balance
            else:
                daily_drawdown_pct = 0.0

            # Calcular capital efectivo (máximo $50)
            effective_capital = min(current_balance, 50.0)

            return {
                'timestamp': datetime.now().isoformat(),
                'current_balance': current_balance,
                'effective_capital': effective_capital,
                'daily_pnl': daily_pnl,
                'daily_drawdown_pct': daily_drawdown_pct,
                'trades_today': self.daily_stats['trades_today'],
                'consecutive_losses': self.daily_stats['consecutive_losses'],
                'active_positions': active_positions_count,
                'total_profit': total_profit,
                'risk_alerts_today': len(self.daily_stats['risk_alerts']),
                'can_trade': self.can_open_trade(current_balance)[0],
                'max_risk_per_trade': effective_capital * self.capital_config['max_risk_per_trade'],
                'remaining_daily_trades': max(
                    0,
                    self.config.get('max_daily_trades', 30) - self.daily_stats['trades_today']
                )
            }

        except Exception as e:
            self.logger.error(f"Error generando resumen de riesgo: {e}")
            return {'error': str(e)}
    
    def update_after_trade(self, result: float, trade_type: str = ""):
        """
        Actualiza estadísticas después de una operación con análisis inteligente
        
        Args:
            result: Resultado de la operación en USD
            trade_type: Tipo de operación ("BUY", "SELL", etc.)
        """
        try:
            self.daily_stats['trades_today'] += 1
            self.daily_stats['last_trade_time'] = datetime.now()
            
            if result < 0:
                # Operación perdedora
                self.daily_stats['loss_today'] += abs(result)
                self.daily_stats['consecutive_losses'] += 1
                
                # Generar alerta si la pérdida supera $2
                if abs(result) > 2.0:
                    self._add_risk_alert(f"Pérdida elevada: ${abs(result):.2f} en operación {trade_type}", "WARNING")
                
                self.logger.warning(f"Operación perdedora: ${result:.2f} - Pérdidas consecutivas: {self.daily_stats['consecutive_losses']}")
                
            else:
                # Operación ganadora
                self.daily_stats['profit_today'] += result
                self.daily_stats['consecutive_losses'] = 0  # Resetear contador
                
                self.logger.info(f"Operación ganadora: ${result:.2f}")
            
            # Actualizar drawdown máximo del día
            current_daily_pnl = self.daily_stats['profit_today'] - self.daily_stats['loss_today']
            if current_daily_pnl < self.daily_stats['max_drawdown_today']:
                self.daily_stats['max_drawdown_today'] = current_daily_pnl
            
            # Generar resumen cada 5 operaciones
            if self.daily_stats['trades_today'] % 5 == 0:
                self._log_performance_summary()
                
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas post-trade: {e}")
    
    def _log_performance_summary(self):
        """Registra un resumen de rendimiento"""
        try:
            total_pnl = self.daily_stats['profit_today'] - self.daily_stats['loss_today']
            win_rate = ((self.daily_stats['trades_today'] - self.daily_stats['consecutive_losses']) / 
                       self.daily_stats['trades_today'] * 100) if self.daily_stats['trades_today'] > 0 else 0
            
            self.logger.info("=" * 50)
            self.logger.info("RESUMEN DE RENDIMIENTO")
            self.logger.info("=" * 50)
            self.logger.info(f"Operaciones hoy: {self.daily_stats['trades_today']}")
            self.logger.info(f"P&L diario: ${total_pnl:.2f}")
            self.logger.info(f"Ganancias: ${self.daily_stats['profit_today']:.2f}")
            self.logger.info(f"Pérdidas: ${self.daily_stats['loss_today']:.2f}")
            self.logger.info(f"Tasa de acierto estimada: {win_rate:.1f}%")
            self.logger.info(f"Pérdidas consecutivas: {self.daily_stats['consecutive_losses']}")
            self.logger.info(f"Alertas de riesgo: {len(self.daily_stats['risk_alerts'])}")
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"Error generando resumen de rendimiento: {e}")

# Mantener compatibilidad con el código existente
class RiskManager(IntelligentRiskManager):
    """Alias para mantener compatibilidad con el código existente"""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger('RiskManager')
    
    def calculate_position_size(self, balance: float, stop_loss_pips: int, market_conditions: dict = None) -> float:
        """Método de compatibilidad que usa el nuevo sistema inteligente"""
        try:
            # Usar el nuevo sistema inteligente
            entry_price = 1.1000  # Precio de ejemplo para EURUSD
            position_params = self.calculate_intelligent_position(balance, entry_price, market_conditions)
            return position_params.volume
            
        except Exception as e:
            self.logger.error(f"Error en calculate_position_size: {e}")