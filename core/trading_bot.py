import logging
import time
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
from mt5_connector.connector import MT5Connector
from strategies.scalping_strategy import ScalpingStrategy
from core.risk_manager import RiskManager
import threading

class ScalpingBot:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('ScalpingBot')
        
        # Inicializar componentes
        self.mt5 = MT5Connector()
        self.strategy = ScalpingStrategy(config['strategy'])
        self.risk_manager = RiskManager(config['scalping'])
        
        # Estado del bot
        self.is_running = False
        self.account_balance = 0
        self.consecutive_holds = 0
        self.symbol = self.config['scalping'].get('symbol', 'EURUSD')
        self.timeframe = self.config['scalping'].get('timeframe', 'M1')
        # Tracking para logging condicional de oportunidades
        self._last_opportunity_recommendation = None
        self._last_opportunity_score = 0.0
        # Gestor de alertas
        try:
            from core.alert_manager import AlertManager
            self.alert_manager = AlertManager()
        except Exception:
            self.alert_manager = None
        
        # Estado del bot
        self.is_running = False
        self.account_balance = 0
        self.consecutive_holds = 0
        self.symbol = self.config['scalping'].get('symbol', 'EURUSD')
        self.timeframe = self.config['scalping'].get('timeframe', 'M1')
        # Tracking para logging condicional de oportunidades
        self._last_opportunity_recommendation = None
        self._last_opportunity_score = 0.0

    def initialize(self):
        """Inicializa el bot"""
        self.logger.info("Inicializando Scalping Bot...")
        
        if not self.mt5.connect():
            return False
        # Seleccionar símbolo objetivo para operar y reducir carga MT5
        self.mt5.ensure_symbol(self.symbol)
        
        # Obtener balance de la cuenta
        account_info = self.mt5.get_account_info()
        if account_info:
            self.account_balance = account_info.balance
            self.risk_manager.initialize_daily_stats(self.account_balance)
            self.logger.info(f"Balance inicial: ${self.account_balance:.2f}")
        else:
            self.logger.warning("No se pudo obtener balance, usando simulacion")
            self.account_balance = 50.0
            self.risk_manager.initialize_daily_stats(self.account_balance)
        
        self.logger.info("Scalping Bot inicializado exitosamente")
        return True
    
    def is_trading_hours(self):
        """Verifica si es horario de trading"""
        now = datetime.now()
        current_hour = now.hour
        
        start = self.config['scalping']['trading_hours']['start']
        end = self.config['scalping']['trading_hours']['end']
        
        in_hours = start <= current_hour < end
        
        if not in_hours:
            self.logger.info(f"Fuera de horario: {current_hour}:00 (Horario: {start}:00-{end}:00 GMT)")
        
        return in_hours
    
    def check_market_conditions(self):
        """Verifica condiciones de mercado con lógica adaptativa"""
        symbol = self.config['scalping']['symbol']
        spread = self.mt5.get_current_spread(symbol)
        
        # Obtener datos para análisis de volatilidad
        df = self.mt5.get_market_data(symbol, 'M1', 20)
        if df is None or len(df) < 10:
            self.logger.warning("Datos insuficientes para análisis de mercado")
            return False
        
        # Calcular volatilidad actual
        df['returns'] = df['close'].pct_change()
        current_volatility = df['returns'].std()
        
        # Determinar condiciones de mercado
        high_vol_threshold = self.config['scalping'].get('high_volatility_threshold', 0.001)
        low_vol_threshold = self.config['scalping'].get('low_volatility_threshold', 0.0001)
        
        is_high_volatility = current_volatility > high_vol_threshold
        is_low_volatility = current_volatility < low_vol_threshold
        
        # Ajustar límites dinámicamente
        if is_high_volatility:
            max_spread = self.config['scalping'].get('max_spread_dynamic', self.config['scalping']['max_spread'] * 2)
            min_volatility = self.config['scalping']['min_volatility'] * 0.7
            self.logger.info(f"Mercado de alta volatilidad detectado - Spread máximo: {max_spread} pips")
        elif is_low_volatility:
            max_spread = self.config['scalping']['max_spread']
            min_volatility = self.config['scalping'].get('min_volatility_adaptive', self.config['scalping']['min_volatility'] * 0.5)
            self.logger.info(f"Mercado de baja volatilidad - Umbral reducido: {min_volatility:.6f}")
        else:
            max_spread = self.config['scalping']['max_spread']
            min_volatility = self.config['scalping']['min_volatility']
        
        # Verificar spread con límites adaptativos
        if spread > max_spread:
            self.logger.warning(f"Spread muy alto: {spread:.1f} pips (Limite: {max_spread} pips)")
            return False
        
        # Verificar volatilidad con umbral adaptativo
        if current_volatility < min_volatility:
            self.logger.warning(f"Volatilidad muy baja: {current_volatility:.6f}")
            return False
        
        self.logger.info(f"Condiciones de mercado OK - Spread: {spread:.1f} pips, Volatilidad: {current_volatility:.6f}")
        return True
    
    def run_trading_cycle(self):
        """Ejecuta un ciclo de trading basado en oportunidades de mercado - OPTIMIZADO"""
        try:
            # Verificar condiciones básicas
            if not self.is_trading_hours():
                time.sleep(60)
                return True

            # Verificar con el nuevo sistema inteligente de riesgo
            can_trade, reason = self.risk_manager.can_open_trade(self.account_balance)
            if not can_trade:
                self.logger.info(f"No se puede operar: {reason}")
                time.sleep(60)
                return True

            # Obtener datos de mercado
            df = self.mt5.get_market_data(
                self.config['scalping']['symbol'],
                self.config['scalping']['timeframe'],
                50
            )

            if df is None or len(df) < 21:
                self.logger.warning("Datos insuficientes")
                time.sleep(10)
                return True

            # Calcular indicadores
            df = self.strategy.calculate_indicators(df)

            if df is None or len(df) < 5:
                self.logger.warning("No hay suficientes datos después de calcular indicadores")
                return True

            # Obtener precio actual y condiciones de mercado
            symbol = self.config['scalping']['symbol']
            current_price = self.mt5.get_current_price(symbol)
            # Generar señal con confirmaciones avanzadas
            signal, confidence = self.strategy.generate_signal(df)
            # Alertas en tiempo real para señales fuertes
            if self.alert_manager and signal in ['BUY', 'SELL'] and confidence >= self.config['strategy'].get('min_confidence', 0.7):
                current = df.iloc[-1]
                context = {
                    'ema_fast': float(current.get('ema_fast', 0.0)),
                    'ema_slow': float(current.get('ema_slow', 0.0)),
                    'rsi': float(current.get('rsi', 50.0)),
                    'macd_histogram': float(current.get('macd_histogram', 0.0)),
                    'mfi': float(current.get('mfi', 50.0)),
                    'near_support': bool(current.get('near_support', False)),
                    'near_resistance': bool(current.get('near_resistance', False)),
                    'fib_retracement': float(current.get('fib_retracement', 0.5))
                }
                self.alert_manager.send_signal_alert(self.symbol, self.timeframe, signal, confidence, context)
            if not current_price:
                self.logger.warning("No se pudo obtener precio actual")
                return True

            # Preparar condiciones de mercado
            market_conditions = {
                'volatility': df['close'].pct_change().std() if 'close' in df.columns else 0.0001,
                'spread': self.mt5.get_current_spread(symbol),
                'trend_strength': 0.5  # Valor por defecto
            }

            # NUEVA LÓGICA: Evaluar oportunidad de mercado
            opportunity = self.risk_manager.evaluate_market_opportunity(df, current_price, market_conditions)
            
            # Registrar evaluación
            self.logger.info("=" * 60)
            self.logger.info("🔍 EVALUACIÓN DE OPORTUNIDAD DE MERCADO")
            self.logger.info("=" * 60)
            
            # Decidir si ejecutar operación basado en la oportunidad
            if opportunity.recommendation in ["BUY", "SELL"] and opportunity.confidence >= 0.6:
                self.logger.info(f"🎯 OPORTUNIDAD DETECTADA: {opportunity.recommendation}")
                self.logger.info(f"📊 Confianza: {opportunity.confidence:.2f}")
                self.logger.info(f"🏆 Puntuación general: {opportunity.overall_score:.2f}")
                
                # Mostrar razones principales
                for reason in opportunity.reasons[:5]:  # Top 5 razones
                    self.logger.info(f"   {reason}")
                
                # Ejecutar la operación
                self.execute_opportunity_based_trade(opportunity.recommendation, df, current_price, market_conditions)
                
            elif opportunity.overall_score >= 0.4:
                self.logger.info(f"⏳ Oportunidad moderada detectada (Score: {opportunity.overall_score:.2f})")
                self.logger.info("📊 Esperando mejores condiciones...")
                
                # Mostrar las principales razones de espera
                for reason in opportunity.reasons[-3:]:  # Últimas 3 razones (problemas)
                    self.logger.info(f"   {reason}")
                    
            else:
                self.logger.info(f"❌ Sin oportunidades claras (Score: {opportunity.overall_score:.2f})")
                self.logger.info("🔄 Continuando monitoreo...")

            # Pausa adaptativa basada en la calidad de la oportunidad
            if opportunity.overall_score >= 0.6:
                time.sleep(5)  # Pausa corta si hay buenas oportunidades
            elif opportunity.overall_score >= 0.3:
                time.sleep(15)  # Pausa media
            else:
                time.sleep(30)  # Pausa larga si no hay oportunidades

            return True

        except Exception as e:
            self.logger.error(f"Error en ciclo de trading: {e}")
            return False
            
    def execute_opportunity_based_trade(self, signal: str, df, current_price: float, market_conditions: dict):
        """
        Ejecuta una operación basada en la evaluación de oportunidad de mercado
        
        Args:
            signal: Señal de trading ("BUY" o "SELL")
            df: DataFrame con datos de mercado
            current_price: Precio actual
            market_conditions: Condiciones de mercado
        """
        try:
            symbol = self.config['scalping']['symbol']
            
            self.logger.info("=" * 60)
            self.logger.info(f"🚀 EJECUTANDO OPERACIÓN BASADA EN OPORTUNIDAD: {signal}")
            self.logger.info("=" * 60)
            
            # Determinar precio de entrada y tipo de orden
            # Asegurar que current_price sea un float (bid/ask) y no una tupla
            if isinstance(current_price, tuple):
                current_bid, current_ask = current_price
            else:
                current_bid = current_price
                current_ask = current_price

            if signal == "BUY":
                entry_price = current_ask
                order_type = mt5.ORDER_TYPE_BUY
                self.logger.info(f"📈 Operación de COMPRA en {entry_price:.5f}")
            else:  # SELL
                entry_price = current_bid
                order_type = mt5.ORDER_TYPE_SELL
                self.logger.info(f"📉 Operación de VENTA en {entry_price:.5f}")

            # Usar el sistema inteligente para calcular parámetros
            position_params = self.risk_manager.calculate_intelligent_position(
                self.account_balance, entry_price, market_conditions
            )

            # Aplicar los parámetros calculados inteligentemente
            volume = position_params.volume
            stop_loss = position_params.stop_loss
            take_profit = position_params.take_profit

            # Verificar posiciones existentes
            open_positions = self.mt5.get_open_positions(symbol)
            max_positions = self.config['scalping'].get('max_simultaneous_trades', 3)
            
            if len(open_positions) >= max_positions:
                self.logger.warning(f"⚠️ Máximo de posiciones simultáneas alcanzado ({len(open_positions)}/{max_positions})")
                return

            # Mostrar información detallada de la operación
            self.logger.info(f"💰 Volumen: {volume} lotes")
            self.logger.info(f"🛑 Stop Loss: {stop_loss:.5f} (${position_params.stop_loss_usd:.2f})")
            self.logger.info(f"🎯 Take Profit: {take_profit:.5f} (${position_params.take_profit_usd:.2f})")
            self.logger.info(f"⚖️ Ratio R:R: {position_params.risk_reward_ratio:.2f}:1")
            self.logger.info(f"🎲 Confianza: {position_params.confidence_score:.2f}")

            # Ejecutar operación
            result = self.mt5.execute_trade(symbol, order_type, volume, stop_loss, take_profit)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info("=" * 60)
                self.logger.info("✅ OPERACIÓN EJECUTADA EXITOSAMENTE")
                self.logger.info("=" * 60)
                self.logger.info(f"🎫 Ticket: {result.order}")
                self.logger.info(f"📊 Gestión inteligente aplicada")
                self.logger.info(f"💡 Sistema de oportunidades: ACTIVO")

                # Inicializar seguimiento con resultado 0 (pendiente)
                self.risk_manager.update_after_trade(0, signal)

                # Actualizar balance
                self.update_account_balance()

                # Iniciar monitoreo continuo de la posición
                if hasattr(self.risk_manager, 'monitor_active_positions'):
                    threading.Thread(target=self._monitor_position_thread, daemon=True).start()

                # Resetear contador de holds consecutivos (ya no es necesario)
                self.consecutive_holds = 0

            else:
                error_code = result.retcode if result else 'Unknown'
                self.logger.error(f"❌ Error ejecutando operación: {error_code}")
                self.logger.error("🔄 Continuando búsqueda de oportunidades...")

        except Exception as e:
            self.logger.error(f"❌ Error en ejecución basada en oportunidad: {e}")

    def check_market_conditions(self):
        """Verifica condiciones de mercado con evaluación inteligente"""
        try:
            symbol = self.config['scalping']['symbol']
            spread = self.mt5.get_current_spread(symbol)
            
            # Obtener datos para análisis
            df = self.mt5.get_market_data(symbol, 'M1', 20)
            if df is None or len(df) < 10:
                self.logger.warning("⚠️ Datos insuficientes para análisis de mercado")
                return False

            # Calcular volatilidad actual
            df['returns'] = df['close'].pct_change()
            current_volatility = df['returns'].std()

            # Condiciones de mercado más flexibles para el sistema de oportunidades
            market_conditions = {
                'spread': spread,
                'volatility': current_volatility,
                'data_quality': len(df)
            }

            # Evaluación inteligente de condiciones
            if spread > self.config['scalping'].get('max_spread', 3.0) * 1.5:  # Más flexible
                self.logger.warning(f"⚠️ Spread muy elevado: {spread:.1f} pips")
                return False

            if current_volatility < 0.00005:  # Umbral muy bajo
                self.logger.warning(f"⚠️ Mercado extremadamente quieto: {current_volatility:.6f}")
                return False

            # Condiciones aceptables
            self.logger.debug(f"✅ Condiciones básicas OK - Spread: {spread:.1f}p, Vol: {current_volatility:.6f}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error verificando condiciones de mercado: {e}")
            return False
            confidence_threshold = 0.55  # Reducido de 0.7 a 0.55
            
            if signal != "HOLD" and confidence >= confidence_threshold:
                self.logger.info(f"Senal generada: {signal} (Confianza: {confidence:.2f})")
                self.consecutive_holds = 0
                
                # Ejecutar operacion
                self.execute_trading_signal(signal, df)
            else:
                self.consecutive_holds += 1
                self.logger.info(f"Sin senal valida: {signal} (Confianza: {confidence:.2f})")
                self.logger.info(f"Holds consecutivos: {self.consecutive_holds}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error en ciclo de trading: {e}")
            return False
    
    def execute_trading_signal(self, signal, df):
        """Ejecuta una señal de trading"""
        symbol = self.config['scalping']['symbol']
        current_bid, current_ask = self.mt5.get_current_price(symbol)
        
        if current_bid is None or current_ask is None:
            self.logger.error("No se pudo obtener precio actual")
            return
        
        if signal == "BUY":
            entry_price = current_ask
            order_type = mt5.ORDER_TYPE_BUY
        else:  # SELL
            entry_price = current_bid
            order_type = mt5.ORDER_TYPE_SELL
        
        # Calcular SL y TP
        stop_loss, take_profit = self.strategy.calculate_dynamic_exits(df, signal, entry_price)
        
        # Calcular parámetros de posición inteligente
        market_conditions = {
            'volatility': df['returns'].std() if 'returns' in df.columns else 0.0001,
            'spread': self.mt5.get_current_spread(symbol),
            'trend_strength': confidence
        }
        
        # Usar el nuevo sistema inteligente de gestión de capital
        position_params = self.risk_manager.calculate_intelligent_position(
            self.account_balance, entry_price, market_conditions
        )
        
        # Aplicar los parámetros calculados inteligentemente
        volume = position_params.volume
        stop_loss = position_params.stop_loss
        take_profit = position_params.take_profit
        
        # Calcular pips para logging
        stop_loss_pips = abs(entry_price - stop_loss) * 10000
        
        # Verificar posiciones existentes
        open_positions = self.mt5.get_open_positions(symbol)
        if len(open_positions) >= self.config['scalping']['max_simultaneous_trades']:
            self.logger.info("Maximo de posiciones simultaneas alcanzado")
            return
        
        # Mostrar informacion de la operacion
        self.logger.info(f"Operacion: {signal}")
        self.logger.info(f"Entry: {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
        self.logger.info(f"Volumen: {volume} lotes, Risk: {stop_loss_pips:.1f} pips")
        
        # Ejecutar operacion
        result = self.mt5.execute_trade(symbol, order_type, volume, stop_loss, take_profit)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Operación ejecutada: {signal} {volume} lotes")
            self.logger.info(f"Gestión inteligente aplicada - SL: ${position_params.stop_loss_usd:.2f}, TP: ${position_params.take_profit_usd:.2f}")
            
            # Inicializar seguimiento de la operación con resultado 0 (pendiente)
            self.risk_manager.update_after_trade(0, signal)
            
            # Actualizar balance
            self.update_account_balance()
            
            # Iniciar monitoreo continuo de la posición
            if hasattr(self.risk_manager, 'monitor_active_positions'):
                threading.Thread(target=self._monitor_position_thread, daemon=True).start()
                
        else:
            error_code = result.retcode if result else 'Unknown'
            self.logger.error(f"Error ejecutando operación: {error_code}")
    
    def _monitor_position_thread(self):
        """Hilo para monitoreo continuo de posiciones activas"""
        try:
            while self.is_running:
                self.risk_manager.monitor_active_positions()
                time.sleep(30)  # Monitoreo cada 30 segundos
        except Exception as e:
            self.logger.error(f"Error en monitoreo de posiciones: {e}")
    
    def update_account_balance(self):
        """Actualiza el balance de la cuenta y verifica posiciones cerradas"""
        try:
            account_info = self.mt5.get_account_info()
            if account_info:
                old_balance = self.account_balance
                new_balance = account_info.balance
                
                if new_balance != old_balance:
                    # Calcular resultado de la operación
                    trade_result = new_balance - old_balance
                    
                    # Actualizar balance
                    self.account_balance = new_balance
                    
                    # Actualizar estadísticas del risk manager si hubo cambio significativo
                    if abs(trade_result) > 0.01:  # Cambio mayor a 1 centavo
                        self.risk_manager.update_after_trade(trade_result)
                        
                        if trade_result > 0:
                            self.logger.info(f"✅ Operación cerrada con ganancia: ${trade_result:.2f}")
                        else:
                            self.logger.warning(f"❌ Operación cerrada con pérdida: ${trade_result:.2f}")
                    
                    self.logger.info(f"Balance actualizado: ${self.account_balance:.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error actualizando balance: {e}")
    
    def run(self):
        # Banner de inicio
        self.logger.info("🚀 INICIANDO BOT CON SISTEMA DE OPORTUNIDADES INTELIGENTE")
        self.logger.info("=" * 70)
        self.logger.info("🎯 Características del Sistema:")
        self.logger.info("   • Evaluación continua de oportunidades de mercado")
        self.logger.info("   • Análisis técnico en tiempo real")
        self.logger.info("   • Condiciones de mercado adaptativas")
        self.logger.info("   • Parámetros de riesgo/recompensa dinámicos")
        self.logger.info("   • Eliminación de restricciones temporales fijas")
        self.logger.info("   • Gestión inteligente de capital")
        self.logger.info("=" * 70)

        self.is_running = True

        if not self.initialize():
            self.logger.error("Error inicializando el bot")
            return

        # Inicializar estadísticas diarias del risk manager
        self.risk_manager.initialize_daily_stats(self.account_balance)
        
        # Resumen inicial
        risk_summary = self.risk_manager.get_risk_summary(self.account_balance)
        self.logger.info("💰 CONFIGURACIÓN DE CAPITAL:")
        self.logger.info(f"   Capital efectivo: ${risk_summary.get('effective_capital', 50):.2f}")
        self.logger.info(f"   Riesgo máximo por operación: ${risk_summary.get('max_risk_per_trade', 1):.2f}")
        self.logger.info(f"   Operaciones restantes hoy: {risk_summary.get('remaining_daily_trades', 30)}")
        self.logger.info("=" * 70)

        cycle_count = 0
        max_cycles = 2000
        consecutive_errors = 0
        max_errors = 5
        
        try:
            while self.is_running and cycle_count < max_cycles:
                cycle_count += 1
                
                try:
                    if cycle_count % 10 == 1:
                        self.logger.info(f"🔄 Ciclo #{cycle_count} - Buscando oportunidades de mercado...")
                    
                    success = self.run_trading_cycle()
                    
                    if success:
                        consecutive_errors = 0
                        if cycle_count % 20 == 0:
                            risk_summary = self.risk_manager.get_risk_summary(self.account_balance)
                            self.logger.info("=" * 50)
                            self.logger.info(f"📊 RESUMEN CICLO #{cycle_count}")
                            self.logger.info(f"💰 Balance: ${self.account_balance:.2f}")
                            self.logger.info(f"📈 Operaciones hoy: {risk_summary.get('trades_today', 0)}")
                            self.logger.info(f"🎯 P&L estimado: ${risk_summary.get('daily_pnl', 0):.2f}")
                            self.logger.info(f"⚡ Sistema de oportunidades: ACTIVO")
                            self.logger.info("=" * 50)
                    else:
                        consecutive_errors += 1
                        self.logger.warning(f"⚠️ Error en ciclo #{cycle_count} (Errores consecutivos: {consecutive_errors})")
                        
                    if consecutive_errors >= max_errors:
                        self.logger.error(f"❌ Demasiados errores consecutivos ({consecutive_errors}). Deteniendo bot.")
                        break
                        
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    self.logger.info("⏹️ Interrupción manual detectada")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(f"❌ Error en ciclo #{cycle_count}: {e}")
                    if consecutive_errors >= max_errors:
                        self.logger.error("❌ Límite de errores alcanzado. Deteniendo sistema.")
                        break
                    time.sleep(10)
                    
        except Exception as e:
            self.logger.error(f"❌ Error crítico en el bot: {e}")
        finally:
            self.logger.info("🏁 Finalizando sistema de oportunidades inteligente...")
            self.shutdown()

    def shutdown(self):
        """Cierra el bot"""
        self.is_running = False
        self.mt5.disconnect()
        self.logger.info("Scalping Bot detenido")