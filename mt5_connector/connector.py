#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conector MetaTrader 5 - Módulo de Comunicación con MT5
------------------------------------------------------
Este módulo proporciona una interfaz unificada para interactuar con
la plataforma MetaTrader 5, gestionando conexiones, obtención de datos
y ejecución de operaciones.

Autor: Trading Bot Team
Versión: 2.0
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union

class MT5Connector:
    """
    Clase para gestionar la conexión y comunicación con MetaTrader 5.
    
    Esta clase proporciona métodos para:
    - Conectar y desconectar de MT5
    - Obtener datos de mercado
    - Ejecutar operaciones
    - Gestionar posiciones abiertas
    """
    
    # Mapeo de timeframes para reutilización
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1
    }
    
    def __init__(self):
        """Inicializa el conector MT5."""
        self.logger = logging.getLogger('MT5Connector')
        self.connected = False
        self.available_symbols = set()
    
    def connect(self) -> bool:
        """
        Establece conexión con MetaTrader 5.
        
        Returns:
            bool: True si la conexión fue exitosa, False en caso contrario
        """
        try:
            # Verificar si ya está conectado
            if self.connected:
                return True
                
            # Inicializar MT5
            if not mt5.initialize():
                self.logger.error(f"Error inicializando MT5: {mt5.last_error()}")
                return False
            
            # Verificar estado de conexión
            if not mt5.terminal_info():
                self.logger.error("No se pudo obtener información del terminal MT5")
                return False
                
            self.connected = True
            self.logger.info("Conectado a MT5 exitosamente")
            
            # Cargar símbolos disponibles
            self._load_available_symbols()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error conectando a MT5: {e}")
            return False
    
    def _load_available_symbols(self) -> None:
        """
        Carga la lista de símbolos disponibles en MT5.
        """
        try:
            symbols = mt5.symbols_get()
            self.available_symbols = {symbol.name for symbol in symbols}
            self.logger.info(f"Cargados {len(self.available_symbols)} símbolos disponibles")
        except Exception as e:
            self.logger.error(f"Error cargando símbolos: {e}")
    
    def check_symbol_available(self, symbol: str) -> str:
        """
        Verifica si un símbolo está disponible, intentando variantes si es necesario.
        
        Args:
            symbol (str): Símbolo a verificar
            
        Returns:
            str: Símbolo disponible o None si no se encuentra
        """
        # Verificar variantes comunes del símbolo
        variants = [symbol, f"{symbol}m", f"{symbol}.a", f"{symbol}.b", f"{symbol}.c"]
        
        for variant in variants:
            if variant in self.available_symbols:
                return variant
                
            # Intentar seleccionar el símbolo si no está en la lista cargada
            if mt5.symbol_select(variant, True):
                self.available_symbols.add(variant)
                return variant
        
        self.logger.error(f"No se pudo encontrar el símbolo {symbol} ni sus variantes")
        return None
    
    def disconnect(self) -> None:
        """
        Desconecta de MetaTrader 5.
        """
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("Desconectado de MT5")
        except Exception as e:
            self.logger.error(f"Error desconectando de MT5: {e}")
    
    def get_account_info(self) -> Optional[mt5.AccountInfo]:
        """
        Obtiene información de la cuenta de trading.
        
        Returns:
            mt5.AccountInfo: Información de la cuenta o None si hay error
        """
        try:
            if not self.connected and not self.connect():
                return None
                
            account_info = mt5.account_info()
            if not account_info:
                self.logger.error(f"Error obteniendo información de cuenta: {mt5.last_error()}")
                return None
                
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error obteniendo información de cuenta: {e}")
            return None
    
    def get_market_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos del mercado.
        
        Args:
            symbol (str): Símbolo a consultar
            timeframe (str): Timeframe ('M1', 'M5', etc.)
            count (int): Cantidad de velas a obtener
            
        Returns:
            pd.DataFrame: DataFrame con los datos o None si hay error
        """
        try:
            if not self.connected and not self.connect():
                return None
            
            # Verificar timeframe válido
            if timeframe not in self.TIMEFRAME_MAP:
                self.logger.error(f"Timeframe no válido: {timeframe}")
                return None
                
            # Obtener símbolo disponible
            valid_symbol = self.check_symbol_available(symbol)
            if not valid_symbol:
                return None
            
            # Obtener datos
            rates = mt5.copy_rates_from_pos(valid_symbol, self.TIMEFRAME_MAP[timeframe], 0, count)
            if rates is None or len(rates) == 0:
                self.logger.error(f"No se pudieron obtener datos para {valid_symbol}: {mt5.last_error()}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calcular campos adicionales útiles
            df['range'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de mercado: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtiene el precio actual (bid/ask) de un símbolo.
        
        Args:
            symbol (str): Símbolo a consultar
            
        Returns:
            tuple: (bid, ask) o (None, None) si hay error
        """
        try:
            if not self.connected and not self.connect():
                return None, None
            
            # Obtener símbolo disponible
            valid_symbol = self.check_symbol_available(symbol)
            if not valid_symbol:
                return None, None
                
            # Obtener tick
            tick = mt5.symbol_info_tick(valid_symbol)
            if not tick:
                self.logger.error(f"No se pudo obtener tick para {valid_symbol}: {mt5.last_error()}")
                return None, None
                
            return tick.bid, tick.ask
            
        except Exception as e:
            self.logger.error(f"Error obteniendo precio actual: {e}")
            return None, None
    
    def get_current_spread(self, symbol: str) -> float:
        """
        Obtiene el spread actual en pips con manejo robusto de símbolos alternativos.
        
        Args:
            symbol (str): Símbolo a consultar
            
        Returns:
            float: Spread en pips o valor conservador si hay error
        """
        try:
            if not self.connected and not self.connect():
                return 2.0  # Valor conservador por defecto
            
            # Obtener símbolo disponible con variantes
            valid_symbol = self.check_symbol_available(symbol)
            if not valid_symbol:
                # Intentar con variaciones adicionales
                alternative_symbols = [
                    f"{symbol}m",  # Micro lotes
                    f"{symbol}.m", # Otra variación
                    symbol.replace("USD", "USDm") if "USD" in symbol else symbol,
                    f"{symbol}.a",  # Variante A
                    f"{symbol}.b"   # Variante B
                ]
                
                for alt_symbol in alternative_symbols:
                    if mt5.symbol_select(alt_symbol, True):
                        valid_symbol = alt_symbol
                        self.logger.info(f"Usando símbolo alternativo: {alt_symbol}")
                        self.available_symbols.add(alt_symbol)
                        break
                
                if not valid_symbol:
                    self.logger.warning(f"No se encontró símbolo válido para {symbol}")
                    return 2.0
                
            # Obtener información del símbolo
            symbol_info = mt5.symbol_info(valid_symbol)
            if not symbol_info:
                self.logger.error(f"No se pudo obtener información del símbolo {valid_symbol}")
                return 2.0
            
            # Obtener tick actual
            tick = mt5.symbol_info_tick(valid_symbol)
            if not tick:
                self.logger.error(f"No se pudo obtener tick para {valid_symbol}")
                return 2.0
            
            # Calcular spread en pips con precisión exacta
            spread = round((tick.ask - tick.bid) * (10 ** (symbol_info.digits - 1)), 1)
            
            # Validar spread razonable
            if spread < 0 or spread > 50:  # Spread fuera de rango normal
                self.logger.warning(f"Spread anormal detectado para {valid_symbol}: {spread} pips")
                return 2.0
            
            self.logger.info(f"Spread actual para {valid_symbol}: {spread:.1f} pips")
            
            return float(spread)
            
        except Exception as e:
            self.logger.error(f"Error calculando spread para {symbol}: {e}")
            return 2.0  # Valor conservador por defecto
    
    def execute_trade(self, symbol: str, order_type: int, volume: float, 
                     sl: Optional[float] = None, tp: Optional[float] = None) -> Optional[mt5.OrderSendResult]:
        """
        Ejecuta una operación de trading.
        
        Args:
            symbol (str): Símbolo a operar
            order_type (int): Tipo de orden (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL)
            volume (float): Volumen en lotes
            sl (float, optional): Nivel de stop loss
            tp (float, optional): Nivel de take profit
            
        Returns:
            mt5.OrderSendResult: Resultado de la orden o None si hay error
        """
        try:
            if not self.connected and not self.connect():
                return None
            
            # Obtener símbolo disponible
            valid_symbol = self.check_symbol_available(symbol)
            if not valid_symbol:
                return None
                
            # Obtener precio actual
            bid, ask = self.get_current_price(valid_symbol)
            if bid is None or ask is None:
                return None
                
            # Determinar precio según tipo de orden
            price = ask if order_type == mt5.ORDER_TYPE_BUY else bid
            
            # Preparar solicitud de orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": valid_symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "deviation": 10,  # Desviación máxima del precio
                "magic": 123456,  # ID mágico para identificar órdenes del bot
                "comment": "Scalping Bot v2.0",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Añadir SL y TP si se proporcionan
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            # Enviar orden
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Error ejecutando orden: {result.retcode}, {result.comment}")
                return None
                
            self.logger.info(f"Orden ejecutada: {valid_symbol}, {volume} lotes, Precio: {price}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error ejecutando operación: {e}")
            return None
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Obtiene las posiciones abiertas.
        
        Args:
            symbol (str, optional): Filtrar por símbolo
            
        Returns:
            list: Lista de posiciones abiertas
        """
        try:
            if not self.connected and not self.connect():
                return []
                
            # Obtener posiciones
            if symbol:
                valid_symbol = self.check_symbol_available(symbol)
                positions = mt5.positions_get(symbol=valid_symbol) if valid_symbol else []
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                self.logger.error(f"Error obteniendo posiciones: {mt5.last_error()}")
                return []
                
            # Convertir a lista de diccionarios
            positions_list = []
            for position in positions:
                positions_list.append({
                    'ticket': position.ticket,
                    'symbol': position.symbol,
                    'type': 'BUY' if position.type == mt5.POSITION_TYPE_BUY else 'SELL',
                    'volume': position.volume,
                    'open_price': position.price_open,
                    'current_price': position.price_current,
                    'sl': position.sl,
                    'tp': position.tp,
                    'profit': position.profit,
                    'time': datetime.fromtimestamp(position.time)
                })
                
            return positions_list
            
        except Exception as e:
            self.logger.error(f"Error obteniendo posiciones abiertas: {e}")
            return []
    
    def close_position(self, ticket: int) -> bool:
        """
        Cierra una posición abierta por su ticket.
        
        Args:
            ticket (int): Número de ticket de la posición
            
        Returns:
            bool: True si se cerró correctamente, False en caso contrario
        """
        try:
            if not self.connected and not self.connect():
                return False
                
            # Obtener la posición
            position = mt5.positions_get(ticket=ticket)
            if not position or len(position) == 0:
                self.logger.error(f"No se encontró la posición con ticket {ticket}")
                return False
                
            position = position[0]
            
            # Obtener tick actual
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                self.logger.error(f"No se pudo obtener tick para {position.symbol}")
                return False
                
            # Determinar tipo de orden para cerrar (contrario a la posición)
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            # Preparar solicitud de cierre
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Cierre - Scalping Bot v2.0",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Enviar orden de cierre
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Error cerrando posición: {result.retcode}, {result.comment}")
                return False
                
            self.logger.info(f"Posición cerrada: Ticket {ticket}, {position.symbol}, {position.volume} lotes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cerrando posición: {e}")
            return False

    def ensure_symbol(self, symbol: str) -> bool:
        valid_symbol = self.check_symbol_available(symbol)
        if not valid_symbol:
            self.logger.warning("No se pudo seleccionar el símbolo {}".format(symbol))
            return False
        if not mt5.symbol_select(valid_symbol, True):
            self.logger.warning("No se pudo seleccionar el símbolo {}".format(valid_symbol))
            return False
        self.logger.info("Símbolo seleccionado: {}".format(valid_symbol))
        try:
            self.symbols = [valid_symbol]
        except Exception:
            pass
        return True
    
        # Limita la lista de símbolos al objetivo para reducir overhead
        try:
            self.symbols = [symbol]
        except Exception:
            pass