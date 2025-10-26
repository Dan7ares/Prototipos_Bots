#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests para MT5Connector - Pruebas automatizadas del conector MT5
---------------------------------------------------------------
Este módulo implementa pruebas unitarias para verificar el correcto
funcionamiento del conector MT5 y sus parámetros.

Autor: Trading Bot Team
Versión: 1.0
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt5_connector.connector import MT5Connector

class TestMT5Connector(unittest.TestCase):
    """
    Pruebas unitarias para la clase MT5Connector.
    
    Esta clase implementa pruebas para verificar:
    - Conexión y desconexión
    - Obtención de datos de mercado
    - Cálculo de spread
    - Ejecución de operaciones
    - Gestión de posiciones
    """
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        self.connector = MT5Connector()
        # Configurar mock para logger
        self.connector.logger = MagicMock()
    
    @patch('mt5_connector.connector.mt5')
    def test_connect(self, mock_mt5):
        """Prueba la conexión a MT5."""
        # Configurar mocks
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = True
        mock_mt5.symbols_get.return_value = [
            MagicMock(name='EURUSD'),
            MagicMock(name='GBPUSD')
        ]
        
        # Ejecutar método
        result = self.connector.connect()
        
        # Verificar resultados
        self.assertTrue(result)
        self.assertTrue(self.connector.connected)
        mock_mt5.initialize.assert_called_once()
        self.assertEqual(len(self.connector.available_symbols), 2)
    
    @patch('mt5_connector.connector.mt5')
    def test_connect_failure(self, mock_mt5):
        """Prueba fallo en la conexión a MT5."""
        # Configurar mocks
        mock_mt5.initialize.return_value = False
        
        # Ejecutar método
        result = self.connector.connect()
        
        # Verificar resultados
        self.assertFalse(result)
        self.assertFalse(self.connector.connected)
        mock_mt5.initialize.assert_called_once()
    
    @patch('mt5_connector.connector.mt5')
    def test_disconnect(self, mock_mt5):
        """Prueba la desconexión de MT5."""
        # Configurar estado inicial
        self.connector.connected = True
        
        # Ejecutar método
        self.connector.disconnect()
        
        # Verificar resultados
        self.assertFalse(self.connector.connected)
        mock_mt5.shutdown.assert_called_once()
    
    @patch('mt5_connector.connector.mt5')
    def test_check_symbol_available(self, mock_mt5):
        """Prueba la verificación de disponibilidad de símbolos."""
        # Configurar mocks
        self.connector.available_symbols = {'EURUSD', 'GBPUSD'}
        
        # Caso 1: Símbolo disponible directamente
        result = self.connector.check_symbol_available('EURUSD')
        self.assertEqual(result, 'EURUSD')
        
        # Caso 2: Símbolo no disponible directamente pero seleccionable
        mock_mt5.symbol_select.return_value = True
        result = self.connector.check_symbol_available('USDJPY')
        self.assertEqual(result, 'USDJPY')
        
        # Caso 3: Símbolo no disponible
        mock_mt5.symbol_select.return_value = False
        result = self.connector.check_symbol_available('UNKNOWN')
        self.assertIsNone(result)
    
    @patch('mt5_connector.connector.mt5')
    def test_get_market_data(self, mock_mt5):
        """Prueba la obtención de datos de mercado."""
        # Configurar mocks
        self.connector.connected = True
        self.connector.check_symbol_available = MagicMock(return_value='EURUSD')
        
        # Crear datos de prueba
        test_data = [
            {'time': 1609459200, 'open': 1.2200, 'high': 1.2250, 'low': 1.2180, 'close': 1.2220, 'tick_volume': 1000},
            {'time': 1609459260, 'open': 1.2220, 'high': 1.2240, 'low': 1.2210, 'close': 1.2230, 'tick_volume': 1200}
        ]
        mock_mt5.copy_rates_from_pos.return_value = test_data
        
        # Ejecutar método
        result = self.connector.get_market_data('EURUSD', 'M1', 2)
        
        # Verificar resultados
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertTrue('range' in result.columns)
        self.assertTrue('body' in result.columns)
        
        # Verificar cálculos adicionales usando assertAlmostEqual para valores flotantes
        self.assertAlmostEqual(result['range'].iloc[0], 0.007, places=6)  # high - low
        self.assertAlmostEqual(result['body'].iloc[0], 0.002, places=6)   # abs(close - open)
    
    @patch('mt5_connector.connector.mt5')
    def test_get_current_price(self, mock_mt5):
        """Prueba la obtención del precio actual."""
        # Configurar mocks
        self.connector.connected = True
        self.connector.check_symbol_available = MagicMock(return_value='EURUSD')
        
        # Crear tick de prueba
        mock_tick = MagicMock()
        mock_tick.bid = 1.2200
        mock_tick.ask = 1.2202
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Ejecutar método
        bid, ask = self.connector.get_current_price('EURUSD')
        
        # Verificar resultados
        self.assertEqual(bid, 1.2200)
        self.assertEqual(ask, 1.2202)
    
    @patch('mt5_connector.connector.mt5')
    def test_get_current_spread(self, mock_mt5):
        """Prueba el cálculo del spread actual."""
        # Configurar mocks
        self.connector.connected = True
        self.connector.check_symbol_available = MagicMock(return_value='EURUSD')
        
        # Crear mocks para symbol_info y tick
        mock_symbol_info = MagicMock()
        mock_symbol_info.digits = 5
        mock_mt5.symbol_info.return_value = mock_symbol_info
        
        mock_tick = MagicMock()
        mock_tick.bid = 1.22000
        mock_tick.ask = 1.22020
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Ejecutar método
        spread = self.connector.get_current_spread('EURUSD')
        
        # Verificar resultados
        self.assertAlmostEqual(spread, 2.0, places=6)  # (1.22020 - 1.22000) * 10^(5-1) = 2.0 pips
    
    @patch('mt5_connector.connector.mt5')
    def test_execute_trade(self, mock_mt5):
        """Prueba la ejecución de operaciones."""
        # Configurar mocks
        self.connector.connected = True
        self.connector.check_symbol_available = MagicMock(return_value='EURUSD')
        self.connector.get_current_price = MagicMock(return_value=(1.2200, 1.2202))
        
        # Crear resultado de orden de prueba
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_mt5.order_send.return_value = mock_result
        
        # Ejecutar método - Compra
        result = self.connector.execute_trade('EURUSD', mock_mt5.ORDER_TYPE_BUY, 0.01, 1.2180, 1.2220)
        
        # Verificar resultados
        self.assertIsNotNone(result)
        mock_mt5.order_send.assert_called_once()
        
        # Verificar parámetros de la orden
        args, kwargs = mock_mt5.order_send.call_args
        request = args[0]
        self.assertEqual(request['symbol'], 'EURUSD')
        self.assertEqual(request['volume'], 0.01)
        self.assertEqual(request['type'], mock_mt5.ORDER_TYPE_BUY)
        self.assertEqual(request['price'], 1.2202)  # Precio ask para compra
        self.assertEqual(request['sl'], 1.2180)
        self.assertEqual(request['tp'], 1.2220)
    
    @patch('mt5_connector.connector.mt5')
    def test_get_open_positions(self, mock_mt5):
        """Prueba la obtención de posiciones abiertas."""
        # Configurar mocks
        self.connector.connected = True
        self.connector.check_symbol_available = MagicMock(return_value='EURUSD')
        
        # Configurar constantes de MT5
        mock_mt5.POSITION_TYPE_BUY = 0
        mock_mt5.POSITION_TYPE_SELL = 1
        
        # Crear posición de prueba
        mock_position = MagicMock()
        mock_position.ticket = 12345
        mock_position.symbol = 'EURUSD'
        mock_position.type = mock_mt5.POSITION_TYPE_BUY  # BUY = 0
        mock_position.volume = 0.01
        mock_position.price_open = 1.2200
        mock_position.price_current = 1.2210
        mock_position.sl = 1.2180
        mock_position.tp = 1.2230
        mock_position.profit = 1.0
        mock_position.time = int(datetime.now().timestamp())
        
        mock_mt5.positions_get.return_value = [mock_position]
        
        # Ejecutar método
        positions = self.connector.get_open_positions('EURUSD')
        
        # Verificar resultados
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['ticket'], 12345)
        self.assertEqual(positions[0]['symbol'], 'EURUSD')
        self.assertEqual(positions[0]['type'], 'BUY')
        self.assertEqual(positions[0]['profit'], 1.0)
    
    @patch('mt5_connector.connector.mt5')
    def test_close_position(self, mock_mt5):
        """Prueba el cierre de posiciones."""
        # Configurar mocks
        self.connector.connected = True
        
        # Configurar constantes de MT5
        mock_mt5.POSITION_TYPE_BUY = 0
        mock_mt5.POSITION_TYPE_SELL = 1
        mock_mt5.ORDER_TYPE_BUY = 0
        mock_mt5.ORDER_TYPE_SELL = 1
        mock_mt5.TRADE_RETCODE_DONE = 10009
        
        # Crear posición de prueba
        mock_position = MagicMock()
        mock_position.symbol = 'EURUSD'
        mock_position.type = mock_mt5.POSITION_TYPE_BUY  # BUY = 0
        mock_position.volume = 0.01
        mock_position.ticket = 12345
        
        mock_mt5.positions_get.return_value = [mock_position]
        
        # Crear tick de prueba
        mock_tick = MagicMock()
        mock_tick.bid = 1.2210
        mock_tick.ask = 1.2212
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        # Crear resultado de orden de prueba
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_mt5.order_send.return_value = mock_result
        
        # Ejecutar método
        result = self.connector.close_position(12345)
        
        # Verificar resultados
        self.assertTrue(result)
        mock_mt5.order_send.assert_called_once()
        
        # Verificar parámetros de la orden de cierre
        args, kwargs = mock_mt5.order_send.call_args
        request = args[0]
        self.assertEqual(request['symbol'], 'EURUSD')
        self.assertEqual(request['volume'], 0.01)
        self.assertEqual(request['type'], mock_mt5.ORDER_TYPE_SELL)  # Venta para cerrar compra
        self.assertEqual(request['position'], 12345)

class TestGetOpenPositions(unittest.TestCase):
    def setUp(self):
        self.connector = MT5Connector()
        # Asume conexión mock si MT5 no está disponible; ajusta para entorno real
        self.connector.connect()
    
    def test_get_open_positions_success(self):
        positions = self.connector.get_open_positions()
        self.assertIsInstance(positions, list)
        # Agrega asserts específicos si hay posiciones mock
    
    def test_get_open_positions_error(self):
        # Simula error desconectando
        self.connector.disconnect()
        positions = self.connector.get_open_positions()
        self.assertEqual(positions, [])
    
    def test_get_open_positions_with_symbol(self):
        positions = self.connector.get_open_positions('EURUSD')
        self.assertIsInstance(positions, list)

if __name__ == '__main__':
    unittest.main()