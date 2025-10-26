import unittest
import sys
import os

# Ajuste temporal para imports desde directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.scalping_strategy import ScalpingStrategy

class TestScalpingStrategy(unittest.TestCase):
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Configuración mock para ScalpingStrategy
        self.config = {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'volume_factor': 2.0,
            'sl_atr_multiplier': 1.5,
            'tp_atr_multiplier': 2.0
        }
        
    def test_rsi_filter(self):
        """Prueba el filtro RSI de la estrategia."""
        try:
            strategy = ScalpingStrategy(self.config)
            
            # Test lógica RSI - valores de sobreventa
            self.assertTrue(hasattr(strategy, 'config'))
            self.assertEqual(strategy.config['rsi_oversold'], 30)
            self.assertEqual(strategy.config['rsi_overbought'], 70)
            
            # Verificar que la estrategia se inicializa correctamente
            self.assertIsNotNone(strategy.logger)
            self.assertEqual(strategy.last_signal, 'HOLD')
            
        except Exception as e:
            self.fail(f"Error en test_rsi_filter: {e}")
    
    def test_strategy_initialization(self):
        """Prueba la inicialización correcta de la estrategia."""
        try:
            strategy = ScalpingStrategy(self.config)
            
            # Verificar que todos los parámetros se configuraron correctamente
            self.assertEqual(strategy.config['ema_fast'], 9)
            self.assertEqual(strategy.config['ema_slow'], 21)
            self.assertEqual(strategy.config['rsi_period'], 14)
            
        except Exception as e:
            self.fail(f"Error en test_strategy_initialization: {e}")

if __name__ == '__main__':
    unittest.main()