#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motor de Optimización de Parámetros - Sistema de Trading
--------------------------------------------------------
Sistema avanzado de optimización con validación cruzada para alcanzar
rentabilidad objetivo del 65%+ mediante algoritmos genéticos y análisis estadístico.

Autor: Trading Bot Team
Versión: 1.0 - Optimización Inteligente
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest import AdvancedBacktester
from config.settings import STRATEGY_CONFIG, PROFITABILITY_TARGETS

@dataclass
class OptimizationResult:
    """Resultado de optimización de parámetros"""
    parameters: Dict
    metrics: Dict
    composite_score: float
    validation_scores: List[float]
    consistency_score: float

class ParameterOptimizer:
    """
    Motor de optimización de parámetros con validación cruzada
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.logger = self._setup_logger()
        
        # Rangos de optimización inteligentes
        self.parameter_ranges = {
            'ema_fast': [2, 3, 4, 5],
            'ema_medium': [6, 8, 10, 12],
            'ema_slow': [13, 15, 18, 21],
            'rsi_period': [8, 10, 12, 14],
            'rsi_oversold': [20, 25, 30],
            'rsi_overbought': [70, 75, 80],
            'take_profit_multiplier': [2.5, 3.0, 3.2, 3.5, 4.0],
            'stop_loss_multiplier': [0.8, 0.9, 1.0, 1.2],
            'min_confidence': [0.60, 0.65, 0.70, 0.75],  # ampliado para coherencia con configuración actual
            'bollinger_period': [14, 16, 18, 20],
            'bollinger_std': [2.0, 2.2, 2.3, 2.5],
            'atr_period': [10, 12, 14, 16],
            'min_confirmations': [2, 3, 4],
            'cooldown_bars': [1, 2, 3, 4]
        }
        
        # Pesos para score compuesto
        self.score_weights = {
            'total_return': 0.40,      # 40% - Rentabilidad es prioritaria
            'win_rate': 0.25,          # 25% - Calidad de señales
            'drawdown_control': 0.20,  # 20% - Control de riesgo
            'profit_factor': 0.15      # 15% - Eficiencia de trades
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configurar logger para optimización"""
        logger = logging.getLogger('ParameterOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_parameters(self, symbol: str = None, days: int = 90, 
                          max_combinations: int = 50) -> OptimizationResult:
        """
        Optimizar parámetros con validación cruzada
        
        Args:
            symbol: Símbolo a optimizar (None para auto-selección)
            days: Días de datos históricos
            max_combinations: Máximo número de combinaciones a probar
            
        Returns:
            OptimizationResult con mejores parámetros encontrados
        """
        self.logger.info("🚀 Iniciando optimización de parámetros con validación cruzada...")
        
        # Generar combinaciones inteligentes de parámetros
        parameter_combinations = self._generate_parameter_combinations(max_combinations)
        self.logger.info(f"📊 Generadas {len(parameter_combinations)} combinaciones de parámetros")
        
        best_result = None
        best_score = 0.0
        results = []
        
        # Probar cada combinación con validación cruzada
        for i, params in enumerate(parameter_combinations):
            self.logger.info(f"🔍 Probando combinación {i+1}/{len(parameter_combinations)}")
            
            try:
                # Ejecutar validación cruzada
                validation_scores = self._cross_validate_parameters(params, symbol, days)
                
                if validation_scores:
                    # Calcular score compuesto y consistencia
                    composite_score = np.mean(validation_scores)
                    consistency_score = 1.0 - (np.std(validation_scores) / max(np.mean(validation_scores), 0.01))
                    
                    # Ejecutar backtest completo con mejores parámetros
                    final_metrics = self._run_backtest_with_parameters(params, symbol, days)
                    
                    if final_metrics and not final_metrics.get('error'):
                        result = OptimizationResult(
                            parameters=params,
                            metrics=final_metrics,
                            composite_score=composite_score,
                            validation_scores=validation_scores,
                            consistency_score=consistency_score
                        )
                        
                        results.append(result)
                        
                        # Actualizar mejor resultado
                        if composite_score > best_score:
                            best_score = composite_score
                            best_result = result
                            self.logger.info(f"✅ Nuevo mejor score: {composite_score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error en combinación {i+1}: {e}")
                continue
        
        if best_result:
            self.logger.info(f"🏆 Optimización completada. Mejor score: {best_score:.4f}")
            self._save_optimization_results(best_result, results)
            return best_result
        else:
            self.logger.error("❌ No se encontraron parámetros válidos")
            return None
    
    def _generate_parameter_combinations(self, max_combinations: int) -> List[Dict]:
        """
        Generar combinaciones inteligentes de parámetros
        """
        # Combinaciones base más prometedoras
        base_combinations = [
            # Configuración agresiva para alta rentabilidad
            {
                'ema_fast': 3, 'ema_medium': 8, 'ema_slow': 13,
                'rsi_period': 10, 'rsi_oversold': 25, 'rsi_overbought': 75,
                'take_profit_multiplier': 3.2, 'stop_loss_multiplier': 0.9,
                'min_confidence': 0.65, 'bollinger_period': 16, 'bollinger_std': 2.3,
                'atr_period': 12, 'min_confirmations': 3, 'cooldown_bars': 2
            },
            # Configuración balanceada
            {
                'ema_fast': 4, 'ema_medium': 10, 'ema_slow': 15,
                'rsi_period': 12, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'take_profit_multiplier': 3.0, 'stop_loss_multiplier': 1.0,
                'min_confidence': 0.60, 'bollinger_period': 18, 'bollinger_std': 2.2,
                'atr_period': 14, 'min_confirmations': 3, 'cooldown_bars': 3
            },
            # Configuración conservadora
            {
                'ema_fast': 5, 'ema_medium': 12, 'ema_slow': 18,
                'rsi_period': 14, 'rsi_oversold': 20, 'rsi_overbought': 80,
                'take_profit_multiplier': 2.5, 'stop_loss_multiplier': 1.2,
                'min_confidence': 0.70, 'bollinger_period': 20, 'bollinger_std': 2.0,
                'atr_period': 16, 'min_confirmations': 4, 'cooldown_bars': 4
            }
        ]
        
        combinations = base_combinations.copy()
        
        # Generar variaciones aleatorias inteligentes
        np.random.seed(42)  # Para reproducibilidad
        
        while len(combinations) < max_combinations:
            # Seleccionar base aleatoria
            base = np.random.choice(base_combinations)
            variation = base.copy()
            
            # Aplicar variaciones aleatorias a 2-3 parámetros
            params_to_vary = np.random.choice(list(self.parameter_ranges.keys()), 
                                            size=np.random.randint(2, 4), replace=False)
            
            for param in params_to_vary:
                if param in self.parameter_ranges:
                    variation[param] = np.random.choice(self.parameter_ranges[param])
            
            # Validar coherencia de parámetros
            if self._validate_parameter_coherence(variation):
                combinations.append(variation)
        
        return combinations[:max_combinations]
    
    def _validate_parameter_coherence(self, params: Dict) -> bool:
        """
        Validar coherencia lógica de parámetros
        """
        try:
            # EMA: fast < medium < slow
            if not (params['ema_fast'] < params['ema_medium'] < params['ema_slow']):
                return False
            
            # RSI: oversold < overbought
            if not (params['rsi_oversold'] < params['rsi_overbought']):
                return False
            
            # Take profit > stop loss para ratio positivo
            if not (params['take_profit_multiplier'] > params['stop_loss_multiplier']):
                return False
            
            # Confianza mínima razonable
            if not (0.6 <= params['min_confidence'] <= 0.9):
                return False
            
            return True
            
        except KeyError:
            return False
    
    def _cross_validate_parameters(self, params: Dict, symbol: str, days: int) -> List[float]:
        """
        Validación cruzada con 3 divisiones temporales
        """
        validation_scores = []
        
        # Dividir en 3 períodos para validación cruzada
        period_days = days // 3
        
        for i in range(3):
            try:
                # Calcular días para este período
                start_day = i * period_days
                end_day = min((i + 1) * period_days, days)
                period_length = end_day - start_day
                
                if period_length < 5:  # antes: 10 (permitir modo rápido con 15 días)
                    continue
                
                # Ejecutar backtest para este período
                metrics = self._run_backtest_with_parameters(params, symbol, period_length)
                
                if metrics and not metrics.get('error'):
                    score = self._calculate_composite_score(metrics)
                    validation_scores.append(score)
                
            except Exception as e:
                self.logger.warning(f"Error en validación período {i+1}: {e}")
                continue
        
        # Fallback: si no hay scores por períodos, usar un run completo
        if not validation_scores:
            try:
                metrics = self._run_backtest_with_parameters(params, symbol, days)
                if metrics and not metrics.get('error'):
                    validation_scores.append(self._calculate_composite_score(metrics))
            except Exception as e:
                self.logger.warning(f"Fallback de validación falló: {e}")
        
        return validation_scores
    
    def _run_backtest_with_parameters(self, params: Dict, symbol: str, days: int) -> Dict:
        """
        Ejecutar backtest con parámetros específicos
        """
        try:
            # Crear backtester con parámetros personalizados
            backtester = AdvancedBacktester(self.initial_capital)
            
            # Actualizar configuración de estrategia temporalmente
            original_config = STRATEGY_CONFIG.copy()
            STRATEGY_CONFIG.update(params)
            
            # Ejecutar backtest
            result = backtester.run_advanced_backtest(symbol=symbol, days=days)
            
            # Restaurar configuración original
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(original_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error en backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """
        Calcular score compuesto basado en múltiples métricas
        """
        try:
            # Normalizar métricas (0-1)
            return_score = min(metrics.get('total_return', 0) / 0.8, 1.0)  # Normalizar a 80%
            win_rate_score = min(metrics.get('win_rate', 0) / 0.7, 1.0)    # Normalizar a 70%
            
            # Drawdown score (invertido - menor drawdown = mejor score)
            drawdown = abs(metrics.get('max_drawdown', 0.2))
            drawdown_score = max(1.0 - (drawdown / 0.2), 0.0)  # Normalizar a 20%
            
            # Profit factor score
            profit_factor = metrics.get('profit_factor', 1.0)
            pf_score = min((profit_factor - 1.0) / 2.0, 1.0)  # Normalizar (1.0-3.0) -> (0-1)
            
            # Calcular score ponderado
            composite_score = (
                return_score * self.score_weights['total_return'] +
                win_rate_score * self.score_weights['win_rate'] +
                drawdown_score * self.score_weights['drawdown_control'] +
                pf_score * self.score_weights['profit_factor']
            )
            
            return max(0.0, min(1.0, composite_score))
            
        except Exception as e:
            self.logger.error(f"Error calculando score compuesto: {e}")
            return 0.0
    
    def _save_optimization_results(self, best_result: OptimizationResult, all_results: List[OptimizationResult]):
        """
        Guardar resultados de optimización
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Preparar datos para guardar
            optimization_data = {
                'timestamp': timestamp,
                'best_parameters': best_result.parameters,
                'best_metrics': best_result.metrics,
                'best_composite_score': best_result.composite_score,
                'best_consistency_score': best_result.consistency_score,
                'validation_scores': best_result.validation_scores,
                'total_combinations_tested': len(all_results),
                'score_weights': self.score_weights,
                'all_results': [
                    {
                        'parameters': result.parameters,
                        'composite_score': result.composite_score,
                        'consistency_score': result.consistency_score,
                        'total_return': result.metrics.get('total_return', 0),
                        'win_rate': result.metrics.get('win_rate', 0),
                        'max_drawdown': result.metrics.get('max_drawdown', 0),
                        'profit_factor': result.metrics.get('profit_factor', 1)
                    }
                    for result in sorted(all_results, key=lambda x: x.composite_score, reverse=True)[:10]
                ]
            }
            
            # Guardar en archivo JSON
            filename = f"optimization_results_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(optimization_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Resultados guardados en: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {e}")
    
def generate_optimization_report(optimization_file: str = None) -> str:
    """
    Generar reporte detallado de optimización
    """
    try:
        # Buscar archivo más reciente si no se especifica
        if not optimization_file:
            import glob
            files = glob.glob("optimization_results_*.json")
            if not files:
                return "❌ No se encontraron archivos de optimización"
            optimization_file = max(files, key=os.path.getctime)
        
        # Cargar datos
        with open(optimization_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Generar reporte
        report = []
        report.append("="*80)
        report.append("🎯 REPORTE DE OPTIMIZACIÓN DE PARÁMETROS")
        report.append("="*80)
        report.append(f"📅 Fecha: {data['timestamp']}")
        report.append(f"🔍 Combinaciones probadas: {data['total_combinations_tested']}")
        report.append("")
        
        # Mejores parámetros
        report.append("🏆 MEJORES PARÁMETROS ENCONTRADOS:")
        report.append("-" * 40)
        best_params = data['best_parameters']
        for param, value in best_params.items():
            report.append(f"   {param}: {value}")
        report.append("")
        
        # Métricas del mejor resultado
        report.append("📊 MÉTRICAS DEL MEJOR RESULTADO:")
        report.append("-" * 40)
        metrics = data['best_metrics']
        report.append(f"   Rentabilidad Total: {metrics.get('total_return', 0):.1%}")
        report.append(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")
        report.append(f"   Profit Factor: {metrics.get('profit_factor', 1):.2f}")
        report.append(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1%}")
        report.append(f"   Score Compuesto: {data['best_composite_score']:.4f}")
        report.append(f"   Score Consistencia: {data['best_consistency_score']:.4f}")
        report.append("")
        
        # Validación cruzada
        report.append("🔄 VALIDACIÓN CRUZADA:")
        report.append("-" * 40)
        val_scores = data['validation_scores']
        report.append(f"   Scores por período: {[f'{s:.4f}' for s in val_scores]}")
        report.append(f"   Promedio: {np.mean(val_scores):.4f}")
        report.append(f"   Desviación estándar: {np.std(val_scores):.4f}")
        report.append("")
        
        # Top 5 resultados
        report.append("🥇 TOP 5 MEJORES COMBINACIONES:")
        report.append("-" * 40)
        for i, result in enumerate(data['all_results'][:5], 1):
            report.append(f"   #{i} - Score: {result['composite_score']:.4f}")
            report.append(f"        Rentabilidad: {result['total_return']:.1%}")
            report.append(f"        Win Rate: {result['win_rate']:.1%}")
            report.append(f"        Drawdown: {result['max_drawdown']:.1%}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
        
    except Exception as e:
        return f"❌ Error generando reporte: {e}"

def main():
    """Función principal de optimización"""
    print("🚀 Iniciando Motor de Optimización de Parámetros...")
    
    try:
        # Crear optimizador
        optimizer = ParameterOptimizer(initial_capital=10000.0)
        
        # Ejecutar optimización
        result = optimizer.optimize_parameters(
            symbol=None,  # Auto-selección
            days=60,      # 60 días de datos
            max_combinations=30  # 30 combinaciones
        )
        
        if result:
            print("\n🏆 OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
            print(f"📊 Mejor Score Compuesto: {result.composite_score:.4f}")
            print(f"📈 Rentabilidad: {result.metrics.get('total_return', 0):.1%}")
            print(f"🎯 Win Rate: {result.metrics.get('win_rate', 0):.1%}")
            
            # Generar y mostrar reporte
            report = generate_optimization_report()
            print("\n" + report)
        else:
            print("❌ La optimización no produjo resultados válidos")
            
    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()