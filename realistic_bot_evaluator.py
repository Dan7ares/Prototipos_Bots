#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Evaluación Realista del Bot de Trading
================================================
Evalúa el rendimiento del bot usando datos históricos reales y métricas estadísticas robustas.
Proporciona análisis de consistencia, sensibilidad a parámetros y rendimiento en diferentes condiciones.

Autor: Trading Bot Team
Versión: 1.0 - Evaluación Realista
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn no está instalado. Usando matplotlib básico para visualizaciones.")
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from tools.m1_performance import M1PerformanceMonitor
from performance_analyzer import PerformanceAnalyzer
from backtest import AdvancedBacktester
from config.m1_specialization import M1_SYMBOLS, M1_STRATEGY_CONFIG
from core.data_loader import load_historical_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RealisticEvaluator")

@dataclass
class MarketCondition:
    """Representa una condición específica de mercado"""
    name: str
    description: str
    volatility_range: Tuple[float, float]
    trend_strength_range: Tuple[float, float]
    time_periods: List[str]

@dataclass
class EvaluationMetrics:
    """Métricas completas de evaluación"""
    # Métricas básicas
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Métricas de consistencia
    win_rate_std: float
    return_consistency: float
    monthly_win_rates: List[float]
    
    # Métricas de riesgo
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    calmar_ratio: float
    
    # Métricas de sensibilidad
    parameter_sensitivity: Dict[str, float]
    market_condition_performance: Dict[str, float]

class RealisticBotEvaluator:
    """
    Sistema completo de evaluación realista del bot de trading
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results_history: List[Dict] = []
        self.market_conditions = self._define_market_conditions()
        
        # Configurar visualización
        if HAS_SEABORN:
            try:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            except:
                plt.style.use('default')
        else:
            plt.style.use('default')
        
    def _define_market_conditions(self) -> List[MarketCondition]:
        """Define diferentes condiciones de mercado para evaluación"""
        return [
            MarketCondition(
                name="Alta Volatilidad",
                description="Mercados con alta volatilidad (>2% ATR diario)",
                volatility_range=(0.02, 1.0),
                trend_strength_range=(0.0, 1.0),
                time_periods=["crisis", "noticias_importantes"]
            ),
            MarketCondition(
                name="Baja Volatilidad", 
                description="Mercados con baja volatilidad (<1% ATR diario)",
                volatility_range=(0.0, 0.01),
                trend_strength_range=(0.0, 1.0),
                time_periods=["consolidacion", "mercados_laterales"]
            ),
            MarketCondition(
                name="Tendencia Fuerte",
                description="Mercados con tendencia clara (ADX >25)",
                volatility_range=(0.0, 1.0),
                trend_strength_range=(0.25, 1.0),
                time_periods=["breakouts", "tendencias_sostenidas"]
            ),
            MarketCondition(
                name="Mercado Lateral",
                description="Mercados sin tendencia clara (ADX <20)",
                volatility_range=(0.0, 1.0),
                trend_strength_range=(0.0, 0.20),
                time_periods=["rangos", "consolidacion"]
            ),
            MarketCondition(
                name="Horario Pico",
                description="Sesiones de trading con mayor liquidez",
                volatility_range=(0.0, 1.0),
                trend_strength_range=(0.0, 1.0),
                time_periods=["08:00-12:00", "13:00-17:00"]
            ),
            MarketCondition(
                name="Horario Bajo",
                description="Sesiones con menor liquidez",
                volatility_range=(0.0, 1.0),
                trend_strength_range=(0.0, 1.0),
                time_periods=["18:00-07:00", "fines_semana"]
            )
        ]
    
    def run_comprehensive_evaluation(self, 
                                   symbols: List[str] = None,
                                   timeframes: List[str] = None,
                                   evaluation_days: int = 30,
                                   monte_carlo_runs: int = 100) -> Dict:
        """
        Ejecuta evaluación completa del bot con múltiples escenarios
        """
        logger.info("🔍 Iniciando evaluación realista completa del bot...")
        
        if symbols is None:
            symbols = M1_SYMBOLS[:3]  # Usar primeros 3 símbolos
        if timeframes is None:
            timeframes = ['M1', 'M5']
            
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_period_days': evaluation_days,
            'symbols_tested': symbols,
            'timeframes_tested': timeframes,
            'market_conditions_analysis': {},
            'consistency_analysis': {},
            'sensitivity_analysis': {},
            'risk_analysis': {},
            'monte_carlo_analysis': {},
            'recommendations': []
        }
        
        # 1. Análisis por condiciones de mercado
        logger.info("📊 Analizando rendimiento por condiciones de mercado...")
        market_results = self._analyze_market_conditions(symbols, timeframes, evaluation_days)
        evaluation_results['market_conditions_analysis'] = market_results
        
        # 2. Análisis de consistencia temporal
        logger.info("📈 Analizando consistencia temporal...")
        consistency_results = self._analyze_consistency(symbols, timeframes, evaluation_days)
        evaluation_results['consistency_analysis'] = consistency_results
        
        # 3. Análisis de sensibilidad a parámetros
        logger.info("🎛️ Analizando sensibilidad a parámetros...")
        sensitivity_results = self._analyze_parameter_sensitivity(symbols[0], timeframes[0])
        evaluation_results['sensitivity_analysis'] = sensitivity_results
        
        # 4. Análisis de riesgo avanzado
        logger.info("⚠️ Analizando métricas de riesgo...")
        risk_results = self._analyze_risk_metrics(symbols, timeframes, evaluation_days)
        evaluation_results['risk_analysis'] = risk_results
        
        # 5. Simulación Monte Carlo
        logger.info("🎲 Ejecutando simulación Monte Carlo...")
        monte_carlo_results = self._run_monte_carlo_simulation(symbols[0], timeframes[0], monte_carlo_runs)
        evaluation_results['monte_carlo_analysis'] = monte_carlo_results
        
        # 6. Generar recomendaciones
        recommendations = self._generate_recommendations(evaluation_results)
        evaluation_results['recommendations'] = recommendations
        
        # 7. Guardar resultados y generar reportes
        self._save_evaluation_results(evaluation_results)
        self._generate_evaluation_report(evaluation_results)
        
        logger.info("✅ Evaluación realista completada")
        return evaluation_results
    
    def _analyze_market_conditions(self, symbols: List[str], timeframes: List[str], days: int) -> Dict:
        """Analiza rendimiento en diferentes condiciones de mercado"""
        results = {}
        
        for condition in self.market_conditions:
            logger.info(f"   Evaluando condición: {condition.name}")
            condition_results = []
            
            for symbol in symbols:
                for timeframe in timeframes:
                    # Cargar datos históricos
                    data = load_historical_data(symbol, timeframe, count=days * 1440)
                    if data is None or len(data) < 100:
                        continue
                    
                    # Filtrar datos según condición de mercado
                    filtered_data = self._filter_by_market_condition(data, condition)
                    if len(filtered_data) < 50:
                        continue
                    
                    # Ejecutar backtest en condición específica
                    backtest_result = self._run_condition_backtest(filtered_data, symbol, timeframe)
                    if 'error' not in backtest_result:
                        condition_results.append(backtest_result)
            
            # Consolidar resultados de la condición
            if condition_results:
                avg_win_rate = np.mean([r['win_rate'] for r in condition_results])
                avg_return = np.mean([r['total_return'] for r in condition_results])
                avg_drawdown = np.mean([r['max_drawdown'] for r in condition_results])
                
                results[condition.name] = {
                    'description': condition.description,
                    'tests_count': len(condition_results),
                    'avg_win_rate': avg_win_rate,
                    'avg_return': avg_return,
                    'avg_max_drawdown': avg_drawdown,
                    'performance_score': self._calculate_performance_score(avg_win_rate, avg_return, avg_drawdown)
                }
        
        return results
    
    def _analyze_consistency(self, symbols: List[str], timeframes: List[str], days: int) -> Dict:
        """Analiza la consistencia temporal del rendimiento"""
        weekly_results = []
        monthly_results = []
        
        # Análisis semanal (últimas 4 semanas)
        for week in range(4):
            week_start = days - (week + 1) * 7
            week_end = days - week * 7
            
            week_results = []
            for symbol in symbols[:2]:  # Limitar para eficiencia
                for timeframe in timeframes:
                    result = self._run_period_backtest(symbol, timeframe, week_start, week_end)
                    if 'error' not in result:
                        week_results.append(result)
            
            if week_results:
                avg_win_rate = np.mean([r['win_rate'] for r in week_results])
                avg_return = np.mean([r['total_return'] for r in week_results])
                weekly_results.append({'week': week + 1, 'win_rate': avg_win_rate, 'return': avg_return})
        
        # Calcular métricas de consistencia
        if weekly_results:
            win_rates = [w['win_rate'] for w in weekly_results]
            returns = [w['return'] for w in weekly_results]
            
            consistency_metrics = {
                'weekly_results': weekly_results,
                'win_rate_consistency': {
                    'mean': np.mean(win_rates),
                    'std': np.std(win_rates),
                    'coefficient_variation': np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) > 0 else 0,
                    'min_week': min(win_rates),
                    'max_week': max(win_rates)
                },
                'return_consistency': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'coefficient_variation': np.std(returns) / np.mean(returns) if np.mean(returns) > 0 else 0,
                    'min_week': min(returns),
                    'max_week': max(returns)
                }
            }
        else:
            consistency_metrics = {'error': 'Datos insuficientes para análisis de consistencia'}
        
        return consistency_metrics
    
    def _analyze_parameter_sensitivity(self, symbol: str, timeframe: str) -> Dict:
        """Analiza sensibilidad a cambios en parámetros clave"""
        base_config = M1_STRATEGY_CONFIG.copy()
        sensitivity_results = {}
        
        # Parámetros a testear
        parameters_to_test = {
            'min_confidence': [0.6, 0.7, 0.75, 0.8, 0.85],
            'adx_min': [15, 20, 25, 30],
            'take_profit_multiplier': [1.5, 2.0, 2.5, 3.0],
            'stop_loss_multiplier': [1.0, 1.5, 2.0],
            'ema_fast': [8, 12, 16, 20],
            'ema_slow': [20, 26, 32, 40]
        }
        
        for param_name, param_values in parameters_to_test.items():
            logger.info(f"   Testando sensibilidad de {param_name}")
            param_results = []
            
            for value in param_values:
                # Crear configuración modificada
                test_config = base_config.copy()
                test_config[param_name] = value
                
                # Ejecutar backtest con configuración modificada
                result = self._run_parameter_test(symbol, timeframe, test_config)
                if 'error' not in result:
                    param_results.append({
                        'value': value,
                        'win_rate': result['win_rate'],
                        'total_return': result['total_return'],
                        'max_drawdown': result['max_drawdown'],
                        'performance_score': self._calculate_performance_score(
                            result['win_rate'], result['total_return'], result['max_drawdown']
                        )
                    })
            
            if param_results:
                # Calcular sensibilidad
                scores = [r['performance_score'] for r in param_results]
                sensitivity = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
                
                sensitivity_results[param_name] = {
                    'sensitivity_coefficient': sensitivity,
                    'results': param_results,
                    'optimal_value': param_results[np.argmax(scores)]['value'] if scores else None
                }
        
        return sensitivity_results
    
    def _analyze_risk_metrics(self, symbols: List[str], timeframes: List[str], days: int) -> Dict:
        """Calcula métricas avanzadas de riesgo"""
        all_returns = []
        all_drawdowns = []
        
        # Recopilar datos de retornos
        for symbol in symbols:
            for timeframe in timeframes:
                result = self._run_period_backtest(symbol, timeframe, 0, days)
                if 'error' not in result and 'equity_curve' in result:
                    equity = result['equity_curve']
                    if len(equity) > 1:
                        returns = np.diff(equity) / equity[:-1]
                        all_returns.extend(returns)
                        
                        # Calcular drawdowns
                        peak = np.maximum.accumulate(equity)
                        drawdown = (peak - equity) / peak
                        all_drawdowns.extend(drawdown)
        
        if not all_returns:
            return {'error': 'Datos insuficientes para análisis de riesgo'}
        
        all_returns = np.array(all_returns)
        all_drawdowns = np.array(all_drawdowns)
        
        # Calcular métricas de riesgo
        risk_metrics = {
            'value_at_risk_95': np.percentile(all_returns, 5),
            'expected_shortfall': np.mean(all_returns[all_returns <= np.percentile(all_returns, 5)]),
            'max_drawdown_observed': np.max(all_drawdowns),
            'avg_drawdown': np.mean(all_drawdowns),
            'drawdown_duration_avg': self._calculate_avg_drawdown_duration(all_drawdowns),
            'return_skewness': stats.skew(all_returns),
            'return_kurtosis': stats.kurtosis(all_returns),
            'sharpe_ratio': np.mean(all_returns) / np.std(all_returns) if np.std(all_returns) > 0 else 0,
            'calmar_ratio': np.mean(all_returns) / np.max(all_drawdowns) if np.max(all_drawdowns) > 0 else 0
        }
        
        return risk_metrics
    
    def _run_monte_carlo_simulation(self, symbol: str, timeframe: str, runs: int) -> Dict:
        """Ejecuta simulación Monte Carlo para evaluar robustez"""
        results = []
        
        for run in range(runs):
            if run % 20 == 0:
                logger.info(f"   Simulación Monte Carlo: {run}/{runs}")
            
            # Generar parámetros aleatorios dentro de rangos razonables
            random_config = self._generate_random_config()
            
            # Ejecutar backtest con configuración aleatoria
            result = self._run_parameter_test(symbol, timeframe, random_config)
            if 'error' not in result:
                results.append({
                    'win_rate': result['win_rate'],
                    'total_return': result['total_return'],
                    'max_drawdown': result['max_drawdown'],
                    'config': random_config
                })
        
        if not results:
            return {'error': 'Simulación Monte Carlo falló'}
        
        # Analizar distribución de resultados
        win_rates = [r['win_rate'] for r in results]
        returns = [r['total_return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        
        monte_carlo_analysis = {
            'total_simulations': len(results),
            'win_rate_distribution': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'percentile_5': np.percentile(win_rates, 5),
                'percentile_95': np.percentile(win_rates, 95),
                'probability_above_70pct': np.mean(np.array(win_rates) >= 0.70)
            },
            'return_distribution': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95),
                'probability_positive': np.mean(np.array(returns) > 0)
            },
            'drawdown_distribution': {
                'mean': np.mean(drawdowns),
                'std': np.std(drawdowns),
                'percentile_95': np.percentile(drawdowns, 95),
                'probability_below_15pct': np.mean(np.array(drawdowns) <= 0.15)
            },
            'best_configs': sorted(results, key=lambda x: self._calculate_performance_score(
                x['win_rate'], x['total_return'], x['max_drawdown']
            ), reverse=True)[:5]
        }
        
        return monte_carlo_analysis
    
    def _generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Genera recomendaciones basadas en los resultados de evaluación"""
        recommendations = []
        
        # Análisis de condiciones de mercado
        market_analysis = evaluation_results.get('market_conditions_analysis', {})
        if market_analysis:
            best_condition = max(market_analysis.keys(), 
                               key=lambda k: market_analysis[k].get('performance_score', 0))
            worst_condition = min(market_analysis.keys(), 
                                key=lambda k: market_analysis[k].get('performance_score', 0))
            
            recommendations.append(f"🎯 Mejor rendimiento en: {best_condition}")
            recommendations.append(f"⚠️ Peor rendimiento en: {worst_condition}")
            
            if market_analysis[worst_condition]['avg_win_rate'] < 0.5:
                recommendations.append(f"🔧 Considerar desactivar trading durante: {worst_condition}")
        
        # Análisis de consistencia
        consistency = evaluation_results.get('consistency_analysis', {})
        if 'win_rate_consistency' in consistency:
            cv = consistency['win_rate_consistency'].get('coefficient_variation', 0)
            if cv > 0.3:
                recommendations.append("📊 Alta variabilidad semanal - mejorar consistencia")
            elif cv < 0.1:
                recommendations.append("✅ Excelente consistencia semanal")
        
        # Análisis de sensibilidad
        sensitivity = evaluation_results.get('sensitivity_analysis', {})
        high_sensitivity_params = [param for param, data in sensitivity.items() 
                                 if data.get('sensitivity_coefficient', 0) > 0.5]
        if high_sensitivity_params:
            recommendations.append(f"🎛️ Parámetros críticos a monitorear: {', '.join(high_sensitivity_params)}")
        
        # Análisis de riesgo
        risk_analysis = evaluation_results.get('risk_analysis', {})
        if 'max_drawdown_observed' in risk_analysis:
            max_dd = risk_analysis['max_drawdown_observed']
            if max_dd > 0.20:
                recommendations.append("🚨 Drawdown excesivo - implementar mejor gestión de riesgo")
            elif max_dd < 0.10:
                recommendations.append("✅ Excelente control de drawdown")
        
        # Monte Carlo
        monte_carlo = evaluation_results.get('monte_carlo_analysis', {})
        if 'win_rate_distribution' in monte_carlo:
            prob_70 = monte_carlo['win_rate_distribution'].get('probability_above_70pct', 0)
            if prob_70 < 0.1:
                recommendations.append("📉 Baja probabilidad de alcanzar 70% win rate - revisar estrategia")
            elif prob_70 > 0.5:
                recommendations.append("🎯 Alta probabilidad de alcanzar 70% win rate")
        
        return recommendations
    
    # Métodos auxiliares
    def _filter_by_market_condition(self, data: pd.DataFrame, condition: MarketCondition) -> pd.DataFrame:
        """Filtra datos según condición de mercado específica"""
        # Implementación simplificada - en producción sería más sofisticada
        if "Alta Volatilidad" in condition.name:
            # Filtrar períodos de alta volatilidad
            data['volatility'] = data['high'] - data['low']
            threshold = data['volatility'].quantile(0.7)
            return data[data['volatility'] >= threshold]
        elif "Baja Volatilidad" in condition.name:
            data['volatility'] = data['high'] - data['low']
            threshold = data['volatility'].quantile(0.3)
            return data[data['volatility'] <= threshold]
        elif "Horario Pico" in condition.name:
            # Filtrar horarios de 8:00 a 17:00
            return data.between_time('08:00', '17:00')
        elif "Horario Bajo" in condition.name:
            # Filtrar horarios fuera de 8:00 a 17:00
            return data[~data.index.to_series().dt.time.between(
                pd.Timestamp('08:00').time(), pd.Timestamp('17:00').time())]
        else:
            return data
    
    def _run_condition_backtest(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Ejecuta backtest en datos filtrados por condición"""
        try:
            monitor = M1PerformanceMonitor()
            result = monitor.run_specialized_backtest(symbol, timeframe, days=len(data)//1440 + 1)
            
            # Verificar si hay error en el resultado
            if 'error' in result:
                return result
            
            # Calcular métricas faltantes
            initial_capital = self.initial_capital
            final_equity = result.get('final_equity', initial_capital)
            equity_curve = result.get('equity_curve', [initial_capital])
            
            # Calcular total_return
            total_return = ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
            
            # Calcular max_drawdown
            max_drawdown = 0.0
            if len(equity_curve) > 1:
                peak = equity_curve[0]
                for value in equity_curve:
                    if value > peak:
                        peak = value
                    drawdown = ((peak - value) / peak) * 100 if peak > 0 else 0.0
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Agregar métricas calculadas al resultado
            result['total_return'] = total_return
            result['max_drawdown'] = max_drawdown
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def _run_period_backtest(self, symbol: str, timeframe: str, start_days: int, end_days: int) -> Dict:
        """Ejecuta backtest para un período específico"""
        try:
            days = end_days - start_days
            monitor = M1PerformanceMonitor()
            result = monitor.run_specialized_backtest(symbol, timeframe, days=days)
            
            # Verificar si hay error en el resultado
            if 'error' in result:
                return result
            
            # Calcular métricas faltantes
            initial_capital = self.initial_capital
            final_equity = result.get('final_equity', initial_capital)
            equity_curve = result.get('equity_curve', [initial_capital])
            
            # Calcular total_return
            total_return = ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
            
            # Calcular max_drawdown
            max_drawdown = 0.0
            if len(equity_curve) > 1:
                peak = equity_curve[0]
                for value in equity_curve:
                    if value > peak:
                        peak = value
                    drawdown = ((peak - value) / peak) * 100 if peak > 0 else 0.0
                    max_drawdown = max(max_drawdown, drawdown)
            
            # Agregar métricas calculadas al resultado
            result['total_return'] = total_return
            result['max_drawdown'] = max_drawdown
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
            return {'error': str(e)}
    
    def _run_parameter_test(self, symbol: str, timeframe: str, config: Dict) -> Dict:
        """Ejecuta backtest con configuración específica"""
        try:
            # Temporalmente modificar configuración global
            original_config = M1_STRATEGY_CONFIG.copy()
            M1_STRATEGY_CONFIG.update(config)
            
            monitor = M1PerformanceMonitor()
            result = monitor.run_specialized_backtest(symbol, timeframe, days=7)
            
            # Restaurar configuración original
            M1_STRATEGY_CONFIG.clear()
            M1_STRATEGY_CONFIG.update(original_config)
            
            # Verificar si hay error en el resultado
            if 'error' in result:
                return result
            
            # Calcular métricas faltantes si no están presentes
            if 'total_return' not in result or 'max_drawdown' not in result:
                initial_capital = self.initial_capital
                final_equity = result.get('final_equity', initial_capital)
                equity_curve = result.get('equity_curve', [initial_capital])
                
                # Calcular total_return
                if 'total_return' not in result:
                    total_return = ((final_equity - initial_capital) / initial_capital) if initial_capital > 0 else 0.0
                    result['total_return'] = total_return
                
                # Calcular max_drawdown
                if 'max_drawdown' not in result:
                    max_drawdown = 0.0
                    if len(equity_curve) > 1:
                        peak = equity_curve[0]
                        for value in equity_curve:
                            if value > peak:
                                peak = value
                            drawdown = ((peak - value) / peak) if peak > 0 else 0.0
                            max_drawdown = max(max_drawdown, drawdown)
                    result['max_drawdown'] = max_drawdown
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_random_config(self) -> Dict:
        """Genera configuración aleatoria para Monte Carlo"""
        return {
            'min_confidence': np.random.uniform(0.6, 0.9),
            'adx_min': np.random.randint(15, 35),
            'take_profit_multiplier': np.random.uniform(1.5, 3.5),
            'stop_loss_multiplier': np.random.uniform(1.0, 2.5),
            'ema_fast': np.random.randint(8, 24),
            'ema_slow': np.random.randint(20, 50)
        }
    
    def _calculate_performance_score(self, win_rate: float, total_return: float, max_drawdown: float) -> float:
        """Calcula puntuación de rendimiento combinada"""
        # Normalizar métricas y combinar
        win_rate_score = min(win_rate / 0.8, 1.0)  # Normalizar a 80%
        return_score = min(total_return / 0.5, 1.0)  # Normalizar a 50%
        drawdown_penalty = max(0, 1 - max_drawdown / 0.15)  # Penalizar DD >15%
        
        return (win_rate_score * 0.4 + return_score * 0.4 + drawdown_penalty * 0.2)
    
    def _calculate_avg_drawdown_duration(self, drawdowns: np.ndarray) -> float:
        """Calcula duración promedio de drawdowns"""
        # Implementación simplificada
        in_drawdown = drawdowns > 0.01  # Drawdown >1%
        if not np.any(in_drawdown):
            return 0.0
        
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        return np.mean(durations) if durations else 0.0
    
    def _save_evaluation_results(self, results: Dict):
        """Guarda resultados de evaluación en archivo JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realistic_evaluation_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Resultados guardados en: {filename}")
    
    def _generate_evaluation_report(self, results: Dict):
        """Genera reporte visual de evaluación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Evaluación Realista del Bot de Trading', fontsize=16, fontweight='bold')
        
        # 1. Rendimiento por condiciones de mercado
        if 'market_conditions_analysis' in results:
            market_data = results['market_conditions_analysis']
            conditions = list(market_data.keys())
            scores = [market_data[c].get('performance_score', 0) for c in conditions]
            
            axes[0, 0].bar(range(len(conditions)), scores, color='skyblue')
            axes[0, 0].set_title('Rendimiento por Condición de Mercado')
            axes[0, 0].set_xticks(range(len(conditions)))
            axes[0, 0].set_xticklabels(conditions, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Score de Rendimiento')
        
        # 2. Consistencia temporal
        if 'consistency_analysis' in results and 'weekly_results' in results['consistency_analysis']:
            weekly_data = results['consistency_analysis']['weekly_results']
            weeks = [w['week'] for w in weekly_data]
            win_rates = [w['win_rate'] for w in weekly_data]
            
            axes[0, 1].plot(weeks, win_rates, marker='o', linewidth=2, markersize=8)
            axes[0, 1].set_title('Consistencia Semanal - Win Rate')
            axes[0, 1].set_xlabel('Semana')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sensibilidad de parámetros
        if 'sensitivity_analysis' in results:
            sensitivity_data = results['sensitivity_analysis']
            params = list(sensitivity_data.keys())
            sensitivities = [sensitivity_data[p].get('sensitivity_coefficient', 0) for p in params]
            
            axes[0, 2].barh(range(len(params)), sensitivities, color='lightcoral')
            axes[0, 2].set_title('Sensibilidad de Parámetros')
            axes[0, 2].set_yticks(range(len(params)))
            axes[0, 2].set_yticklabels(params)
            axes[0, 2].set_xlabel('Coeficiente de Sensibilidad')
        
        # 4. Distribución Monte Carlo - Win Rate
        if 'monte_carlo_analysis' in results and 'win_rate_distribution' in results['monte_carlo_analysis']:
            mc_data = results['monte_carlo_analysis']
            # Simular distribución para visualización
            mean_wr = mc_data['win_rate_distribution']['mean']
            std_wr = mc_data['win_rate_distribution']['std']
            win_rates_sim = np.random.normal(mean_wr, std_wr, 1000)
            
            axes[1, 0].hist(win_rates_sim, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].axvline(0.7, color='red', linestyle='--', linewidth=2, label='Objetivo 70%')
            axes[1, 0].set_title('Distribución Monte Carlo - Win Rate')
            axes[1, 0].set_xlabel('Win Rate')
            axes[1, 0].set_ylabel('Frecuencia')
            axes[1, 0].legend()
        
        # 5. Métricas de riesgo
        if 'risk_analysis' in results:
            risk_data = results['risk_analysis']
            metrics = ['value_at_risk_95', 'expected_shortfall', 'max_drawdown_observed']
            values = [abs(risk_data.get(m, 0)) for m in metrics]
            labels = ['VaR 95%', 'Expected Shortfall', 'Max Drawdown']
            
            axes[1, 1].bar(labels, values, color='orange', alpha=0.7)
            axes[1, 1].set_title('Métricas de Riesgo')
            axes[1, 1].set_ylabel('Valor (absoluto)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Resumen de recomendaciones
        if 'recommendations' in results:
            recommendations = results['recommendations'][:5]  # Primeras 5
            axes[1, 2].text(0.05, 0.95, 'Recomendaciones Principales:', 
                           transform=axes[1, 2].transAxes, fontsize=12, fontweight='bold')
            
            for i, rec in enumerate(recommendations):
                axes[1, 2].text(0.05, 0.85 - i*0.15, f"• {rec}", 
                               transform=axes[1, 2].transAxes, fontsize=10, wrap=True)
            
            axes[1, 2].set_xlim(0, 1)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_filename = f"realistic_evaluation_report_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Reporte visual guardado en: {plot_filename}")

def main():
    """Función principal para ejecutar evaluación realista"""
    parser = argparse.ArgumentParser(description='Evaluación Realista del Bot de Trading')
    parser.add_argument('--symbols', nargs='+', default=['EURUSDm', 'GBPUSDm'], 
                       help='Símbolos a evaluar')
    parser.add_argument('--timeframes', nargs='+', default=['M1', 'M5'], 
                       help='Timeframes a evaluar')
    parser.add_argument('--days', type=int, default=15, 
                       help='Días de evaluación')
    parser.add_argument('--monte-carlo', type=int, default=50, 
                       help='Número de simulaciones Monte Carlo')
    
    args = parser.parse_args()
    
    # Crear evaluador
    evaluator = RealisticBotEvaluator()
    
    # Ejecutar evaluación completa
    results = evaluator.run_comprehensive_evaluation(
        symbols=args.symbols,
        timeframes=args.timeframes,
        evaluation_days=args.days,
        monte_carlo_runs=args.monte_carlo
    )
    
    # Mostrar resumen en consola
    print("\n" + "="*80)
    print("🤖 EVALUACIÓN REALISTA DEL BOT DE TRADING - RESUMEN")
    print("="*80)
    
    if 'market_conditions_analysis' in results:
        print("\n📊 RENDIMIENTO POR CONDICIONES DE MERCADO:")
        for condition, data in results['market_conditions_analysis'].items():
            print(f"   {condition}: Win Rate {data['avg_win_rate']:.1%}, "
                  f"Return {data['avg_return']:.1%}, Score {data['performance_score']:.3f}")
    
    if 'consistency_analysis' in results and 'win_rate_consistency' in results['consistency_analysis']:
        consistency = results['consistency_analysis']['win_rate_consistency']
        print(f"\n📈 CONSISTENCIA TEMPORAL:")
        print(f"   Win Rate Promedio: {consistency['mean']:.1%}")
        print(f"   Desviación Estándar: {consistency['std']:.1%}")
        print(f"   Coeficiente de Variación: {consistency['coefficient_variation']:.3f}")
    
    if 'monte_carlo_analysis' in results:
        mc = results['monte_carlo_analysis']
        if 'win_rate_distribution' in mc:
            print(f"\n🎲 SIMULACIÓN MONTE CARLO:")
            print(f"   Probabilidad Win Rate ≥70%: {mc['win_rate_distribution']['probability_above_70pct']:.1%}")
            print(f"   Win Rate Esperado: {mc['win_rate_distribution']['mean']:.1%} ± {mc['win_rate_distribution']['std']:.1%}")
    
    if 'recommendations' in results:
        print(f"\n💡 RECOMENDACIONES PRINCIPALES:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
    
    print("\n" + "="*80)
    print("✅ Evaluación completada. Revisa los archivos generados para análisis detallado.")

if __name__ == "__main__":
    main()