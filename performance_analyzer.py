#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analizador de Rendimiento y Comparación
---------------------------------------
Sistema para analizar y comparar métricas antes y después de optimizaciones.

Autor: Trading Bot Team
Versión: 1.0 - Análisis Comparativo
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

from backtest import AdvancedBacktester, print_detailed_report
from config.settings import STRATEGY_CONFIG, PROFITABILITY_TARGETS

class PerformanceAnalyzer:
    """
    Analizador de rendimiento con comparación antes/después
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.logger = logging.getLogger('PerformanceAnalyzer')
        self.baseline_results = None
        self.optimized_results = None
        self.initial_capital = initial_capital
        
        # Configuración baseline para comparaciones
        self.baseline_config = {
            'ema_fast': 5, 'ema_medium': 10, 'ema_slow': 21,
            'rsi_period': 8, 'rsi_oversold': 35, 'rsi_overbought': 65,
            'bollinger_period': 20, 'bollinger_std': 2.0,
            'atr_period': 14, 'min_confirmations': 4,
            'take_profit_multiplier': 2.0, 'stop_loss_multiplier': 1.0,
            'min_confidence': 0.70
        }
    
    def run_baseline_analysis(self, days: int = 30) -> Dict:
        """
        Ejecuta análisis con configuración base (antes de optimización)
        """
        self.logger.info("📊 Ejecutando análisis baseline...")
        
        # Configuración original (antes de optimización)
        original_config = {
            'ema_fast': 5, 'ema_medium': 10, 'ema_slow': 21,
            'rsi_period': 8, 'rsi_oversold': 35, 'rsi_overbought': 65,
            'bollinger_period': 20, 'bollinger_std': 2.0,
            'atr_period': 14, 'min_confirmations': 4,
            'take_profit_multiplier': 2.0, 'stop_loss_multiplier': 1.0,
            'min_confidence': 0.70
        }
        
        # Actualizar configuración temporalmente
        current_config = STRATEGY_CONFIG.copy()
        STRATEGY_CONFIG.clear()
        STRATEGY_CONFIG.update(original_config)
        
        try:
            backtester = AdvancedBacktester(initial_capital=self.initial_capital)
            self.baseline_results = backtester.run_advanced_backtest(days=days)
            self.baseline_results['config_type'] = 'baseline'
            
        finally:
            # Restaurar configuración
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(current_config)
        
        return self.baseline_results
    
    def _run_analysis_with_config_fast(self, config: Dict, label: str = "FAST", symbol: str = None, days: int = 15) -> object:
        """
        Ejecuta análisis rápido con configuración específica.
        Devuelve un objeto con métricas clave para impresión rápida.
        """
        self.logger.info(f"⚡ Ejecutando análisis rápido {label}...")
        # Resguardar configuración actual
        original_config = STRATEGY_CONFIG.copy()
        try:
            # Copia segura: si config es STRATEGY_CONFIG, evitar limpiar el mismo dict
            safe_config = dict(config) if isinstance(config, dict) else {}
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(safe_config)
            backtester = AdvancedBacktester(initial_capital=self.initial_capital)
            results = backtester.run_backtest(symbol=symbol, days=days, save_results=False) or {}

            # Persistir baseline/optimized para reportes comparativos si corresponde
            if "BASELINE" in label.upper():
                self.baseline_results = results
            if "OPTIMIZED" in label.upper():
                self.optimized_results = results

            initial_capital = self.initial_capital
            class FastMetrics:
                def __init__(self, data: Dict, initial_capital: float):
                    # Soporta clave 'profitability' o 'total_return'
                    self.total_return = data.get('total_return', data.get('profitability', 0.0))
                    self.win_rate = data.get('win_rate', 0.0)
                    self.profit_factor = data.get('profit_factor', 0.0)
                    self.max_drawdown = data.get('max_drawdown', 0.0)
                    self.total_trades = data.get('total_trades', 0)
                    self.final_equity = data.get('final_equity', initial_capital)
                    self.initial_equity = initial_capital
            return FastMetrics(results, initial_capital)
        except Exception as e:
            self.logger.error(f"❌ Error en análisis rápido {label}: {e}")
            return self._create_empty_metrics()
        finally:
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(original_config)
    
    def _run_analysis_with_config(self, config: Dict, symbol: str = None, days: int = 30, label: str = "ANALYSIS") -> object:
        """
        Ejecuta análisis con configuración específica
        """
        self.logger.info(f"📊 Ejecutando análisis {label}...")
        try:
            # Actualizar configuración temporalmente ANTES de crear el backtester
            original_config = STRATEGY_CONFIG.copy()
            safe_config = dict(config) if isinstance(config, dict) else {}
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(safe_config)

            backtester = AdvancedBacktester(initial_capital=self.initial_capital)
            results = backtester.run_backtest(symbol=symbol, days=days, save_results=False)
            # Restaurar configuración original SIEMPRE
            STRATEGY_CONFIG.clear()
            STRATEGY_CONFIG.update(original_config)
            
            if results:
                class MetricsResult:
                    def __init__(self, data):
                        self.total_return = data.get('total_return', 0.0)
                        self.win_rate = data.get('win_rate', 0.0)
                        self.profit_factor = data.get('profit_factor', 0.0)
                        self.max_drawdown = data.get('max_drawdown', 0.0)
                        self.total_trades = data.get('total_trades', 0)
                        self.sharpe_ratio = data.get('sharpe_ratio', 0.0)
                return MetricsResult(results)
            else:
                return self._create_empty_metrics()
        except Exception as e:
            self.logger.error(f"❌ Error en análisis {label}: {e}")
            return self._create_empty_metrics()
    
    def _create_empty_metrics(self):
        """Crear métricas vacías en caso de error"""
        class EmptyMetrics:
            def __init__(self):
                self.total_return = 0.0
                self.win_rate = 0.0
                self.profit_factor = 0.0
                self.max_drawdown = 0.0
                self.total_trades = 0
                self.sharpe_ratio = 0.0
        
        return EmptyMetrics()
    
    def run_optimized_analysis(self, days: int = 30) -> Dict:
        """
        Ejecuta análisis con configuración optimizada
        """
        self.logger.info("🚀 Ejecutando análisis optimizado...")
        
        backtester = AdvancedBacktester(initial_capital=self.initial_capital)
        self.optimized_results = backtester.run_advanced_backtest(days=days)
        self.optimized_results['config_type'] = 'optimized'
        
        return self.optimized_results
    
    def run_comprehensive_analysis(self, optimized_config: Dict, symbol: str = None, days: int = 15):
        """
        Ejecuta análisis baseline y optimizado (versión rápida) y prepara recomendación.
        """
        baseline_metrics = self._run_analysis_with_config_fast(self.baseline_config, "BASELINE", symbol, days)
        optimized_metrics = self._run_analysis_with_config_fast(optimized_config, "OPTIMIZED", symbol, days)

        class ComparisonResult:
            def __init__(self, baseline, optimized, recommendation: str):
                self.baseline_metrics = baseline
                self.optimized_metrics = optimized
                self.recommendation = recommendation

        # Recomendación simple basada en objetivos
        recommendation = "Mantener configuración" if optimized_metrics.total_return >= PROFITABILITY_TARGETS.get('initial_target', 0.7) else "Ajustar parámetros y reevaluar"
        return ComparisonResult(baseline_metrics, optimized_metrics, recommendation)
    
    def generate_comparison_report(self) -> str:
        """
        Genera reporte comparativo detallado
        """
        if not self.baseline_results or not self.optimized_results:
            return "❌ Error: Faltan resultados para comparación"
        
        baseline = self.baseline_results
        optimized = self.optimized_results
        
        # Calcular mejoras
        return_improvement = optimized['total_return'] - baseline['total_return']
        winrate_improvement = optimized['win_rate'] - baseline['win_rate']
        pf_improvement = optimized['profit_factor'] - baseline['profit_factor']
        drawdown_improvement = baseline['max_drawdown'] - optimized['max_drawdown']
        
        report = f"""
🎯 REPORTE COMPARATIVO DE RENDIMIENTO
{'='*70}

📊 CONFIGURACIÓN BASELINE (ANTES):
{'-'*50}
   • Rentabilidad Total: {baseline['total_return']:.1%}
   • Win Rate: {baseline['win_rate']:.1%}
   • Profit Factor: {baseline['profit_factor']:.2f}
   • Max Drawdown: {baseline['max_drawdown']:.1%}
   • Total Trades: {baseline['total_trades']}
   • Trades Ganadores: {baseline['winning_trades']}
   • Confianza Promedio: {baseline.get('avg_signal_confidence', 0):.2f}

🚀 CONFIGURACIÓN OPTIMIZADA (DESPUÉS):
{'-'*50}
   • Rentabilidad Total: {optimized['total_return']:.1%}
   • Win Rate: {optimized['win_rate']:.1%}
   • Profit Factor: {optimized['profit_factor']:.2f}
   • Max Drawdown: {optimized['max_drawdown']:.1%}
   • Total Trades: {optimized['total_trades']}
   • Trades Ganadores: {optimized['winning_trades']}
   • Confianza Promedio: {optimized.get('avg_signal_confidence', 0):.2f}

📈 MEJORAS OBTENIDAS:
{'-'*50}
   • Rentabilidad: {return_improvement:+.1%} ({self._get_improvement_icon(return_improvement)})
   • Win Rate: {winrate_improvement:+.1%} ({self._get_improvement_icon(winrate_improvement)})
   • Profit Factor: {pf_improvement:+.2f} ({self._get_improvement_icon(pf_improvement)})
   • Drawdown: {drawdown_improvement:+.1%} ({self._get_improvement_icon(drawdown_improvement)})

🎯 CUMPLIMIENTO DE OBJETIVOS:
{'-'*50}
                    BASELINE    OPTIMIZADO    OBJETIVO
   Rentabilidad:    {baseline['total_return']:.1%}        {optimized['total_return']:.1%}        ≥65%
   Win Rate:        {baseline['win_rate']:.1%}        {optimized['win_rate']:.1%}        ≥60%
   Max Drawdown:    {baseline['max_drawdown']:.1%}        {optimized['max_drawdown']:.1%}        ≤12%
   Profit Factor:   {baseline['profit_factor']:.2f}         {optimized['profit_factor']:.2f}         ≥1.8

✅ OBJETIVOS ALCANZADOS:
{'-'*50}
"""
        
        # Verificar objetivos (dinámicos desde configuración)
        targets = PROFITABILITY_TARGETS
        objectives = [
            (f"Rentabilidad ≥{int(targets.get('initial_target', 0.15)*100)}%", optimized['total_return'] >= targets.get('initial_target', 0.15)),
            (f"Win Rate ≥{int(targets.get('min_win_rate', 0.60)*100)}%", optimized['win_rate'] >= targets.get('min_win_rate', 0.60)),
            (f"Drawdown ≤{int(targets.get('max_drawdown', 0.12)*100)}%", optimized['max_drawdown'] <= targets.get('max_drawdown', 0.12)),
            (f"Profit Factor ≥{targets.get('min_profit_factor', 1.8):.1f}", optimized['profit_factor'] >= targets.get('min_profit_factor', 1.8))
        ]
        
        achieved = sum(1 for _, achieved in objectives if achieved)
        
        for objective, is_achieved in objectives:
            icon = "✅" if is_achieved else "❌"
            report += f"   {icon} {objective}\n"
        
        report += f"\n📊 RESUMEN: {achieved}/4 objetivos alcanzados ({achieved/4*100:.0f}%)\n"
        
        # Recomendaciones
        report += f"\n💡 RECOMENDACIONES:\n{'-'*50}\n"
        
        if optimized['total_return'] >= targets.get('initial_target', 0.15):
            report += "   ✅ Rentabilidad objetivo alcanzada - Mantener configuración\n"
        else:
            report += "   📈 Rentabilidad por debajo del objetivo - Considerar ajustes adicionales\n"
        
        if return_improvement > 0:
            report += "   🎯 Optimización exitosa - Mejora significativa detectada\n"
        else:
            report += "   ⚠️ Optimización no efectiva - Revisar parámetros\n"
        
        if optimized['max_drawdown'] <= targets.get('max_drawdown', 0.12):
            report += "   🛡️ Riesgo controlado adecuadamente\n"
        else:
            report += "   ⚠️ Drawdown elevado - Implementar mejor gestión de riesgo\n"
        
        report += f"\n{'='*70}\n"
        return report

    def create_performance_visualization(self, filename: str = "performance_visualization.png"):
        """
        Crea una visualización comparativa simple (barras) entre baseline y optimizado.
        """
        try:
            if not self.baseline_results or not self.optimized_results:
                raise ValueError("Faltan resultados baseline/optimized para visualizar")
    
            metrics = ['total_return', 'win_rate', 'profit_factor']
            baseline_vals = [
                self.baseline_results.get(m, self.baseline_results.get('profitability', 0.0) if m == 'total_return' else 0)
                for m in metrics
            ]
            optimized_vals = [
                self.optimized_results.get(m, self.optimized_results.get('profitability', 0.0) if m == 'total_return' else 0)
                for m in metrics
            ]
    
            plt.figure(figsize=(8, 5))
            x = np.arange(len(metrics))
            plt.bar(x - 0.2, baseline_vals, width=0.4, label='Baseline')
            plt.bar(x + 0.2, optimized_vals, width=0.4, label='Optimized')
            plt.xticks(x, ['Rentabilidad', 'Win Rate', 'PF'])
            plt.ylabel('Valor')
            plt.title('Comparativa de Rendimiento')
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename)
            self.logger.info(f"📈 Visualización guardada en {filename}")
            return filename
        except Exception as e:
            self.logger.warning(f"⚠️ No se pudo crear visualización: {e}")
            raise

    def _get_improvement_icon(self, improvement: float) -> str:
        """Retorna icono basado en la mejora"""
        if improvement > 0.05:  # Mejora significativa
            return "🚀"
        elif improvement > 0:   # Mejora leve
            return "📈"
        elif improvement == 0:  # Sin cambio
            return "➡️"
        else:                   # Empeoramiento
            return "📉"
    
    def save_comparison_data(self, filename: str = None):
        """
        Guarda datos de comparación en archivo JSON
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_comparison_{timestamp}.json"
        
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'baseline': self.baseline_results,
            'optimized': self.optimized_results,
            'improvements': {
                'return_improvement': self.optimized_results['total_return'] - self.baseline_results['total_return'],
                'winrate_improvement': self.optimized_results['win_rate'] - self.baseline_results['win_rate'],
                'pf_improvement': self.optimized_results['profit_factor'] - self.baseline_results['profit_factor'],
                'drawdown_improvement': self.baseline_results['max_drawdown'] - self.optimized_results['max_drawdown']
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"📁 Datos de comparación guardados en: {filename}")
        return filename

    def run_hourly_volatility_analysis(self, symbol: str = None, timeframe: str = None, days: int = 15, hours_window: Tuple[str, str] = ('08:00', '17:00')) -> Dict:
        """
        Ejecuta backtests separados por horario pico (08:00–17:00) y fuera de pico,
        y prepara métricas clave para comparación.
        """
        backtester = AdvancedBacktester(initial_capital=self.initial_capital)
        peak = backtester.run_advanced_backtest(symbol=symbol, timeframe=timeframe, days=days, trading_hours_mode='peak', hours_window=hours_window)
        off = backtester.run_advanced_backtest(symbol=symbol, timeframe=timeframe, days=days, trading_hours_mode='off', hours_window=hours_window)
        
        def summarize(r: Dict) -> Dict:
            avg_win = r.get('avg_win', 0.0)
            avg_loss = r.get('avg_loss', 0.0)
            rr = (avg_win / abs(avg_loss)) if (avg_win > 0 and avg_loss < 0) else 0.0
            return {
                'win_rate': r.get('win_rate', 0.0),
                'profitability': r.get('total_return', r.get('profitability', 0.0)),
                'profit_factor': r.get('profit_factor', 0.0),
                'max_drawdown': r.get('max_drawdown', 0.0),
                'total_trades': r.get('total_trades', 0),
                'risk_reward': rr
            }
        
        def compute_segment_metrics(r: Dict) -> Dict:
            trades = r.get('trades_data', []) or []
            # Limpiar tiempos
            def to_hour(et):
                try:
                    return int(pd.to_datetime(et).hour)
                except Exception:
                    return None
            hours = [to_hour(t.get('entry_time')) for t in trades]
            profits = [float(t.get('profit', 0.0) or 0.0) for t in trades]
            adx_vals = [float(t.get('entry_adx', np.nan)) for t in trades]
            vol_vals = [float(t.get('entry_volatility', np.nan)) for t in trades]
            total = len(trades)
            wins = sum(1 for p in profits if p > 0)
            # Segmentos
            adx_min = float(STRATEGY_CONFIG.get('adx_min', 20))
            high_adx_mask = [a for a in adx_vals if not np.isnan(a) and a >= adx_min]
            low_adx_mask = [a for a in adx_vals if not np.isnan(a) and a < adx_min]
            high_adx_wins = sum(1 for i, p in enumerate(profits) if (not np.isnan(adx_vals[i]) and adx_vals[i] >= adx_min and p > 0))
            low_adx_wins = sum(1 for i, p in enumerate(profits) if (not np.isnan(adx_vals[i]) and adx_vals[i] < adx_min and p > 0))
            # Volatilidad (percentil 90)
            if len([v for v in vol_vals if not np.isnan(v)]) >= 5:
                p90 = np.nanpercentile(vol_vals, 90)
            else:
                p90 = np.nanmax(vol_vals) if vol_vals else np.nan
            spike_wins = sum(1 for i, p in enumerate(profits) if (not np.isnan(vol_vals[i]) and vol_vals[i] >= p90 and p > 0))
            spike_total = sum(1 for v in vol_vals if not np.isnan(v) and v >= p90)
            return {
                'hours_covered': len([h for h in hours if h is not None]),
                'wins': wins,
                'total': total,
                'trend_win_rate_high_adx': (high_adx_wins / max(len(high_adx_mask), 1)) if high_adx_mask else 0.0,
                'sideways_win_rate_low_adx': (low_adx_wins / max(len(low_adx_mask), 1)) if low_adx_mask else 0.0,
                'spike_adaptability_win_rate': (spike_wins / max(spike_total, 1)) if spike_total else 0.0
            }
        
        self.hourly_analysis = {
            'peak': peak,
            'off': off,
            'peak_summary': summarize(peak or {}),
            'off_summary': summarize(off or {}),
            'peak_segments': compute_segment_metrics(peak or {}),
            'off_segments': compute_segment_metrics(off or {})
        }
        return self.hourly_analysis
    
    def create_hourly_performance_charts(self, peak_report: Dict, off_report: Dict, filename: str = "hourly_performance.png"):
        """
        Crea gráfico comparativo del win rate por hora entre 08:00–17:00.
        """
        def group_by_hour(trades: List[Dict]) -> Dict[int, Dict]:
            buckets = {h: {'wins': 0, 'losses': 0, 'profit': 0.0, 'count': 0} for h in range(24)}
            for t in trades or []:
                et = t.get('entry_time')
                if et is None:
                    continue
                try:
                    ts = pd.to_datetime(et)
                except Exception:
                    continue
                h = int(ts.hour)
                profit = float(t.get('profit', 0.0) or 0.0)
                buckets[h]['count'] += 1
                buckets[h]['profit'] += profit
                if profit > 0:
                    buckets[h]['wins'] += 1
                else:
                    buckets[h]['losses'] += 1
            return buckets
        
        peak_buckets = group_by_hour(peak_report.get('trades_data', []))
        off_buckets = group_by_hour(off_report.get('trades_data', []))
        
        hours = list(range(8, 18))
        peak_win_rates = [(peak_buckets[h]['wins'] / max(peak_buckets[h]['count'], 1)) if peak_buckets[h]['count'] > 0 else 0 for h in hours]
        off_win_rates = [(off_buckets[h]['wins'] / max(off_buckets[h]['count'], 1)) if off_buckets[h]['count'] > 0 else 0 for h in hours]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(hours))
        plt.bar(x - 0.2, peak_win_rates, width=0.4, label='Horas pico (Win Rate)')
        plt.bar(x + 0.2, off_win_rates, width=0.4, label='Fuera de pico (Win Rate)')
        plt.xticks(x, [f"{h:02d}:00" for h in hours], rotation=45)
        plt.ylabel('Win Rate')
        plt.title('Comparativa horaria de Win Rate (08:00–17:00)')
        plt.legend()
        # Anotación si no hay datos
        if sum(peak_buckets[h]['count'] for h in hours) == 0 and sum(off_buckets[h]['count'] for h in hours) == 0:
            plt.text(0.5, 0.5, 'Sin operaciones en horas seleccionadas', ha='center', va='center', transform=plt.gca().transAxes)
            self.logger.warning("⚠️ Gráfico horario sin datos (no hubo trades en 08:00–17:00)")
        plt.tight_layout()
        plt.savefig(filename)
        self.logger.info(f"📈 Gráfico horario guardado en {filename}")
        return filename

def run_performance_analysis():
    """
    Ejecuta análisis completo de rendimiento
    """
    print("📊 Iniciando análisis comparativo de rendimiento...")
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    analyzer = PerformanceAnalyzer()
    
    # Ejecutar análisis baseline
    print("\n1️⃣ Ejecutando análisis baseline...")
    baseline = analyzer.run_baseline_analysis(days=30)
    
    if 'error' in baseline:
        print(f"❌ Error en análisis baseline: {baseline['error']}")
        return
    
    print("✅ Análisis baseline completado")
    
    # Ejecutar análisis optimizado
    print("\n2️⃣ Ejecutando análisis optimizado...")
    optimized = analyzer.run_optimized_analysis(days=30)
    
    if 'error' in optimized:
        print(f"❌ Error en análisis optimizado: {optimized['error']}")
        return
    
    print("✅ Análisis optimizado completado")
    
    # Generar reporte comparativo
    print("\n3️⃣ Generando reporte comparativo...")
    comparison_report = analyzer.generate_comparison_report()
    print(comparison_report)
    
    # Guardar datos
    filename = analyzer.save_comparison_data()
    print(f"📁 Análisis guardado en: {filename}")
    
    # Extra: análisis horario 08:00–17:00
    print("\n4️⃣ Análisis horario (08:00–17:00 vs fuera de pico)...")
    hourly = analyzer.run_hourly_volatility_analysis(days=7)
    chart_file = analyzer.create_hourly_performance_charts(hourly['peak'], hourly['off'])
    print(f"📈 Gráfico horario guardado: {chart_file}")
    return analyzer

if __name__ == '__main__':
    run_performance_analysis()