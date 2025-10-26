#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Herramienta de Comparación de Rendimiento
========================================
Compara el rendimiento actual del bot con expectativas realistas
y proporciona análisis de brechas de rendimiento.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PerformanceComparison")

@dataclass
class PerformanceGap:
    """Representa una brecha de rendimiento"""
    metric_name: str
    current_value: float
    expected_value: float
    gap_percentage: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendations: List[str]

class PerformanceComparisonTool:
    """
    Herramienta para comparar rendimiento actual vs esperado
    """
    
    def __init__(self):
        self.realistic_expectations = {
            'win_rate': {
                'excellent': 0.65,
                'good': 0.55,
                'acceptable': 0.45,
                'poor': 0.35
            },
            'monthly_return': {
                'excellent': 0.15,
                'good': 0.08,
                'acceptable': 0.03,
                'poor': -0.05
            },
            'max_drawdown': {
                'excellent': 0.05,
                'good': 0.10,
                'acceptable': 0.15,
                'poor': 0.25
            },
            'profit_factor': {
                'excellent': 2.0,
                'good': 1.5,
                'acceptable': 1.2,
                'poor': 1.0
            }
        }
    
    def analyze_performance_gaps(self, current_results: Dict) -> List[PerformanceGap]:
        """Analiza brechas entre rendimiento actual y esperado"""
        gaps = []
        
        # Analizar Win Rate
        current_wr = current_results.get('win_rate', 0)
        expected_wr = self.realistic_expectations['win_rate']['good']
        
        if current_wr < expected_wr:
            gap_pct = ((expected_wr - current_wr) / expected_wr) * 100
            severity = self._determine_severity(gap_pct)
            
            recommendations = self._get_win_rate_recommendations(current_wr, expected_wr)
            
            gaps.append(PerformanceGap(
                metric_name="Win Rate",
                current_value=current_wr,
                expected_value=expected_wr,
                gap_percentage=gap_pct,
                severity=severity,
                recommendations=recommendations
            ))
        
        # Analizar Return
        current_return = current_results.get('total_return', 0)
        expected_return = self.realistic_expectations['monthly_return']['good']
        
        if current_return < expected_return:
            gap_pct = ((expected_return - current_return) / expected_return) * 100
            severity = self._determine_severity(gap_pct)
            
            recommendations = self._get_return_recommendations(current_return, expected_return)
            
            gaps.append(PerformanceGap(
                metric_name="Monthly Return",
                current_value=current_return,
                expected_value=expected_return,
                gap_percentage=gap_pct,
                severity=severity,
                recommendations=recommendations
            ))
        
        # Analizar Drawdown
        current_dd = current_results.get('max_drawdown', 0)
        expected_dd = self.realistic_expectations['max_drawdown']['good']
        
        if current_dd > expected_dd:
            gap_pct = ((current_dd - expected_dd) / expected_dd) * 100
            severity = self._determine_severity(gap_pct)
            
            recommendations = self._get_drawdown_recommendations(current_dd, expected_dd)
            
            gaps.append(PerformanceGap(
                metric_name="Max Drawdown",
                current_value=current_dd,
                expected_value=expected_dd,
                gap_percentage=gap_pct,
                severity=severity,
                recommendations=recommendations
            ))
        
        return gaps
    
    def _determine_severity(self, gap_percentage: float) -> str:
        """Determina la severidad de la brecha"""
        if gap_percentage <= 10:
            return 'low'
        elif gap_percentage <= 25:
            return 'medium'
        elif gap_percentage <= 50:
            return 'high'
        else:
            return 'critical'
    
    def _get_win_rate_recommendations(self, current: float, expected: float) -> List[str]:
        """Recomendaciones para mejorar win rate"""
        recommendations = []
        
        if current < 0.4:
            recommendations.extend([
                "Revisar filtros de entrada - demasiadas señales falsas",
                "Aumentar min_confidence a 0.8 o superior",
                "Implementar filtros adicionales (ADX, Bollinger, RSI)",
                "Evaluar horarios de trading - evitar baja liquidez"
            ])
        elif current < 0.5:
            recommendations.extend([
                "Ajustar parámetros de EMAs para mejor timing",
                "Optimizar niveles de TP/SL",
                "Mejorar análisis de confirmaciones"
            ])
        else:
            recommendations.extend([
                "Ajustes finos en parámetros existentes",
                "Monitorear consistencia temporal"
            ])
        
        return recommendations
    
    def _get_return_recommendations(self, current: float, expected: float) -> List[str]:
        """Recomendaciones para mejorar retornos"""
        recommendations = []
        
        if current < 0:
            recommendations.extend([
                "Revisar gestión de riesgo - pérdidas excesivas",
                "Reducir tamaño de posición hasta estabilizar",
                "Considerar pausa temporal para reoptimización"
            ])
        elif current < 0.03:
            recommendations.extend([
                "Aumentar take_profit_multiplier gradualmente",
                "Optimizar ratio riesgo/recompensa",
                "Buscar más oportunidades de trading"
            ])
        else:
            recommendations.extend([
                "Mantener estrategia actual",
                "Monitorear sostenibilidad"
            ])
        
        return recommendations
    
    def _get_drawdown_recommendations(self, current: float, expected: float) -> List[str]:
        """Recomendaciones para controlar drawdown"""
        recommendations = []
        
        if current > 0.2:
            recommendations.extend([
                "Implementar stop-loss más estricto",
                "Reducir significativamente el tamaño de posición",
                "Activar modo conservador inmediatamente"
            ])
        elif current > 0.15:
            recommendations.extend([
                "Mejorar gestión de riesgo",
                "Implementar trailing stops",
                "Reducir exposición en mercados volátiles"
            ])
        else:
            recommendations.extend([
                "Mantener gestión de riesgo actual",
                "Monitorear tendencias"
            ])
        
        return recommendations
    
    def generate_comparison_report(self, current_results: Dict, gaps: List[PerformanceGap]):
        """Genera reporte de comparación visual"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Analisis de Brechas de Rendimiento', fontsize=16, fontweight='bold')
        
        # 1. Comparación de métricas clave
        metrics = ['win_rate', 'total_return', 'max_drawdown', 'profit_factor']
        current_values = [current_results.get(m, 0) for m in metrics]
        expected_values = [
            self.realistic_expectations['win_rate']['good'],
            self.realistic_expectations['monthly_return']['good'],
            self.realistic_expectations['max_drawdown']['good'],
            self.realistic_expectations['profit_factor']['good']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, current_values, width, label='Actual', color='lightcoral')
        axes[0, 0].bar(x + width/2, expected_values, width, label='Esperado', color='lightgreen')
        axes[0, 0].set_title('Comparacion Actual vs Esperado')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(['Win Rate', 'Return', 'Drawdown', 'Profit Factor'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Severidad de brechas
        if gaps:
            gap_names = [g.metric_name for g in gaps]
            gap_percentages = [g.gap_percentage for g in gaps]
            colors = ['red' if g.severity == 'critical' else 
                     'orange' if g.severity == 'high' else
                     'yellow' if g.severity == 'medium' else 'lightblue' 
                     for g in gaps]
            
            axes[0, 1].barh(gap_names, gap_percentages, color=colors)
            axes[0, 1].set_title('Brechas de Rendimiento (%)')
            axes[0, 1].set_xlabel('Brecha Porcentual')
        
        # 3. Evolución temporal (simulada)
        weeks = list(range(1, 5))
        current_trend = [current_results.get('win_rate', 0.4) + np.random.normal(0, 0.05) for _ in weeks]
        expected_line = [self.realistic_expectations['win_rate']['good']] * len(weeks)
        
        axes[1, 0].plot(weeks, current_trend, marker='o', label='Win Rate Actual', color='red')
        axes[1, 0].plot(weeks, expected_line, '--', label='Win Rate Esperado', color='green')
        axes[1, 0].set_title('Tendencia Win Rate (4 semanas)')
        axes[1, 0].set_xlabel('Semana')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Resumen de recomendaciones
        all_recommendations = []
        for gap in gaps:
            all_recommendations.extend(gap.recommendations[:2])  # Top 2 por gap
        
        axes[1, 1].text(0.05, 0.95, 'Recomendaciones Prioritarias:', 
                       transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
        
        for i, rec in enumerate(all_recommendations[:6]):  # Top 6 total
            axes[1, 1].text(0.05, 0.85 - i*0.12, f"• {rec}", 
                           transform=axes[1, 1].transAxes, fontsize=9)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Guardar reporte
        filename = f"performance_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Reporte de comparacion guardado: {filename}")
        
        return filename

def main():
    """Función principal para ejecutar comparación de rendimiento"""
    
    # Simular resultados actuales (en producción vendrían de run_m1_optimized.py)
    current_results = {
        'win_rate': 0.477,  # Del terminal: weighted_win_rate
        'total_return': 0.024,  # Calculado aproximado
        'max_drawdown': 0.0398,  # Del terminal
        'profit_factor': 1.24,  # Del terminal
        'total_trades': 44
    }
    
    # Crear herramienta de comparación
    comparison_tool = PerformanceComparisonTool()
    
    # Analizar brechas
    gaps = comparison_tool.analyze_performance_gaps(current_results)
    
    # Generar reporte
    report_file = comparison_tool.generate_comparison_report(current_results, gaps)
    
    # Mostrar análisis en consola
    print("\n" + "="*80)
    print("ANALISIS DE BRECHAS DE RENDIMIENTO")
    print("="*80)
    
    print(f"\nMETRICAS ACTUALES:")
    print(f"   Win Rate: {current_results['win_rate']:.1%}")
    print(f"   Return Mensual: {current_results['total_return']:.1%}")
    print(f"   Max Drawdown: {current_results['max_drawdown']:.1%}")
    print(f"   Profit Factor: {current_results['profit_factor']:.2f}")
    
    if gaps:
        print(f"\nBRECHAS IDENTIFICADAS:")
        for gap in gaps:
            severity_icon = "[CRITICO]" if gap.severity == 'critical' else "[ALTO]" if gap.severity == 'high' else "[MEDIO]"
            print(f"   {severity_icon} {gap.metric_name}: {gap.gap_percentage:.1f}% de brecha ({gap.severity})")
            print(f"      Actual: {gap.current_value:.1%} | Esperado: {gap.expected_value:.1%}")
        
        print(f"\nRECOMENDACIONES PRIORITARIAS:")
        priority_recs = []
        for gap in sorted(gaps, key=lambda x: x.gap_percentage, reverse=True):
            priority_recs.extend(gap.recommendations[:2])
        
        for i, rec in enumerate(priority_recs[:5], 1):
            print(f"   {i}. {rec}")
    else:
        print(f"\nNo se detectaron brechas significativas de rendimiento")
    
    print(f"\nReporte visual generado: {report_file}")
    print("="*80)

if __name__ == "__main__":
    main()