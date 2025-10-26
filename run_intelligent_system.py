#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Trading Inteligente con Evaluación de Oportunidades
-------------------------------------------------------------
Este script ejecuta el bot de trading con el nuevo sistema que opera
basado en oportunidades de mercado en lugar de restricciones temporales.

Características principales:
- Análisis técnico en tiempo real
- Evaluación de condiciones de mercado
- Análisis de riesgo/recompensa
- Eliminación de restricciones temporales fijas
- Gestión inteligente de capital

Autor: Trading Bot Team
Versión: 2.0
"""

import sys
import os
import time
import json
from datetime import datetime

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.trading_bot import ScalpingBot
from core.risk_manager import IntelligentRiskManager
from config.settings import SCALPING_CONFIG

def display_system_info():
    """Muestra información del sistema inteligente de oportunidades"""
    print("🤖 SISTEMA DE TRADING INTELIGENTE CON OPORTUNIDADES")
    print("=" * 60)
    print("🎯 CARACTERÍSTICAS PRINCIPALES:")
    print("   ✅ Análisis técnico en tiempo real (EMA, RSI, Bollinger)")
    print("   ✅ Evaluación de condiciones de mercado (volatilidad, spread)")
    print("   ✅ Análisis de riesgo/recompensa dinámico")
    print("   ✅ Eliminación de restricciones temporales fijas")
    print("   ✅ Gestión inteligente de capital (2% riesgo por operación)")
    print("   ✅ Stop loss y take profit adaptativos")
    print("   ✅ Monitoreo continuo de posiciones")
    print("   ✅ Escalado parcial de posiciones")
    print("   ✅ Sistema de alertas inteligentes")
    print()
    print("🔍 CRITERIOS DE OPORTUNIDAD:")
    print("   • Fuerza de señal técnica > 0.6")
    print("   • Condiciones de mercado favorables")
    print("   • Ratio riesgo/recompensa > 1.5:1")
    print("   • Confianza general > 60%")
    print()
    print("⚡ VENTAJAS DEL NUEVO SISTEMA:")
    print("   • Opera solo en las mejores oportunidades")
    print("   • Maximiza la eficiencia del capital")
    print("   • Reduce operaciones innecesarias")
    print("   • Mejora la precisión de entrada")
    print("   • Adapta estrategia a condiciones de mercado")
    print("=" * 60)
    print()

def run_opportunity_system():
    """Ejecuta el sistema de trading basado en oportunidades"""
    
    display_system_info()
    
    print("🚀 INICIANDO SISTEMA DE OPORTUNIDADES INTELIGENTE")
    print("-" * 50)
    
    try:
        from config.settings import SCALPING_CONFIG, STRATEGY_CONFIG
        bot = ScalpingBot({
            'scalping': SCALPING_CONFIG.copy(),
            'strategy': STRATEGY_CONFIG.copy()
        })

        # Inicializar antes de mostrar resumen
        bot.initialize()

        print("✅ Bot inicializado correctamente")
        print("✅ Sistema de oportunidades activado")
        print("✅ Gestión inteligente de capital habilitada")
        print()
        
        # Mostrar configuración inicial
        risk_manager = bot.risk_manager
        
        print("💰 CONFIGURACIÓN DE CAPITAL:")
        summary = risk_manager.get_risk_summary(bot.account_balance)
        print(f"   Capital efectivo: ${summary.get('effective_capital', 50.0):.2f}")
        print(f"   Riesgo máximo por operación: ${summary.get('max_risk_per_trade', 1.0):.2f}")
        print(f"   Límite de drawdown diario: {summary.get('daily_drawdown_pct', 0.0)*100:.2f}%")
        print(f"   Operaciones restantes hoy: {summary.get('remaining_daily_trades', 30)}")
        print()
        
        print("🎯 UMBRALES DE OPORTUNIDAD:")
        print("   Confianza mínima: 60%")
        print("   Puntuación técnica mínima: 0.6")
        print("   Ratio R:R mínimo: 1.5:1")
        print()
        
        print("🔄 INICIANDO EVALUACIÓN CONTINUA DE OPORTUNIDADES...")
        print("   (El bot evaluará el mercado y operará solo en las mejores oportunidades)")
        print("   (Presiona Ctrl+C para detener)")
        print("-" * 50)
        
        # Ejecutar el bot
        bot.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Sistema detenido por el usuario")
        print("📊 Generando resumen final...")
        
        if 'bot' in locals():
            # Mostrar resumen final
            summary = bot.risk_manager.get_risk_summary(bot.account_balance)
            print("\n📈 RESUMEN FINAL DEL SISTEMA:")
            print(f"   Operaciones ejecutadas: {summary.get('trades_today', 0)}")
            print(f"   P&L diario: ${summary.get('daily_pnl', 0.0):.2f}")
            print(f"   Drawdown diario: {summary.get('daily_drawdown_pct', 0.0)*100:.2f}%")
            print(f"   Alertas de riesgo hoy: {summary.get('risk_alerts_today', 0)}")
        
        print("✅ Sistema finalizado correctamente")
        
    except Exception as e:
        print(f"❌ Error en el sistema: {e}")
        print("🔧 Revisa la configuración y conexiones")
        return False
    
    return True

def main():
    """Función principal"""
    
    print("🎯 SISTEMA DE TRADING CON OPORTUNIDADES INTELIGENTES")
    print("=" * 60)
    print()
    
    while True:
        print("Selecciona una opción:")
        print("1. 🧪 Ejecutar pruebas del sistema")
        print("2. 🚀 Iniciar trading con oportunidades inteligentes")
        print("3. 📊 Ver configuración del sistema")
        print("4. ❌ Salir")
        print()
        
        choice = input("Ingresa tu opción (1-4): ").strip()
        
        if choice == "1":
            print("\n🧪 EJECUTANDO PRUEBAS DEL SISTEMA...")
            print("-" * 40)
            
            # Importar y ejecutar pruebas
            try:
                from test_intelligent_capital import test_market_opportunity_system
                test_market_opportunity_system()
                
                print("\n✅ Pruebas completadas exitosamente")
                input("\nPresiona Enter para continuar...")
                
            except Exception as e:
                print(f"❌ Error en las pruebas: {e}")
                input("\nPresiona Enter para continuar...")
        
        elif choice == "2":
            print("\n🚀 INICIANDO SISTEMA DE TRADING...")
            print("-" * 40)
            
            confirm = input("¿Confirmas iniciar el trading en vivo? (s/N): ").strip().lower()
            if confirm in ['s', 'si', 'sí', 'y', 'yes']:
                run_opportunity_system()
            else:
                print("❌ Operación cancelada")
            
            input("\nPresiona Enter para continuar...")
        
        elif choice == "3":
            print("\n📊 CONFIGURACIÓN DEL SISTEMA:")
            print("-" * 40)
            display_system_info()
            
            # Mostrar configuración técnica
            print("⚙️ CONFIGURACIÓN TÉCNICA:")
            print(f"   Símbolo: {SCALPING_CONFIG.get('symbol', 'EURUSD')}")
            print(f"   Timeframe: {SCALPING_CONFIG.get('timeframe', '1m')}")
            print(f"   Lotes base: {SCALPING_CONFIG.get('base_lot_size', 0.01)}")
            print(f"   Spread máximo: {SCALPING_CONFIG.get('max_spread', 3.0)} pips")
            print()
            
            input("Presiona Enter para continuar...")
        
        elif choice == "4":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")
            input("Presiona Enter para continuar...")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()