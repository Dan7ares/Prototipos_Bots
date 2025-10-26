#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Trading Algorítmico - Módulo Principal
------------------------------------------------
Este módulo inicia y configura el sistema de trading algorítmico,
estableciendo el entorno de ejecución y gestionando el ciclo de vida
del bot de trading con sistema de monitoreo y autoreparación.

Autor: Trading Bot Team
Versión: 2.1
"""

import logging
import sys
import os
import argparse
import MetaTrader5 as mt5
import threading
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler
from core.trading_bot import ScalpingBot
from config.settings import SCALPING_CONFIG, STRATEGY_CONFIG
from core.data_loader import MarketAnalyzer

# Constantes globales
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'scalping_bot.log'
MAINTENANCE_LOG = 'system_maintenance.log'
VERSION = '2.1'

# Configuración del sistema de monitoreo
MONITORING_CONFIG = {
    'check_interval_minutes': 5,  # Intervalo de verificación configurable
    'max_retry_attempts': 3,
    'critical_alert_threshold': 3,
    'log_retention_days': 30,
    'availability_target': 99.9,  # 99.9% disponibilidad
    'transaction_capacity': 1000,  # Transacciones por hora
    'profitability_target': 70.0,  # 70% rentabilidad objetivo
    'initial_capital': 50.0  # Capital inicial objetivo
}

@dataclass
class SystemStatus:
    """Estructura para el estado del sistema"""
    timestamp: str
    mt5_connection: bool
    bot_status: str
    balance: float
    spread: float
    volatility: float
    last_signal: str
    uptime_hours: float
    error_count: int
    warning_count: int
    transactions_hour: int
    profitability: float
    recommendations: List[str]

@dataclass
class HealthCheckResult:
    """Resultado de verificación de salud del sistema"""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: str
    auto_fixed: bool = False

def configurar_logging():
    """
    Configura el sistema de logging avanzado con rotación y múltiples handlers.
    Utiliza codificación UTF-8 para soportar caracteres especiales.
    """
    # Configurar logger principal
    main_handler = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=MONITORING_CONFIG['log_retention_days'],
        encoding='utf-8'
    )
    
    # Configurar logger de mantenimiento
    maintenance_handler = RotatingFileHandler(
        MAINTENANCE_LOG,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=MONITORING_CONFIG['log_retention_days'],
        encoding='utf-8'
    )
    
    # Configurar formato
    formatter = logging.Formatter(LOG_FORMAT)
    main_handler.setFormatter(formatter)
    maintenance_handler.setFormatter(formatter)
    
    # Configurar logger principal
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            main_handler
        ]
    )
    
    # Configurar logger de mantenimiento
    maintenance_logger = logging.getLogger('Maintenance')
    maintenance_logger.addHandler(maintenance_handler)
    maintenance_logger.setLevel(logging.INFO)
    
    return logging.getLogger('Main')

class SystemMonitor:
    """
    Sistema de monitoreo y autoreparación del bot de trading.
    Implementa verificaciones periódicas, diagnósticos y recuperación automática.
    """
    
    def __init__(self, bot_instance: Optional[ScalpingBot] = None):
        self.bot = bot_instance
        self.logger = logging.getLogger('SystemMonitor')
        self.maintenance_logger = logging.getLogger('Maintenance')
        self.start_time = datetime.now()
        self.is_running = False
        self.monitor_thread = None
        self.error_count = 0
        self.warning_count = 0
        self.last_check = None
        self.system_alerts = []
        
    def start_monitoring(self):
        """Inicia el sistema de monitoreo con gestión inteligente de capital"""
        if self.is_running:
            self.logger.warning("El sistema de monitoreo ya está ejecutándose")
            return
        
        self.logger.info("🚀 Iniciando Sistema de Monitoreo Avanzado con Gestión Inteligente")
        self.logger.info("=" * 60)
        
        # Mostrar información del sistema de gestión inteligente
        if self.bot and hasattr(self.bot, 'risk_manager'):
            try:
                balance = getattr(self.bot, 'account_balance', 50.0)
                risk_summary = self.bot.risk_manager.get_risk_summary(balance)
                
                self.logger.info("📊 CONFIGURACIÓN DE GESTIÓN INTELIGENTE:")
                self.logger.info(f"   • Capital efectivo: ${risk_summary.get('effective_capital', 50):.2f}")
                self.logger.info(f"   • Riesgo máximo por operación: ${risk_summary.get('max_risk_per_trade', 1):.2f}")
                self.logger.info(f"   • Límite de drawdown diario: 5%")
                self.logger.info(f"   • Stop Loss dinámico: ~$2.00")
                self.logger.info(f"   • Take Profit inteligente: Ratio 1.5:1 mínimo")
                self.logger.info(f"   • Operaciones máximas diarias: {risk_summary.get('remaining_daily_trades', 30)}")
                self.logger.info("=" * 60)
                
            except Exception as e:
                self.logger.warning(f"No se pudo obtener resumen de riesgo: {e}")
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("✅ Sistema de monitoreo iniciado correctamente")
        self.logger.info(f"🔄 Verificaciones cada {MONITORING_CONFIG['check_interval_minutes']} minutos")
        
    def stop_monitoring(self):
        """Detiene el sistema de monitoreo"""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Sistema de monitoreo detenido")
        
    def _monitoring_loop(self):
        """Bucle principal de monitoreo"""
        while self.is_running:
            try:
                # Realizar verificación completa
                health_results = self._perform_health_check()
                
                # Generar reporte de estado
                status_report = self._generate_status_report(health_results)
                
                # Procesar alertas y autoreparación
                self._process_alerts_and_repair(health_results)
                
                # Guardar reporte
                self._save_status_report(status_report)
                
                self.last_check = datetime.now()
                
                # Esperar hasta la próxima verificación
                time.sleep(MONITORING_CONFIG['check_interval_minutes'] * 60)
                
            except Exception as e:
                self.logger.error(f"Error en bucle de monitoreo: {e}")
                self.maintenance_logger.error(f"Excepción en monitoreo: {traceback.format_exc()}")
                time.sleep(30)  # Espera corta antes de reintentar
                
    def _perform_health_check(self) -> List[HealthCheckResult]:
        """
        Realiza verificaciones completas de salud del sistema
        
        Returns:
            Lista de resultados de verificación
        """
        results = []
        timestamp = datetime.now().isoformat()
        
        # 1. Verificar conexión MT5
        try:
            if not mt5.initialize():
                results.append(HealthCheckResult(
                    component="MT5_Connection",
                    status="critical",
                    message="No se pudo inicializar MT5",
                    timestamp=timestamp
                ))
            else:
                account_info = mt5.account_info()
                if account_info is None:
                    results.append(HealthCheckResult(
                        component="MT5_Connection",
                        status="warning",
                        message="MT5 inicializado pero sin información de cuenta",
                        timestamp=timestamp
                    ))
                else:
                    results.append(HealthCheckResult(
                        component="MT5_Connection",
                        status="healthy",
                        message=f"Conexión MT5 activa - Balance: ${account_info.balance:.2f}",
                        timestamp=timestamp
                    ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="MT5_Connection",
                status="critical",
                message=f"Error verificando MT5: {str(e)}",
                timestamp=timestamp
            ))
        
        # 2. Verificar estado del bot
        try:
            if self.bot is None:
                results.append(HealthCheckResult(
                    component="Trading_Bot",
                    status="warning",
                    message="Bot no inicializado",
                    timestamp=timestamp
                ))
            elif hasattr(self.bot, 'is_running') and self.bot.is_running:
                last_activity = getattr(self.bot, 'last_activity', None)
                if last_activity and (datetime.now() - last_activity).seconds > 300:  # 5 minutos
                    results.append(HealthCheckResult(
                        component="Trading_Bot",
                        status="warning",
                        message="Bot inactivo por más de 5 minutos",
                        timestamp=timestamp
                    ))
                else:
                    results.append(HealthCheckResult(
                        component="Trading_Bot",
                        status="healthy",
                        message="Bot ejecutándose correctamente",
                        timestamp=timestamp
                    ))
            else:
                results.append(HealthCheckResult(
                    component="Trading_Bot",
                    status="warning",
                    message="Bot detenido",
                    timestamp=timestamp
                ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="Trading_Bot",
                status="error",
                message=f"Error verificando bot: {str(e)}",
                timestamp=timestamp
            ))
        
        # 3. Verificar condiciones de mercado
        try:
            symbol = SCALPING_CONFIG.get('symbol', 'EURUSD')
            tick = mt5.symbol_info_tick(symbol) or mt5.symbol_info_tick(f"{symbol}m")
            
            if tick is None:
                results.append(HealthCheckResult(
                    component="Market_Conditions",
                    status="warning",
                    message=f"No se pueden obtener datos de {symbol}",
                    timestamp=timestamp
                ))
            else:
                spread = (tick.ask - tick.bid) / 0.0001
                max_spread = SCALPING_CONFIG.get('max_spread', 2.0)
                
                if spread > max_spread:
                    results.append(HealthCheckResult(
                        component="Market_Conditions",
                        status="warning",
                        message=f"Spread alto: {spread:.1f} pips (límite: {max_spread})",
                        timestamp=timestamp
                    ))
                else:
                    results.append(HealthCheckResult(
                        component="Market_Conditions",
                        status="healthy",
                        message=f"Condiciones normales - Spread: {spread:.1f} pips",
                        timestamp=timestamp
                    ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="Market_Conditions",
                status="error",
                message=f"Error verificando mercado: {str(e)}",
                timestamp=timestamp
            ))
        
        # 4. Verificar integridad de datos
        try:
            config_files = ['config/settings.py']
            missing_files = [f for f in config_files if not os.path.exists(f)]
            
            if missing_files:
                results.append(HealthCheckResult(
                    component="Data_Integrity",
                    status="critical",
                    message=f"Archivos faltantes: {', '.join(missing_files)}",
                    timestamp=timestamp
                ))
            else:
                results.append(HealthCheckResult(
                    component="Data_Integrity",
                    status="healthy",
                    message="Todos los archivos críticos presentes",
                    timestamp=timestamp
                ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="Data_Integrity",
                status="error",
                message=f"Error verificando integridad: {str(e)}",
                timestamp=timestamp
            ))
        
        # 5. Verificar rendimiento del sistema
        try:
            uptime = datetime.now() - self.start_time
            uptime_hours = uptime.total_seconds() / 3600
            
            if uptime_hours < 1:
                status = "healthy"
                message = f"Sistema recién iniciado - Uptime: {uptime_hours:.1f}h"
            elif self.error_count > 10:
                status = "warning"
                message = f"Muchos errores detectados: {self.error_count}"
            else:
                status = "healthy"
                message = f"Rendimiento normal - Uptime: {uptime_hours:.1f}h, Errores: {self.error_count}"
            
            results.append(HealthCheckResult(
                component="System_Performance",
                status=status,
                message=message,
                timestamp=timestamp
            ))
        except Exception as e:
            results.append(HealthCheckResult(
                component="System_Performance",
                status="error",
                message=f"Error verificando rendimiento: {str(e)}",
                timestamp=timestamp
            ))
        
        return results

    def _generate_status_report(self, health_results: List[HealthCheckResult]) -> SystemStatus:
        """
        Genera un reporte completo del estado del sistema
        
        Args:
            health_results: Resultados de las verificaciones de salud
            
        Returns:
            Reporte de estado del sistema
        """
        try:
            timestamp = datetime.now().isoformat()
            uptime = datetime.now() - self.start_time
            
            # Estado de conexión MT5
            mt5_connected = any(r.component == "MT5_Connection" and r.status == "healthy" for r in health_results)
            
            # Estado del bot
            bot_status = "running" if self.bot and hasattr(self.bot, 'is_running') and self.bot.is_running else "stopped"
            
            # Obtener métricas de trading
            balance = 0.0
            spread = 0.0
            volatility = 0.0
            last_signal = "HOLD"
            
            if mt5_connected:
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                
                # Obtener spread actual
                symbol = SCALPING_CONFIG.get('symbol', 'EURUSD')
                tick = mt5.symbol_info_tick(symbol) or mt5.symbol_info_tick(f"{symbol}m")
                if tick:
                    spread = (tick.ask - tick.bid) / 0.0001
            
            # Calcular rentabilidad
            initial_capital = MONITORING_CONFIG['initial_capital']
            profitability = ((balance - initial_capital) / initial_capital * 100) if balance > 0 else 0.0
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(health_results, profitability)
            
            return SystemStatus(
                timestamp=timestamp,
                mt5_connection=mt5_connected,
                bot_status=bot_status,
                balance=balance,
                spread=spread,
                volatility=volatility,
                last_signal=last_signal,
                uptime_hours=uptime.total_seconds() / 3600,
                error_count=self.error_count,
                warning_count=self.warning_count,
                transactions_hour=0,  # TODO: Implementar contador de transacciones
                profitability=profitability,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de estado: {e}")
            return SystemStatus(
                timestamp=datetime.now().isoformat(),
                mt5_connection=False,
                bot_status="error",
                balance=0.0,
                spread=0.0,
                volatility=0.0,
                last_signal="ERROR",
                uptime_hours=0.0,
                error_count=self.error_count + 1,
                warning_count=self.warning_count,
                transactions_hour=0,
                profitability=0.0,
                recommendations=["❌ Error generando reporte del sistema"]
            )

    def _generate_recommendations(self, health_results: List[HealthCheckResult], profitability: float) -> List[str]:
        """
        Genera recomendaciones basadas en el estado del sistema
        
        Args:
            health_results: Resultados de verificaciones de salud
            profitability: Rentabilidad actual del sistema
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Analizar problemas críticos
        critical_issues = [r for r in health_results if r.status == "critical"]
        if critical_issues:
            recommendations.append("🔴 CRÍTICO: Problemas críticos detectados que requieren atención inmediata")
            for issue in critical_issues:
                recommendations.append(f"   • {issue.component}: {issue.message}")
        
        # Analizar warnings
        warning_issues = [r for r in health_results if r.status == "warning"]
        if warning_issues:
            recommendations.append("🟡 ADVERTENCIA: Condiciones subóptimas detectadas")
            for issue in warning_issues:
                recommendations.append(f"   • {issue.component}: {issue.message}")
        
        # Analizar rentabilidad
        target_profitability = MONITORING_CONFIG['profitability_target']
        if profitability < 0:
            recommendations.append(f"📉 RENTABILIDAD: Pérdidas detectadas ({profitability:.1f}%)")
            recommendations.append("   • Revisar estrategia de trading")
            recommendations.append("   • Considerar ajustar parámetros de riesgo")
        elif profitability < target_profitability:
            recommendations.append(f"📊 RENTABILIDAD: Por debajo del objetivo ({profitability:.1f}% vs {target_profitability}%)")
            recommendations.append("   • Optimizar configuración de trading")
        else:
            recommendations.append(f"✅ RENTABILIDAD: Objetivo superado ({profitability:.1f}% vs {target_profitability}%)")
        
        # Recomendaciones generales si todo está bien
        if not critical_issues and not warning_issues and profitability >= target_profitability:
            recommendations.append("✅ Sistema funcionando óptimamente")
            recommendations.append("📈 Todos los indicadores en rangos normales")
            recommendations.append("🎯 Objetivos de rentabilidad cumplidos")
        
        return recommendations

    def _process_alerts_and_repair(self, health_results: List[HealthCheckResult]):
        """
        Procesa alertas y ejecuta mecanismos de autoreparación
        
        Args:
            health_results: Resultados de las verificaciones de salud
        """
        critical_issues = [r for r in health_results if r.status == "critical"]
        warning_issues = [r for r in health_results if r.status == "warning"]
        
        # Manejar problemas críticos
        for issue in critical_issues:
            self._handle_critical_issue(issue)
            self.error_count += 1
        
        # Manejar warnings
        for issue in warning_issues:
            self._handle_warning_issue(issue)
            self.warning_count += 1
        
        # Enviar alertas si hay problemas críticos
        if critical_issues:
            self._send_critical_alert(health_results)

    def _handle_critical_issue(self, result: HealthCheckResult):
        """
        Maneja problemas críticos con autoreparación
        
        Args:
            result: Resultado de verificación crítica
        """
        self.maintenance_logger.critical(f"Problema crítico detectado: {result.component} - {result.message}")
        
        if result.component == "MT5_Connection":
            if self._attempt_mt5_recovery():
                result.auto_fixed = True
                self.maintenance_logger.info("Conexión MT5 recuperada automáticamente")
        elif result.component == "Trading_Bot":
            if self._attempt_bot_restart():
                result.auto_fixed = True
                self.maintenance_logger.info("Bot reiniciado automáticamente")

    def _handle_warning_issue(self, result: HealthCheckResult):
        """
        Maneja problemas de advertencia
        
        Args:
            result: Resultado de verificación con warning
        """
        self.maintenance_logger.warning(f"Advertencia: {result.component} - {result.message}")

    def _attempt_mt5_recovery(self) -> bool:
        """
        Intenta recuperar la conexión con MT5
        
        Returns:
            True si la recuperación fue exitosa
        """
        try:
            # Cerrar conexión actual
            mt5.shutdown()
            time.sleep(2)
            
            # Intentar reconectar
            for attempt in range(MONITORING_CONFIG['max_retry_attempts']):
                if mt5.initialize():
                    account_info = mt5.account_info()
                    if account_info:
                        self.maintenance_logger.info(f"MT5 recuperado en intento {attempt + 1}")
                        return True
                time.sleep(5)
            
            return False
            
        except Exception as e:
            self.maintenance_logger.error(f"Error en recuperación MT5: {e}")
            return False

    def _attempt_bot_restart(self) -> bool:
        """
        Intenta reiniciar el bot de trading
        
        Returns:
            True si el reinicio fue exitoso
        """
        try:
            if self.bot:
                # Detener bot actual
                if hasattr(self.bot, 'stop'):
                    self.bot.stop()
                time.sleep(3)
                
                # Reiniciar bot
                if hasattr(self.bot, 'start'):
                    self.bot.start()
                    time.sleep(2)
                    
                    # Verificar que esté funcionando
                    if hasattr(self.bot, 'is_running') and self.bot.is_running:
                        self.maintenance_logger.info("Bot reiniciado exitosamente")
                        return True
            
            return False
            
        except Exception as e:
            self.maintenance_logger.error(f"Error reiniciando bot: {e}")
            return False

    def _send_critical_alert(self, health_results: List[HealthCheckResult]):
        """
        Envía alertas críticas al administrador
        
        Args:
            health_results: Resultados de verificaciones de salud
        """
        critical_issues = [r for r in health_results if r.status == "critical"]
        
        alert_message = f"ALERTA CRÍTICA - {len(critical_issues)} problemas detectados:\n"
        for issue in critical_issues:
            status = "✅ REPARADO" if issue.auto_fixed else "❌ REQUIERE ATENCIÓN"
            alert_message += f"• {issue.component}: {issue.message} [{status}]\n"
        
        alert_message += f"\nTimestamp: {datetime.now().isoformat()}"
        alert_message += f"\nUptime: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} horas"
        
        # Log de la alerta
        self.maintenance_logger.critical("=" * 50)
        self.maintenance_logger.critical("ALERTA CRÍTICA DEL SISTEMA")
        self.maintenance_logger.critical("=" * 50)
        self.maintenance_logger.critical(alert_message)
        self.maintenance_logger.critical("=" * 50)
        
        # Aquí se podría implementar envío por email, SMS, etc.
        # Por ahora solo se registra en logs

    def _save_status_report(self, status: SystemStatus):
        """
        Guarda el reporte de estado en archivo JSON
        
        Args:
            status: Estado del sistema a guardar
        """
        try:
            # Crear directorio si no existe
            os.makedirs("status_reports", exist_ok=True)
            
            # Nombre del archivo basado en la fecha
            report_date = datetime.now().strftime('%Y%m%d')
            report_file = f"status_reports/status_{report_date}.json"
            
            # Cargar reportes existentes o crear lista nueva
            reports = []
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    reports = json.load(f)
            
            # Agregar nuevo reporte
            reports.append(asdict(status))
            
            # Guardar archivo actualizado
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(reports, f, indent=2, ensure_ascii=False)
            
            self.maintenance_logger.info(f"Reporte guardado: {report_file}")
            
        except Exception as e:
            self.maintenance_logger.error(f"Error guardando reporte: {e}")
        """
        Realiza verificación completa de salud del sistema
        
        Returns:
            Lista de resultados de verificación
        """
        results = []
        timestamp = datetime.now().isoformat()
        
        # 1. Verificar conexión MT5
        mt5_result = self._check_mt5_connection()
        results.append(HealthCheckResult(
            component="MT5_Connection",
            status=mt5_result['status'],
            message=mt5_result['message'],
            timestamp=timestamp,
            auto_fixed=mt5_result.get('auto_fixed', False)
        ))
        
        # 2. Verificar estado del bot
        if self.bot:
            bot_result = self._check_bot_status()
            results.append(HealthCheckResult(
                component="Trading_Bot",
                status=bot_result['status'],
                message=bot_result['message'],
                timestamp=timestamp,
                auto_fixed=bot_result.get('auto_fixed', False)
            ))
        
        # 3. Verificar condiciones de mercado
        market_result = self._check_market_conditions()
        results.append(HealthCheckResult(
            component="Market_Conditions",
            status=market_result['status'],
            message=market_result['message'],
            timestamp=timestamp
        ))
        
        # 4. Verificar integridad de datos
        data_result = self._check_data_integrity()
        results.append(HealthCheckResult(
            component="Data_Integrity",
            status=data_result['status'],
            message=data_result['message'],
            timestamp=timestamp,
            auto_fixed=data_result.get('auto_fixed', False)
        ))
        
        # 5. Verificar rendimiento del sistema
        performance_result = self._check_system_performance()
        results.append(HealthCheckResult(
            component="System_Performance",
            status=performance_result['status'],
            message=performance_result['message'],
            timestamp=timestamp
        ))
        
        return results
        
    def _check_mt5_connection(self) -> Dict:
        """Verifica la conexión con MetaTrader 5"""
        try:
            if not mt5.initialize():
                # Intentar reconexión automática
                self.logger.warning("Conexión MT5 perdida, intentando reconectar...")
                
                for attempt in range(MONITORING_CONFIG['max_retry_attempts']):
                    time.sleep(2 ** attempt)  # Backoff exponencial
                    if mt5.initialize():
                        self.maintenance_logger.info(f"Reconexión MT5 exitosa en intento {attempt + 1}")
                        return {
                            'status': 'healthy',
                            'message': f'Conexión MT5 restaurada (intento {attempt + 1})',
                            'auto_fixed': True
                        }
                
                return {
                    'status': 'critical',
                    'message': 'No se pudo restablecer conexión MT5 después de múltiples intentos'
                }
            
            # Verificar información de la cuenta
            account_info = mt5.account_info()
            if account_info is None:
                return {
                    'status': 'warning',
                    'message': 'Conexión MT5 activa pero sin información de cuenta'
                }
            
            return {
                'status': 'healthy',
                'message': f'Conexión MT5 activa - Balance: ${account_info.balance:.2f}'
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Error verificando conexión MT5: {str(e)}'
            }
            
    def _check_bot_status(self) -> Dict:
        """Verifica el estado del bot de trading"""
        try:
            if not hasattr(self.bot, 'is_running') or not self.bot.is_running:
                return {
                    'status': 'warning',
                    'message': 'Bot de trading no está ejecutándose'
                }
            
            # Verificar si el bot está respondiendo
            if hasattr(self.bot, 'last_activity'):
                time_since_activity = datetime.now() - self.bot.last_activity
                if time_since_activity > timedelta(minutes=10):
                    return {
                        'status': 'warning',
                        'message': f'Bot inactivo por {time_since_activity.total_seconds()/60:.1f} minutos'
                    }
            
            return {
                'status': 'healthy',
                'message': 'Bot de trading funcionando correctamente'
            }
            
        except Exception as e:
            return {
                'status': 'critical',
                'message': f'Error verificando estado del bot: {str(e)}'
            }
            
    def _check_market_conditions(self) -> Dict:
        """Verifica las condiciones actuales del mercado"""
        try:
            # Obtener datos de mercado
            symbol = SCALPING_CONFIG.get('symbol', 'EURUSD')
            
            # Verificar spread
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                # Intentar con símbolo alternativo
                alt_symbol = f"{symbol}m"
                tick = mt5.symbol_info_tick(alt_symbol)
                
            if tick is None:
                return {
                    'status': 'warning',
                    'message': f'No se pueden obtener datos de mercado para {symbol}'
                }
            
            spread = (tick.ask - tick.bid) / 0.0001
            max_spread = SCALPING_CONFIG.get('max_spread', 1.5)
            
            if spread > max_spread * 2:
                return {
                    'status': 'warning',
                    'message': f'Spread muy alto: {spread:.1f} pips (normal: <{max_spread} pips)'
                }
            
            return {
                'status': 'healthy',
                'message': f'Condiciones de mercado normales - Spread: {spread:.1f} pips'
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Error verificando condiciones de mercado: {str(e)}'
            }
            
    def _check_data_integrity(self) -> Dict:
        """Verifica la integridad de los datos críticos"""
        try:
            # Verificar archivos de configuración
            config_files = ['config/settings.py']
            missing_files = []
            
            for file_path in config_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                return {
                    'status': 'critical',
                    'message': f'Archivos de configuración faltantes: {missing_files}'
                }
            
            # Verificar logs
            if not os.path.exists(LOG_FILE):
                # Crear archivo de log si no existe
                with open(LOG_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"# Log iniciado: {datetime.now().isoformat()}\n")
                
                return {
                    'status': 'healthy',
                    'message': 'Archivo de log recreado automáticamente',
                    'auto_fixed': True
                }
            
            return {
                'status': 'healthy',
                'message': 'Integridad de datos verificada correctamente'
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Error verificando integridad de datos: {str(e)}'
            }
            
    def _check_system_performance(self) -> Dict:
        """Verifica el rendimiento del sistema"""
        try:
            uptime = datetime.now() - self.start_time
            uptime_hours = uptime.total_seconds() / 3600
            
            # Calcular disponibilidad
            target_availability = MONITORING_CONFIG['availability_target']
            
            # Verificar capacidad de transacciones (simulado)
            transaction_capacity = MONITORING_CONFIG['transaction_capacity']
            
            performance_issues = []
            
            if uptime_hours > 24:
                availability = ((uptime_hours * 60 - self.error_count * 5) / (uptime_hours * 60)) * 100
                if availability < target_availability:
                    performance_issues.append(f'Disponibilidad: {availability:.1f}% (objetivo: {target_availability}%)')
            
            if self.error_count > 10:
                performance_issues.append(f'Errores elevados: {self.error_count} errores')
            
            if performance_issues:
                return {
                    'status': 'warning',
                    'message': f'Problemas de rendimiento: {"; ".join(performance_issues)}'
                }
            
            return {
                'status': 'healthy',
                'message': f'Rendimiento óptimo - Uptime: {uptime_hours:.1f}h, Errores: {self.error_count}'
            }
            
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Error verificando rendimiento: {str(e)}'
            }

def analizar_mercados(symbols, timeframes):
    """
    Analiza múltiples mercados para determinar el óptimo para operar.
    
    Args:
        symbols (list): Lista de símbolos a analizar
        timeframes (list): Lista de timeframes a analizar
        
    Returns:
        dict: Resultados del análisis con puntuaciones
    """
    logger = logging.getLogger('MarketAnalysis')
    logger.info(f"Analizando {len(symbols)} mercados en {len(timeframes)} timeframes")
    
    analyzer = MarketAnalyzer()
    results = analyzer.compare_markets(symbols, timeframes)
    
    # Encontrar el mejor mercado
    best_symbol = max(results['scores'], key=lambda x: results['scores'][x]['total'])
    best_score = results['scores'][best_symbol]['total']
    
    logger.info(f"Mejor mercado: {best_symbol} (Puntuación: {best_score:.2f}/20)")
    
    return {
        'best_symbol': best_symbol,
        'best_score': best_score,
        'full_analysis': results
    }

def mostrar_banner(symbol, timeframe):
    """
    Muestra un banner informativo con los detalles del bot.
    
    Args:
        symbol (str): Símbolo a operar
        timeframe (str): Timeframe a utilizar
    """
    print("=" * 50)
    print(f"SCALPING BOT v{VERSION} - {symbol} {timeframe}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d')}")
    print("ESTRATEGIA: EMA + RSI + Bollinger Bands + Patrones")
    print("=" * 50)

def iniciar_bot(config, auto_start=False, enable_monitoring=True):
    """
    Inicia el bot de trading con la configuración proporcionada y sistema de monitoreo.
    
    Args:
        config (dict): Configuración del bot
        auto_start (bool): Si es True, inicia el bot automáticamente
        enable_monitoring (bool): Si es True, activa el sistema de monitoreo
        
    Returns:
        bool: True si el bot se inició correctamente, False en caso contrario
    """
    logger = logging.getLogger('Main')
    monitor = None
    
    # Crear instancia del bot
    bot = ScalpingBot(config)
    
    try:
        # Inicializar sistema de monitoreo si está habilitado
        if enable_monitoring:
            monitor = SystemMonitor(bot)
            monitor.start_monitoring()
            logger.info("Sistema de monitoreo activado")
        
        if auto_start:
            logger.info("Iniciando bot automáticamente con monitoreo avanzado")
            bot.run()
            return True
        else:
            confirm = input("Iniciar Scalping Bot con Sistema de Monitoreo Avanzado? (s/n): ")
            if confirm.lower() == 's':
                bot.run()
                return True
            else:
                print("Ejecución cancelada")
                return False
                
    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        return False
    finally:
        # Detener monitoreo y bot
        if monitor:
            monitor.stop_monitoring()
        bot.shutdown()

def parse_arguments():
    """
    Procesa los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos procesados
    """
    parser = argparse.ArgumentParser(description='Sistema de Trading Algorítmico v2.1')
    parser.add_argument('--auto', action='store_true', help='Iniciar bot automáticamente')
    parser.add_argument('--analyze', action='store_true', help='Analizar mercados antes de iniciar')
    parser.add_argument('--symbol', type=str, default=None, help='Símbolo a operar')
    parser.add_argument('--timeframe', type=str, default=None, help='Timeframe a utilizar')
    parser.add_argument('--no-monitor', action='store_true', help='Desactivar sistema de monitoreo')
    parser.add_argument('--monitor-interval', type=int, default=5, help='Intervalo de monitoreo en minutos')
    
    return parser.parse_args()

def main():
    """
    Función principal que coordina la ejecución del sistema con monitoreo avanzado.
    """
    # Configurar logging avanzado
    logger = configurar_logging()
    logger.info(f"Iniciando Sistema de Trading Algorítmico v{VERSION}")
    
    # Procesar argumentos
    args = parse_arguments()
    
    # Configurar intervalo de monitoreo si se especifica
    if args.monitor_interval:
        MONITORING_CONFIG['check_interval_minutes'] = args.monitor_interval
        logger.info(f"Intervalo de monitoreo configurado: {args.monitor_interval} minutos")
    
    # Configuración inicial
    config = {
        'scalping': SCALPING_CONFIG.copy(),
        'strategy': STRATEGY_CONFIG.copy()
    }
    
    # Analizar mercados si se solicita
    if args.analyze:
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP']
        timeframes = ['M1', 'M5']
        
        analysis = analizar_mercados(symbols, timeframes)
        
        # Usar el mejor símbolo si no se especificó uno
        if not args.symbol:
            config['scalping']['symbol'] = analysis['best_symbol']
            logger.info(f"Usando el mejor símbolo según análisis: {analysis['best_symbol']}")
    
    # Sobrescribir con argumentos de línea de comandos
    if args.symbol:
        config['scalping']['symbol'] = args.symbol
    
    if args.timeframe:
        config['scalping']['timeframe'] = args.timeframe
    
    # Mostrar banner con información de monitoreo
    mostrar_banner_avanzado(config['scalping']['symbol'], config['scalping']['timeframe'])
    
    # Iniciar bot con o sin monitoreo
    enable_monitoring = not args.no_monitor
    iniciar_bot(config, args.auto, enable_monitoring)

def mostrar_banner_avanzado(symbol, timeframe):
    """
    Muestra un banner informativo avanzado con los detalles del bot y monitoreo.
    
    Args:
        symbol (str): Símbolo a operar
        timeframe (str): Timeframe a utilizar
    """
    print("=" * 60)
    print(f"SCALPING BOT v{VERSION} - {symbol} {timeframe}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("ESTRATEGIA: EMA + RSI + Bollinger Bands + Patrones")
    print("-" * 60)
    print("🔍 SISTEMA DE MONITOREO AVANZADO ACTIVADO")
    print(f"📊 Objetivo de Rentabilidad: {MONITORING_CONFIG['profitability_target']}%")
    print(f"💰 Capital Inicial Objetivo: ${MONITORING_CONFIG['initial_capital']}")
    print(f"⏱️  Intervalo de Verificación: {MONITORING_CONFIG['check_interval_minutes']} minutos")
    print(f"🎯 Disponibilidad Objetivo: {MONITORING_CONFIG['availability_target']}%")
    print(f"⚡ Capacidad: {MONITORING_CONFIG['transaction_capacity']} transacciones/hora")
    print("-" * 60)
    print("🛡️  CARACTERÍSTICAS DE AUTOREPARACIÓN:")
    print("   • Reconexión automática MT5")
    print("   • Reinicio inteligente de componentes")
    print("   • Alertas críticas en tiempo real")
    print("   • Reportes de estado automatizados")
    print("   • Gestión de logs con rotación")
    print("=" * 60)

    def _generate_status_report(self, health_results: List[HealthCheckResult]) -> SystemStatus:
        """
        Genera un reporte completo del estado del sistema
        
        Args:
            health_results: Resultados de las verificaciones de salud
            
        Returns:
            Reporte de estado del sistema
        """
        try:
            timestamp = datetime.now().isoformat()
            uptime = datetime.now() - self.start_time
            
            # Estado de conexión MT5
            mt5_connected = any(r.component == "MT5_Connection" and r.status == "healthy" for r in health_results)
            
            # Estado del bot
            bot_status = "running" if self.bot and hasattr(self.bot, 'is_running') and self.bot.is_running else "stopped"
            
            # Obtener métricas de trading
            balance = 0.0
            spread = 0.0
            volatility = 0.0
            last_signal = "HOLD"
            
            if mt5_connected:
                account_info = mt5.account_info()
                if account_info:
                    balance = account_info.balance
                
                # Obtener spread actual
                symbol = SCALPING_CONFIG.get('symbol', 'EURUSD')
                tick = mt5.symbol_info_tick(symbol) or mt5.symbol_info_tick(f"{symbol}m")
                if tick:
                    spread = (tick.ask - tick.bid) / 0.0001
            
            # Calcular rentabilidad
            initial_capital = MONITORING_CONFIG['initial_capital']
            profitability = ((balance - initial_capital) / initial_capital * 100) if balance > 0 else 0.0
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(health_results, profitability)
            
            return SystemStatus(
                timestamp=timestamp,
                mt5_connection=mt5_connected,
                bot_status=bot_status,
                balance=balance,
                spread=spread,
                volatility=volatility,
                last_signal=last_signal,
                uptime_hours=uptime.total_seconds() / 3600,
                error_count=self.error_count,
                warning_count=self.warning_count,
                transactions_hour=0,  # Se actualizaría con datos reales
                profitability=profitability,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de estado: {e}")
            return SystemStatus(
                timestamp=datetime.now().isoformat(),
                mt5_connection=False,
                bot_status="error",
                balance=0.0,
                spread=0.0,
                volatility=0.0,
                last_signal="ERROR",
                uptime_hours=0.0,
                error_count=self.error_count + 1,
                warning_count=self.warning_count,
                transactions_hour=0,
                profitability=0.0,
                recommendations=["Sistema en estado de error - Revisar logs"]
            )
            
    def _generate_recommendations(self, health_results: List[HealthCheckResult], profitability: float) -> List[str]:
        """
        Genera recomendaciones específicas basadas en el estado del sistema
        
        Args:
            health_results: Resultados de verificaciones de salud
            profitability: Rentabilidad actual del sistema
            
        Returns:
            Lista de recomendaciones de acción
        """
        recommendations = []
        
        # Analizar resultados de salud
        critical_issues = [r for r in health_results if r.status == "critical"]
        warning_issues = [r for r in health_results if r.status == "warning"]
        
        # Recomendaciones para problemas críticos
        for issue in critical_issues:
            if issue.component == "MT5_Connection":
                recommendations.append("🔴 CRÍTICO: Restablecer conexión MT5 - Verificar credenciales y red")
            elif issue.component == "Trading_Bot":
                recommendations.append("🔴 CRÍTICO: Reiniciar bot de trading - Verificar configuración")
            elif issue.component == "Data_Integrity":
                recommendations.append("🔴 CRÍTICO: Restaurar archivos de configuración desde backup")
        
        # Recomendaciones para advertencias
        for issue in warning_issues:
            if issue.component == "Market_Conditions":
                recommendations.append("🟡 ADVERTENCIA: Condiciones de mercado adversas - Considerar pausa temporal")
            elif issue.component == "System_Performance":
                recommendations.append("🟡 ADVERTENCIA: Rendimiento degradado - Revisar recursos del sistema")
        
        # Recomendaciones basadas en rentabilidad
        target_profitability = MONITORING_CONFIG['profitability_target']
        
        if profitability < 0:
            recommendations.append("📉 RENTABILIDAD: Pérdidas detectadas - Revisar estrategia y gestión de riesgo")
        elif profitability < target_profitability / 2:
            recommendations.append("📊 RENTABILIDAD: Por debajo del objetivo - Optimizar parámetros de trading")
        elif profitability >= target_profitability:
            recommendations.append("✅ RENTABILIDAD: Objetivo alcanzado - Mantener configuración actual")
        
        # Recomendaciones generales
        if not recommendations:
            recommendations.append("✅ SISTEMA: Funcionamiento óptimo - No se requieren acciones")
        
        return recommendations
        
    def _process_alerts_and_repair(self, health_results: List[HealthCheckResult]):
        """
        Procesa alertas y ejecuta acciones de autoreparación
        
        Args:
            health_results: Resultados de las verificaciones de salud
        """
        critical_count = sum(1 for r in health_results if r.status == "critical")
        warning_count = sum(1 for r in health_results if r.status == "warning")
        
        # Actualizar contadores
        self.error_count += critical_count
        self.warning_count += warning_count
        
        # Procesar problemas críticos
        for result in health_results:
            if result.status == "critical":
                self._handle_critical_issue(result)
            elif result.status == "warning":
                self._handle_warning_issue(result)
        
        # Alertas inmediatas para condiciones críticas
        if critical_count >= MONITORING_CONFIG['critical_alert_threshold']:
            self._send_critical_alert(health_results)
            
    def _handle_critical_issue(self, result: HealthCheckResult):
        """
        Maneja problemas críticos con autoreparación
        
        Args:
            result: Resultado de verificación crítica
        """
        self.maintenance_logger.critical(f"PROBLEMA CRÍTICO: {result.component} - {result.message}")
        
        if result.component == "MT5_Connection" and not result.auto_fixed:
            # Intentar reinicialización completa de MT5
            self._attempt_mt5_recovery()
        elif result.component == "Trading_Bot":
            # Intentar reinicio del bot
            self._attempt_bot_restart()
            
    def _handle_warning_issue(self, result: HealthCheckResult):
        """
        Maneja advertencias del sistema
        
        Args:
            result: Resultado de verificación con advertencia
        """
        self.maintenance_logger.warning(f"ADVERTENCIA: {result.component} - {result.message}")
        
    def _attempt_mt5_recovery(self) -> bool:
        """
        Intenta recuperar la conexión MT5
        
        Returns:
            True si la recuperación fue exitosa
        """
        try:
            self.maintenance_logger.info("Iniciando recuperación de conexión MT5...")
            
            # Cerrar conexión actual
            mt5.shutdown()
            time.sleep(2)
            
            # Reinicializar
            if mt5.initialize():
                self.maintenance_logger.info("Recuperación MT5 exitosa")
                return True
            else:
                self.maintenance_logger.error("Fallo en recuperación MT5")
                return False
                
        except Exception as e:
            self.maintenance_logger.error(f"Error en recuperación MT5: {e}")
            return False
            
    def _attempt_bot_restart(self) -> bool:
        """
        Intenta reiniciar el bot de trading
        
        Returns:
            True si el reinicio fue exitoso
        """
        try:
            if self.bot and hasattr(self.bot, 'shutdown'):
                self.maintenance_logger.info("Reiniciando bot de trading...")
                self.bot.shutdown()
                time.sleep(3)
                
                # Aquí se podría reimplementar la lógica de reinicio del bot
                # Por ahora solo registramos el intento
                self.maintenance_logger.info("Intento de reinicio de bot completado")
                return True
                
        except Exception as e:
            self.maintenance_logger.error(f"Error reiniciando bot: {e}")
            return False
            
    def _send_critical_alert(self, health_results: List[HealthCheckResult]):
        """
        Envía alerta crítica al administrador
        
        Args:
            health_results: Resultados de verificaciones de salud
        """
        critical_issues = [r for r in health_results if r.status == "critical"]
        
        alert_message = f"""
        🚨 ALERTA CRÍTICA DEL SISTEMA DE TRADING 🚨
        
        Timestamp: {datetime.now().isoformat()}
        Problemas críticos detectados: {len(critical_issues)}
        
        Detalles:
        """
        
        for issue in critical_issues:
            alert_message += f"\n- {issue.component}: {issue.message}"
        
        alert_message += f"""
        
        Uptime del sistema: {(datetime.now() - self.start_time).total_seconds() / 3600:.1f} horas
        Total de errores: {self.error_count}
        
        ACCIÓN REQUERIDA: Revisar sistema inmediatamente
        """
        
        # Log de la alerta crítica
        self.maintenance_logger.critical(alert_message)
        
        # Aquí se podría implementar envío por email, SMS, etc.
        print(alert_message)  # Por ahora solo imprime en consola
        
    def _save_status_report(self, status: SystemStatus):
        """
        Guarda el reporte de estado en archivo JSON
        
        Args:
            status: Estado del sistema a guardar
        """
        try:
            report_file = f"status_reports/status_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Crear directorio si no existe
            os.makedirs("status_reports", exist_ok=True)
            
            # Cargar reportes existentes del día
            reports = []
            if os.path.exists(report_file):
                with open(report_file, 'r', encoding='utf-8') as f:
                    reports = json.load(f)
            
            # Agregar nuevo reporte
            reports.append(asdict(status))
            
            # Guardar reportes actualizados
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(reports, f, indent=2, ensure_ascii=False)
                
            self.maintenance_logger.info(f"Reporte de estado guardado: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Error guardando reporte de estado: {e}")

if __name__ == "__main__":
    main()