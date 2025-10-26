#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sistema de Pruebas Automatizadas - Bot de Trading
-------------------------------------------------
Este script ejecuta todas las pruebas automatizadas del sistema
y genera un reporte de resultados.

Autor: Trading Bot Team
Versión: 1.0
"""

import os
import sys
import unittest
import logging
import datetime
import argparse
from unittest import TestLoader, TextTestRunner, TestResult

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('TestRunner')

def create_test_directory():
    """Crea el directorio de pruebas si no existe."""
    # Usar ruta absoluta del script actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, 'tests')
    
    if not os.path.exists(test_dir):
        try:
            os.makedirs(test_dir)
            logger.info(f"Directorio de pruebas creado en {test_dir}")
            
            # Crear archivo __init__.py en el directorio de pruebas
            init_file = os.path.join(test_dir, '__init__.py')
            with open(init_file, 'w') as f:
                f.write('"""Paquete de pruebas para el Bot de Trading"""\n')
            logger.info(f"Archivo {init_file} creado correctamente")
        except Exception as e:
            logger.error(f"Error al crear directorio de pruebas: {str(e)}")
            sys.exit(1)
    return test_dir

def run_tests(test_pattern=None):
    """
    Ejecuta las pruebas y genera un reporte.
    
    Args:
        test_pattern (str, optional): Patrón para filtrar pruebas específicas
    
    Returns:
        tuple: (éxito, reporte)
    """
    # Asegurar que existe el directorio de pruebas
    test_dir = create_test_directory()
    
    # Configurar el cargador de pruebas
    loader = TestLoader()
    
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Añadir el directorio actual al path para que Python pueda encontrar los módulos
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Determinar qué pruebas ejecutar
    try:
        if test_pattern:
            logger.info(f"Buscando pruebas con patrón: *{test_pattern}*.py")
            tests = loader.discover(test_dir, pattern=f'*{test_pattern}*.py')
        else:
            logger.info(f"Buscando todas las pruebas en: {test_dir}")
            tests = loader.discover(test_dir)
        
        # Verificar si se encontraron pruebas
        if not tests or tests.countTestCases() == 0:
            logger.warning(f"No se encontraron pruebas {'con el patrón especificado' if test_pattern else ''}")
            return False, "No se encontraron pruebas para ejecutar"
        
        # Ejecutar pruebas
        logger.info(f"Iniciando ejecución de {tests.countTestCases()} pruebas...")
        runner = TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(tests)
        
        # Generar reporte
        report = generate_report(result)
        
        return result.wasSuccessful(), report
    except Exception as e:
        error_msg = f"Error al ejecutar las pruebas: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def generate_report(result):
    """
    Genera un reporte detallado de las pruebas.
    
    Args:
        result (TestResult): Resultado de las pruebas
    
    Returns:
        str: Reporte en formato texto
    """
    report = []
    report.append("=" * 70)
    report.append(f"REPORTE DE PRUEBAS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    # Resumen
    report.append(f"\nRESUMEN:")
    report.append(f"  Pruebas ejecutadas: {result.testsRun}")
    report.append(f"  Pruebas exitosas: {result.testsRun - len(result.failures) - len(result.errors)}")
    report.append(f"  Pruebas fallidas: {len(result.failures)}")
    report.append(f"  Pruebas con error: {len(result.errors)}")
    
    # Detalles de fallos
    if result.failures:
        report.append("\nDETALLE DE FALLOS:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            report.append(f"\n  {i}. {test}")
            report.append("  " + "-" * 68)
            report.append("    " + "\n    ".join(traceback.split("\n")))
    
    # Detalles de errores
    if result.errors:
        report.append("\nDETALLE DE ERRORES:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            report.append(f"\n  {i}. {test}")
            report.append("  " + "-" * 68)
            report.append("    " + "\n    ".join(traceback.split("\n")))
    
    # Recomendaciones
    report.append("\nRECOMENDACIONES:")
    if not result.wasSuccessful():
        if result.failures:
            report.append("  - Revisar los casos de prueba fallidos y ajustar el código correspondiente")
        if result.errors:
            report.append("  - Corregir los errores de ejecución en las pruebas")
    else:
        report.append("  - Todas las pruebas pasaron correctamente")
        report.append("  - Considerar añadir más casos de prueba para aumentar la cobertura")
    
    return "\n".join(report)

def generate_pytest_report(summary: dict):
    """
    Genera un reporte en texto a partir de un resumen de pytest.
    
    Args:
        summary (dict): Diccionario con conteos de pruebas
        
    Returns:
        str: Reporte en formato texto
    """
    report = []
    report.append("=" * 70)
    report.append(f"REPORTE DE PRUEBAS (PYTEST) - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    report.append("\nRESUMEN:")
    report.append(f"  Pruebas totales: {summary.get('total', 0)}")
    report.append(f"  Pruebas exitosas: {summary.get('passed', 0)}")
    report.append(f"  Pruebas fallidas: {summary.get('failures', 0)}")
    report.append(f"  Pruebas con error: {summary.get('errors', 0)}")
    report.append(f"  Pruebas omitidas: {summary.get('skipped', 0)}")
    report.append(f"  Win Rate: {summary.get('win_rate_pct', 0.0)}%")
    
    report.append("\nRECOMENDACIONES:")
    if summary.get('failures', 0) > 0 or summary.get('errors', 0) > 0:
        report.append("  - Revisar los casos de prueba fallidos/erróneos.")
        report.append("  - Revisar pytest_report.xml para ver el detalle de los fallos.")
    else:
        report.append("  - Todas las pruebas pasaron correctamente.")
    
    return "\n".join(report)

def parse_junit_summary(xml_path: str) -> dict:
    """
    Parsea un archivo JUnit XML y devuelve conteos agregados y win rate.
    
    Args:
        xml_path (str): Ruta al archivo JUnit XML
        
    Returns:
        dict: Diccionario con conteos y win rate
    """
    import xml.etree.ElementTree as ET
    summary = {"total": 0, "failures": 0, "errors": 0, "skipped": 0, "passed": 0, "win_rate_pct": 0.0}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        total = failures = errors = skipped = 0

        # Maneja raíz <testsuite> o <testsuites> con múltiples <testsuite>
        suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
        for ts in suites:
            total += int(ts.attrib.get("tests", 0))
            failures += int(ts.attrib.get("failures", 0))
            errors += int(ts.attrib.get("errors", 0))
            skipped += int(ts.attrib.get("skipped", 0))

        passed = max(total - failures - errors - skipped, 0)
        win_rate_pct = round((passed / total) * 100.0, 2) if total else 0.0
        return {
            "total": total,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "passed": passed,
            "win_rate_pct": win_rate_pct,
        }
    except Exception as e:
        logger.error(f"Error al parsear JUnit XML: {e}")
        return summary

def run_pytest_with_junit(pattern: str | None = None):
    """Ejecuta pytest y calcula win rate usando JUnit XML."""
    try:
        import pytest, json
        # Ejecutar pytest en el directorio de tests con reporte JUnit
        args = ['-q', '--junitxml=pytest_report.xml', 'tests']
        if pattern:
            args.extend(['-k', pattern])
        exit_code = pytest.main(args)
        # Parsear resumen
        summary = parse_junit_summary('pytest_report.xml')
        print(f"📊 Win Rate: {summary['win_rate_pct']}% ({summary['passed']}/{summary['total']})")
        with open('test_results_summary.json', 'w', encoding='utf-8') as f:
            json.dump({**summary, "exit_code": exit_code}, f, ensure_ascii=False, indent=2)
        return {**summary, "exit_code": exit_code}
    except Exception as e:
        logger.error(f"Error ejecutando pytest: {e}")
        return None

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Ejecutar pruebas automatizadas')
    parser.add_argument('--runner', choices=['pytest','unittest'], default='pytest', help='Selecciona el runner de pruebas')
    parser.add_argument('--pattern', type=str, help='Patrón para filtrar pruebas (pytest -k / unittest discover)')
    parser.add_argument('--report', type=str, help='Archivo para guardar el reporte (por defecto unittest_report.txt)')
    args = parser.parse_args()

    try:
        if args.runner == 'pytest':
            summary = run_pytest_with_junit(args.pattern)
            if summary:
                # Generar, imprimir y guardar reporte
                report = generate_pytest_report(summary)
                print(report)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                report_path = args.report or os.path.join(current_dir, 'unittest_report.txt')
                try:
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    logger.info(f"Reporte de texto guardado en {report_path}")
                except Exception as e:
                    logger.error(f"Error al guardar el reporte: {str(e)}")
                sys.exit(0 if summary['failures'] == 0 and summary['errors'] == 0 else 1)
            else:
                logger.error("No se pudo generar el resumen de pytest.")
                sys.exit(1)
        else:
            # Runner unittest clásico
            success, report = run_tests(args.pattern)
            print(report)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            report_path = args.report or os.path.join(current_dir, 'unittest_report.txt')
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Reporte guardado en {report_path}")
            except Exception as e:
                logger.error(f"Error al guardar el reporte: {str(e)}")
            sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Error en la ejecución principal: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()