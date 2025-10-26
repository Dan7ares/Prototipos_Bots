import os, sys, unittest
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")

from performance_analyzer import PerformanceAnalyzer

class TestIntegrationPipeline(unittest.TestCase):
    def test_comparison_report_generates_without_error(self):
        analyzer = PerformanceAnalyzer(initial_capital=10000.0)
        comparison = analyzer.run_comprehensive_analysis(optimized_config=analyzer.baseline_config, symbol=None, days=7)
        report = analyzer.generate_comparison_report()
        self.assertIsInstance(report, str)
        self.assertIn("REPORTE COMPARATIVO", report or "")

if __name__ == '__main__':
    unittest.main()