import os, sys, unittest
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")

from performance_analyzer import PerformanceAnalyzer
from config.settings import STRATEGY_CONFIG

class TestPerformanceAnalyzerFast(unittest.TestCase):
    def test_run_analysis_with_config_fast_returns_metrics(self):
        analyzer = PerformanceAnalyzer(initial_capital=10000.0)
        metrics = analyzer._run_analysis_with_config_fast(analyzer.baseline_config, "BASELINE", symbol=None, days=7)
        self.assertTrue(hasattr(metrics, 'total_return'))
        self.assertTrue(hasattr(metrics, 'win_rate'))
        self.assertTrue(hasattr(metrics, 'profit_factor'))
        self.assertTrue(hasattr(metrics, 'max_drawdown'))
        self.assertTrue(hasattr(metrics, 'total_trades'))

    def test_run_comprehensive_analysis_returns_recommendation(self):
        analyzer = PerformanceAnalyzer(initial_capital=10000.0)
        comparison = analyzer.run_comprehensive_analysis(optimized_config=analyzer.baseline_config, symbol=None, days=7)
        self.assertTrue(hasattr(comparison, 'optimized_metrics'))
        self.assertTrue(hasattr(comparison, 'baseline_metrics'))
        self.assertTrue(hasattr(comparison, 'recommendation'))

if __name__ == '__main__':
    unittest.main()