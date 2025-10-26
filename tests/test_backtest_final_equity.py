import os, sys, unittest
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "\\..")

from backtest import AdvancedBacktester

class TestBacktestFinalEquity(unittest.TestCase):
    def test_final_equity_present_in_optimized_flow(self):
        backtester = AdvancedBacktester(initial_capital=10000.0)
        report = backtester.run_advanced_backtest(symbol=None, days=7)
        self.assertIsInstance(report, dict)
        self.assertIn('final_equity', report)
        self.assertIn('initial_equity', report)
        self.assertIn('profitability', report)
        self.assertIn('total_return', report)

if __name__ == '__main__':
    unittest.main()