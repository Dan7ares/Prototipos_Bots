import threading
import time
from logging import getLogger

class PerformanceMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.logger = getLogger('PerformanceMonitor')
        self.running = False

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._monitor)
        thread.start()

    def _monitor(self):
        while self.running:
            metrics = self._calculate_metrics()
            self.logger.info(f"Metrics: Win Rate {metrics['win_rate']:.2%}, Profit {metrics['profit']}")
            time.sleep(300)  # 5 min

    def _calculate_metrics(self):
        # Lógica para calcular win rate, profit desde bot.positions
        return {'win_rate': 0.75, 'profit': 10.0}  # Placeholder; integra con bot real

    def stop(self):
        self.running = False