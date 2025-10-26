class AlertManager:
    def __init__(self):
        import logging
        self.logger = logging.getLogger('Alerts')

    def send_signal_alert(self, symbol: str, timeframe: str, signal: str, confidence: float, context: dict = None) -> None:
        details = []
        if context:
            # Tomamos algunos campos relevantes si existen
            for k in ['ema_fast', 'ema_slow', 'rsi', 'macd_histogram', 'mfi', 'near_support', 'near_resistance', 'fib_retracement']:
                if k in context:
                    details.append(f"{k}={context[k]}")
        msg = f"⚡ Señal {signal} {symbol} {timeframe} | Confianza {confidence:.2f}" + (f" | " + ", ".join(details) if details else "")
        self.logger.info(msg)
        print(msg)