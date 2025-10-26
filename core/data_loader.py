import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from datetime import datetime, timedelta

class MarketAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('MarketAnalyzer')
        
    def analyze_market(self, symbol, timeframe, periods=500):
        """
        Realiza un análisis completo del mercado para un símbolo y timeframe específicos
        """
        try:
            # Obtener datos históricos
            if timeframe == "M1":
                mt5_timeframe = mt5.TIMEFRAME_M1
            elif timeframe == "M5":
                mt5_timeframe = mt5.TIMEFRAME_M5
            else:
                self.logger.error(f"Timeframe no soportado: {timeframe}")
                return None
                
            # Obtener datos
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.error(f"No se pudieron obtener datos para {symbol} en {timeframe}")
                return None
                
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Calcular indicadores básicos
            df = self.calculate_indicators(df)
            
            # Calcular métricas de mercado
            metrics = self.calculate_market_metrics(df, symbol)
            
            return {
                'data': df,
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error analizando mercado {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """
        Calcula indicadores técnicos para el análisis
        """
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR para volatilidad
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Estructura de mercado (HH/HL vs LH/LL)
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2))
        df['higher_low'] = (df['low'] > df['low'].shift(1)) & (df['low'] > df['low'].shift(2))
        df['lower_high'] = (df['high'] < df['high'].shift(1)) & (df['high'] < df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2))
        
        # Detección de patrones de velas
        self._add_candle_patterns(df)
        
        return df
    
    def _add_candle_patterns(self, df):
        """
        Detecta patrones de velas japonesas
        """
        # Engulfing
        df['bullish_engulfing'] = (df['open'].shift(1) > df['close'].shift(1)) & \
                                 (df['close'] > df['open']) & \
                                 (df['open'] <= df['close'].shift(1)) & \
                                 (df['close'] >= df['open'].shift(1))
                                 
        df['bearish_engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & \
                                 (df['open'] > df['close']) & \
                                 (df['close'] <= df['open'].shift(1)) & \
                                 (df['open'] >= df['close'].shift(1))
        
        # Pinbar (aproximación)
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['body'] = abs(df['close'] - df['open'])
        
        df['bullish_pinbar'] = (df['lower_wick'] > df['body'] * 2) & \
                              (df['lower_wick'] > df['upper_wick'] * 2)
                              
        df['bearish_pinbar'] = (df['upper_wick'] > df['body'] * 2) & \
                              (df['upper_wick'] > df['lower_wick'] * 2)
        
        # Inside bar
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & \
                          (df['low'] > df['low'].shift(1))
        
        # Doji
        df['doji'] = df['body'] < (df['high'] - df['low']) * 0.1
    
    def calculate_market_metrics(self, df, symbol):
        """
        Calcula métricas de mercado para evaluación
        """
        metrics = {}
        
        # Volatilidad
        metrics['volatility'] = {
            'atr_avg': df['atr'].mean(),
            'price_std': df['close'].std(),
            'daily_range_avg': ((df['high'] - df['low']) / df['low'] * 10000).mean()  # En pips
        }
        
        # Liquidez (aproximación por volumen)
        metrics['liquidity'] = {
            'volume_avg': df['tick_volume'].mean(),
            'volume_std': df['tick_volume'].std()
        }
        
        # Tendencia
        ema_trend = df['ema_9'] > df['ema_21']
        trend_changes = ema_trend.diff().abs().sum()
        
        metrics['trend'] = {
            'strength': abs(df['ema_9'].iloc[-1] - df['ema_50'].iloc[-1]) / df['atr'].iloc[-1],
            'direction': 1 if df['ema_9'].iloc[-1] > df['ema_50'].iloc[-1] else -1,
            'stability': 1 - (trend_changes / len(df))  # 1 = estable, 0 = inestable
        }
        
        # Spread (si está disponible)
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                metrics['spread'] = symbol_info.spread * symbol_info.point
            else:
                metrics['spread'] = None
        except:
            metrics['spread'] = None
        
        # Patrones de velas
        metrics['patterns'] = {
            'engulfing': (df['bullish_engulfing'].sum() + df['bearish_engulfing'].sum()) / len(df),
            'pinbar': (df['bullish_pinbar'].sum() + df['bearish_pinbar'].sum()) / len(df),
            'inside_bar': df['inside_bar'].sum() / len(df),
            'doji': df['doji'].sum() / len(df)
        }
        
        return metrics
    
    def compare_markets(self, symbols, timeframes=["M1", "M5"]):
        """
        Compara múltiples mercados y devuelve un ranking
        """
        results = {}
        
        for symbol in symbols:
            symbol_results = {}
            
            for timeframe in timeframes:
                analysis = self.analyze_market(symbol, timeframe)
                if analysis:
                    symbol_results[timeframe] = analysis
            
            if symbol_results:
                results[symbol] = symbol_results
        
        # Calcular puntuaciones
        scores = self._calculate_market_scores(results)
        
        return {
            'detailed_analysis': results,
            'scores': scores
        }
    
    def _calculate_market_scores(self, results):
        """
        Calcula puntuaciones para cada mercado basado en criterios óptimos
        """
        scores = {}
        
        for symbol, timeframes in results.items():
            symbol_score = 0
            timeframe_scores = {}
            
            for timeframe, analysis in timeframes.items():
                metrics = analysis['metrics']
                score = 0
                
                # Volatilidad (25%)
                volatility_score = min(metrics['volatility']['atr_avg'] * 10000, 10) / 2  # Max 5 puntos
                
                # Liquidez (25%)
                liquidity_score = min(metrics['liquidity']['volume_avg'] / 100, 5)  # Max 5 puntos
                
                # Tendencia (25%)
                trend_score = metrics['trend']['stability'] * 2.5 + min(metrics['trend']['strength'], 1) * 2.5  # Max 5 puntos
                
                # Spread (15%)
                spread_score = 0
                if metrics['spread'] is not None:
                    spread_score = max(0, 3 - metrics['spread'] * 10000)  # Max 3 puntos
                
                # Patrones (10%)
                pattern_score = (metrics['patterns']['engulfing'] + 
                               metrics['patterns']['pinbar'] + 
                               metrics['patterns']['inside_bar'] / 2) * 20  # Max 2 puntos
                
                # Puntuación total
                score = volatility_score + liquidity_score + trend_score + spread_score + pattern_score
                timeframe_scores[timeframe] = {
                    'total': score,
                    'components': {
                        'volatility': volatility_score,
                        'liquidity': liquidity_score,
                        'trend': trend_score,
                        'spread': spread_score,
                        'patterns': pattern_score
                    }
                }
                
                symbol_score += score / len(timeframes)
            
            scores[symbol] = {
                'total': symbol_score,
                'timeframes': timeframe_scores
            }
        
        return scores# utils/data_loader.py - VERSIÓN SIMPLIFICADA

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import logging

class DataLoader:

    def __init__(self):
        self.logger = logging.getLogger('DataLoader')
        self.symbol = "EURUSDm"
        self.timeframe = mt5.TIMEFRAME_M15
        self.rates_total = 500
    
    def get_current_data(self, symbol=None):
        """Obtiene datos actuales del mercado"""
        try:
            if symbol:
                self.symbol = symbol
            
            print(f"📡 Conectando con MT5...")
            if not mt5.initialize():
                self.logger.error("No se pudo inicializar MT5")
                return None
            
            print(f"📊 Obteniendo datos para {self.symbol}...")
            rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, self.rates_total)
            
            if rates is None:
                self.logger.error(f"No se pudieron obtener datos para {self.symbol}")
                return None
            
            # Crear DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Calcular indicadores básicos
            df = self._calculate_basic_indicators(df)
            
            print(f"✅ Datos obtenidos: {len(df)} registros, {len(df.columns)} indicadores")
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
        finally:
            try:
                mt5.shutdown()
            except:
                pass

    def _calculate_basic_indicators(self, df):
        """Calcula indicadores técnicos básicos"""
        try:
            close = df['close'].values
            
            # Medias móviles
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI manual
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD manual
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Estocástico manual
            low_14 = df['low'].rolling(14).min()
            high_14 = df['high'].rolling(14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # ATR manual
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ADX manual básico
            df['adx_14'] = 25.0  # Placeholder
            
            # Rellenar NaN values
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculando indicadores: {e}")
            # Indicadores básicos de respaldo
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi_14'] = 50.0
            df['macd'] = 0.0
            return df


def load_historical_data(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    """
    Carga datos históricos optimizados para backtesting de alta rentabilidad.
    
    Args:
        symbol (str): Símbolo del instrumento (ej: 'EURUSD')
        timeframe (str): Timeframe (ej: 'M1', 'M5', 'M15')
        count (int): Número de barras a cargar
        
    Returns:
        pd.DataFrame: Datos con indicadores y validaciones de calidad
    """
    logger = logging.getLogger('DataLoader')
    try:
        if not mt5.initialize():
            logger.error("No se pudo inicializar MT5")
            return None
        
        # Mapear timeframe expandido
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        mt5_timeframe = timeframe_map.get(timeframe)
        if not mt5_timeframe:
            logger.error(f"Timeframe no soportado: {timeframe}")
            return None
        
        # Seleccionar símbolo con fallback a variantes comunes
        base_symbol = symbol
        selected = mt5.symbol_select(base_symbol, True)
        symbol_to_use = base_symbol
        if not selected:
            alt = f"{base_symbol}m"
            if mt5.symbol_select(alt, True):
                symbol_to_use = alt
            else:
                available = mt5.symbols_get()
                match = next((s.name for s in available if s.name.startswith(base_symbol)), None)
                if match and mt5.symbol_select(match, True):
                    symbol_to_use = match
    
        rates = mt5.copy_rates_from_pos(symbol_to_use, mt5_timeframe, 0, count)
        if (rates is None or len(rates) == 0) and symbol_to_use != base_symbol:
            # Último intento con símbolo base
            rates = mt5.copy_rates_from_pos(base_symbol, mt5_timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"No se pudieron obtener datos para {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Renombrar columnas para consistencia
        df.rename(columns={
            'tick_volume': 'volume'
        }, inplace=True)
        # Spread real desde MT5
        try:
            symbol_info = mt5.symbol_info(symbol_to_use)
            df['spread'] = (symbol_info.spread * symbol_info.point) if symbol_info else 0.0
        except Exception:
            df['spread'] = 0.0
        
        # Validar calidad de datos
        if len(df) < 50:
            logger.warning(f"Datos insuficientes: {len(df)} barras para {symbol}")
            return None
        
        # Verificar gaps excesivos
        price_changes = df['close'].pct_change().abs()
        if price_changes.max() > 0.05:  # 5% cambio máximo
            logger.warning(f"Gap excesivo detectado en {symbol}: {price_changes.max():.4f}")
        
        # Verificar datos faltantes
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            logger.warning(f"Datos faltantes en {symbol}: {missing_data} valores")
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calcular indicadores (reutilizando lógica existente)
        analyzer = MarketAnalyzer()  # Instancia de la clase existente
        df = analyzer.calculate_indicators(df)
        
        logger.info(f"✅ Datos cargados: {symbol} {timeframe} - {len(df)} barras")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error cargando datos históricos: {e}")
        return None
    finally:
        mt5.shutdown()