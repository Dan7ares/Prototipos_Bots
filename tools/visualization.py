import logging
logger = logging.getLogger('Visualization')

def plot_technical_dashboard(df, save_path=None):
    """
    Dibuja un panel técnico con precio+MAs+BB, MACD, RSI/MFI y volumen/OBV.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))

        # Precio + EMAs + Bollinger
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(df.index, df['close'], label='Close', color='black')
        for col, color in [('ema_fast','blue'), ('ema_slow','orange'), ('ema_medium','purple')]:
            if col in df.columns:
                ax1.plot(df.index, df[col], label=col, color=color, alpha=0.8)
        if {'bb_upper','bb_middle','bb_lower'}.issubset(df.columns):
            ax1.plot(df.index, df['bb_upper'], color='green', alpha=0.5, linestyle='--')
            ax1.plot(df.index, df['bb_middle'], color='gray', alpha=0.5, linestyle=':')
            ax1.plot(df.index, df['bb_lower'], color='red', alpha=0.5, linestyle='--')
        if {'nearest_support','nearest_resistance'}.issubset(df.columns):
            ax1.plot(df.index, df['nearest_support'], color='green', alpha=0.3, label='Support')
            ax1.plot(df.index, df['nearest_resistance'], color='red', alpha=0.3, label='Resistance')
        ax1.legend(loc='upper left')
        ax1.set_title('Precio y niveles')

        # MACD
        ax2 = fig.add_subplot(4, 1, 2)
        if {'macd','macd_signal','macd_histogram'}.issubset(df.columns):
            ax2.plot(df.index, df['macd'], label='MACD', color='blue')
            ax2.plot(df.index, df['macd_signal'], label='Signal', color='orange')
            ax2.bar(df.index, df['macd_histogram'], label='Hist', color='gray', alpha=0.5)
        ax2.legend(loc='upper left')
        ax2.set_title('MACD')

        # RSI y MFI
        ax3 = fig.add_subplot(4, 1, 3)
        if 'rsi' in df.columns:
            ax3.plot(df.index, df['rsi'], label='RSI', color='purple')
            ax3.axhline(30, color='red', linestyle='--', alpha=0.3)
            ax3.axhline(70, color='green', linestyle='--', alpha=0.3)
        if 'mfi' in df.columns:
            ax3.plot(df.index, df['mfi'], label='MFI', color='brown')
            ax3.axhline(20, color='red', linestyle=':', alpha=0.3)
            ax3.axhline(80, color='green', linestyle=':', alpha=0.3)
        ax3.legend(loc='upper left')
        ax3.set_title('RSI / MFI')

        # Volumen y OBV
        ax4 = fig.add_subplot(4, 1, 4)
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
        if vol_col:
            ax4.bar(df.index, df[vol_col], label='Volume', color='steelblue', alpha=0.5)
        if 'obv' in df.columns:
            ax4.plot(df.index, df['obv'], label='OBV', color='black')
        ax4.legend(loc='upper left')
        ax4.set_title('Volumen/OBV')

        fig.tight_layout()
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"No se pudo renderizar el panel técnico: {e}")