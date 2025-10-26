#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import json
from datetime import datetime
from typing import Dict, Callable, List, Tuple

import numpy as np
import pandas as pd

from strategies.m1_strategy import M1ScalpingStrategy
from core.execution_m1 import ExecutionEngineM1
from core.risk_manager_m1 import RiskManagerM1
from core.data_loader import load_historical_data
from config.m1_specialization import M1_STRATEGY_CONFIG, RISK_CONFIG_M1

class ParameterTunerM1:
    def __init__(self, data_provider: Callable[[str, str, int], pd.DataFrame] = None):
        self.data_provider = data_provider or load_historical_data

    def evaluate_config(self, symbol: str, timeframe: str, config: Dict, days: int) -> Dict:
        df = self.data_provider(symbol, timeframe, count=days * (1440 // (1 if timeframe == "M1" else 5)))
        if df is None or len(df) < 120:
            return {"error": "Datos insuficientes"}
        strat = M1ScalpingStrategy(config)
        df = strat.calculate_indicators(df)
        rm = RiskManagerM1(RISK_CONFIG_M1)
        engine = ExecutionEngineM1(rm, commission_pct=RISK_CONFIG_M1.get("commission_pct", 0.00010))
        out = engine.backtest_loop(df, strat, symbol, config)

        trades = out["trades"]
        equity = out["equity_curve"]
        wins = sum(1 for t in trades if t["profit"] > 0)
        losses = sum(1 for t in trades if t["profit"] <= 0)
        total = len(trades)
        win_rate = (wins / total) if total else 0.0
        gross_profit = sum(t["profit"] for t in trades if t["profit"] > 0)
        gross_loss = abs(sum(t["profit"] for t in trades if t["profit"] <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
        returns = np.diff(equity) / np.clip(equity[:-1], 1e-9, None) if len(equity) > 1 else np.array([])
        sharpe = float((np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(len(returns))) if returns.size > 1 and np.std(returns, ddof=1) > 0 else 0.0
        avg_win = np.mean([t["profit"] for t in trades if t["profit"] > 0]) if wins else 0.0
        avg_loss = np.mean([t["profit"] for t in trades if t["profit"] <= 0]) if losses else 0.0
        expectancy = float(win_rate * avg_win + (1.0 - win_rate) * (-avg_loss))
        return {
            "total_trades": total,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "expectancy": expectancy,
            "final_equity": out["final_equity"]
        }

    def grid_search(self, symbol: str, timeframe: str, base_config: Dict, param_ranges: Dict[str, List], days: int, max_combinations: int = 30) -> Dict:
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        # Recorta número de combinaciones para evitar ejecuciones muy largas
        combos = list(itertools.product(*values))[:max_combinations]

        results = []
        for comb in combos:
            cfg = base_config.copy()
            for k, v in zip(keys, comb):
                cfg[k] = v
            metrics = self.evaluate_config(symbol, timeframe, cfg, days)
            if "error" in metrics:
                continue
            # Score compuesto: prioriza win rate, luego PF y expectativa
            score = (metrics["win_rate"] * 0.5) + (min(metrics["profit_factor"], 3.0) / 3.0 * 0.3) + (max(metrics["expectancy"], -50) / 50.0 * 0.2)
            results.append({"config": cfg, "metrics": metrics, "score": float(score)})

        if not results:
            return {"error": "Sin resultados"}

        best = max(results, key=lambda r: r["score"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"m1_param_tuning_results_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"best": best, "all": results}, f, ensure_ascii=False, indent=2)
        return {"best": best, "file": out_file, "tested": len(results)}

    def time_cv_search(self, symbol: str, timeframe: str, base_config: Dict, param_ranges: Dict[str, List],
                       days: int, folds: int = 3, method: str = "grid", max_combinations: int = 30) -> Dict:
        """
        Búsqueda con validación cruzada temporal (folds consecutivos por tiempo).
        """
        df = load_historical_data(symbol, timeframe, count=days * (1440 // (1 if timeframe == "M1" else 5)))
        if df is None or len(df) < 120:
            return {"error": "Datos insuficientes para CV"}
        # Preparar folds por índice temporal
        n = len(df)
        fold_size = n // folds
        windows = [(i*fold_size, (i+1)*fold_size) for i in range(folds-1)] + [((folds-1)*fold_size, n)]
        # Generar combinaciones
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        combos = list(itertools.product(*values))
        combos = combos[:max_combinations] if method == "grid" else [tuple(np.random.choice(v) for v in values) for _ in range(max_combinations)]

        def eval_cfg(cfg: Dict) -> Dict:
            metrics_list = []
            for (s, e) in windows:
                window = df.iloc[s:e]
                if len(window) < 120: 
                    continue
                strat = M1ScalpingStrategy(cfg)
                wcalc = strat.calculate_indicators(window)
                rm = RiskManagerM1(RISK_CONFIG_M1)
                engine = ExecutionEngineM1(rm, commission_pct=RISK_CONFIG_M1.get("commission_pct", 0.00010))
                out = engine.backtest_loop(wcalc, strat, symbol, cfg)
                trades = out["trades"]; equity = out["equity_curve"]
                wins = sum(1 for t in trades if t["profit"] > 0)
                total = len(trades)
                win_rate = (wins / total) if total else 0.0
                gp = sum(t["profit"] for t in trades if t["profit"] > 0); gl = abs(sum(t["profit"] for t in trades if t["profit"] <= 0))
                pf = (gp / gl) if gl > 0 else (2.0 if gp > 0 else 1.0)
                returns = np.diff(equity) / np.clip(equity[:-1], 1e-9, None) if len(equity) > 1 else np.array([])
                sharpe = float((np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(len(returns))) if returns.size > 1 and np.std(returns, ddof=1) > 0 else 0.0
                metrics_list.append({"win_rate": win_rate, "profit_factor": pf, "sharpe_ratio": sharpe, "trades": total})
            if not metrics_list:
                return {"score": -np.inf, "avg": {}}
            avg_wr = float(np.mean([m["win_rate"] for m in metrics_list]))
            avg_pf = float(np.mean([m["profit_factor"] for m in metrics_list]))
            avg_sh = float(np.mean([m["sharpe_ratio"] for m in metrics_list]))
            score = (avg_wr * 0.6) + (min(avg_pf, 3.0) / 3.0 * 0.3) + (max(avg_sh, 0) / 2.0 * 0.1)
            return {"score": score, "avg": {"win_rate": avg_wr, "profit_factor": avg_pf, "sharpe_ratio": avg_sh}}

        results = []
        for comb in combos:
            cfg = base_config.copy()
            for k, v in zip(keys, comb):
                cfg[k] = v
            ev = eval_cfg(cfg)
            results.append({"config": cfg, "cv": ev})

        if not results:
            return {"error": "Sin resultados en CV"}

        best = max(results, key=lambda r: r["cv"]["score"])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"m1_param_tuning_cv_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"best": best, "all": results}, f, ensure_ascii=False, indent=2)
        return {"best": best, "file": out_file, "tested": len(results)}

    def suggest_next_config(self, base_config: Dict, metrics: Dict) -> Dict:
        cfg = base_config.copy()
        # Heurísticas simples: si winrate bajo, subir min_confidence; si profit_factor bajo, aumentar TP/ajustar SL
        if metrics.get("win_rate", 0.0) < 0.35:
            cfg["min_confidence"] = min(cfg.get("min_confidence", 0.70) + 0.05, 0.85)
        if metrics.get("profit_factor", 0.0) < 1.0:
            cfg["take_profit_multiplier"] = min(cfg.get("take_profit_multiplier", 1.6) + 0.2, 3.0)
            cfg["stop_loss_multiplier"] = max(cfg.get("stop_loss_multiplier", 1.0) - 0.1, 0.6)
        # Ajuste de EMAs para tendencia más clara si Sharpe bajo
        if metrics.get("sharpe_ratio", 0.0) < 0.3:
            cfg["ema_fast"] = max(int(cfg.get("ema_fast", 4)) - 1, 2)
            cfg["ema_slow"] = min(int(cfg.get("ema_slow", 21)) + 2, 34)
        return cfg

    def advanced_optimization_70_wr(self, symbol: str, timeframe: str, base_config: Dict, days: int = 7) -> Dict:
        """
        Optimización avanzada específicamente diseñada para alcanzar win rate del 70%+
        Implementa múltiples estrategias de optimización combinadas.
        """
        print(f"🎯 Iniciando optimización avanzada para Win Rate 70%+ en {symbol} {timeframe}")
        
        # Configuraciones candidatas basadas en análisis de resultados históricos
        high_wr_configs = [
            # Config 1: Ultra conservadora - alta confianza
            {**base_config, "min_confidence": 0.80, "adx_min": 26, "take_profit_multiplier": 1.6, "stop_loss_multiplier": 0.8},
            # Config 2: EMAs optimizadas + TP/SL balanceado
            {**base_config, "ema_fast": 3, "ema_medium": 8, "ema_slow": 18, "min_confidence": 0.70, "adx_min": 22, "take_profit_multiplier": 1.8, "stop_loss_multiplier": 0.9},
            # Config 3: Filtros estrictos + confirmaciones extra
            {**base_config, "min_confidence": 0.75, "min_confirmations": 4, "adx_min": 25, "take_profit_multiplier": 1.7, "stop_loss_multiplier": 0.85},
            # Config 4: Balance óptimo encontrado en backtests
            {**base_config, "ema_fast": 3, "ema_medium": 8, "ema_slow": 18, "min_confidence": 0.72, "adx_min": 24, "take_profit_multiplier": 1.8, "stop_loss_multiplier": 1.0},
            # Config 5: Híbrida con stops muy ajustados
            {**base_config, "ema_fast": 3, "ema_medium": 8, "ema_slow": 18, "min_confidence": 0.68, "adx_min": 22, "take_profit_multiplier": 2.0, "stop_loss_multiplier": 0.8},
        ]
        
        results = []
        best_wr = 0.0
        
        for i, config in enumerate(high_wr_configs, 1):
            print(f"📊 Evaluando configuración {i}/5...")
            metrics = self.evaluate_config(symbol, timeframe, config, days)
            
            if "error" in metrics:
                continue
                
            wr = metrics["win_rate"]
            pf = metrics["profit_factor"]
            trades = metrics["total_trades"]
            
            # Score ponderado priorizando win rate
            score = (wr * 0.7) + (min(pf, 3.0) / 3.0 * 0.2) + (min(trades, 200) / 200.0 * 0.1)
            
            results.append({
                "config": config,
                "metrics": metrics,
                "score": score
            })
            
            print(f"   Win Rate: {wr:.1%}, PF: {pf:.2f}, Trades: {trades}, Score: {score:.3f}")
            
            if wr > best_wr:
                best_wr = wr
                
        if not results:
            return {"error": "No se pudieron evaluar configuraciones"}
            
        # Ordenar por win rate primero, luego por score
        results.sort(key=lambda x: (x["metrics"]["win_rate"], x["score"]), reverse=True)
        best = results[0]
        
        # Si el mejor resultado está cerca del 70%, hacer ajuste fino
        if best["metrics"]["win_rate"] >= 0.65:
            print(f"🔧 Win Rate actual: {best['metrics']['win_rate']:.1%}. Aplicando ajuste fino...")
            fine_tuned = self.fine_tune_for_70_wr(symbol, timeframe, best["config"], days)
            if fine_tuned["metrics"]["win_rate"] > best["metrics"]["win_rate"]:
                best = fine_tuned
                
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"m1_advanced_70wr_optimization_{ts}.json"
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "target": "Win Rate 70%+",
                "best": best,
                "all_results": results,
                "summary": {
                    "best_win_rate": best["metrics"]["win_rate"],
                    "target_achieved": best["metrics"]["win_rate"] >= 0.70,
                    "total_configs_tested": len(results)
                }
            }, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Optimización completada. Mejor Win Rate: {best['metrics']['win_rate']:.1%}")
        print(f"📁 Resultados guardados en: {out_file}")
        
        return {
            "best": best,
            "file": out_file,
            "target_achieved": best["metrics"]["win_rate"] >= 0.70,
            "tested": len(results)
        }

    def fine_tune_for_70_wr(self, symbol: str, timeframe: str, base_config: Dict, days: int) -> Dict:
        """
        Ajuste fino específico para alcanzar exactamente 70% de win rate
        """
        print("🎯 Aplicando ajuste fino para 70% win rate...")
        
        # Micro-ajustes basados en la configuración base
        fine_adjustments = [
            # Aumentar ligeramente la confianza
            {**base_config, "min_confidence": min(base_config.get("min_confidence", 0.70) + 0.02, 0.85)},
            # Ajustar ADX para ser más selectivo
            {**base_config, "adx_min": min(base_config.get("adx_min", 22) + 1, 28)},
            # Optimizar ratio TP/SL
            {**base_config, "take_profit_multiplier": base_config.get("take_profit_multiplier", 1.8) * 0.95, "stop_loss_multiplier": base_config.get("stop_loss_multiplier", 0.9) * 0.95},
            # Combinación de ajustes
            {**base_config, "min_confidence": min(base_config.get("min_confidence", 0.70) + 0.01, 0.85), "adx_min": min(base_config.get("adx_min", 22) + 1, 28)},
        ]
        
        best_result = {"metrics": {"win_rate": 0.0}}
        
        for config in fine_adjustments:
            metrics = self.evaluate_config(symbol, timeframe, config, days)
            if "error" not in metrics and metrics["win_rate"] > best_result["metrics"]["win_rate"]:
                best_result = {"config": config, "metrics": metrics}
                
        return best_result if best_result["metrics"]["win_rate"] > 0 else {"config": base_config, "metrics": self.evaluate_config(symbol, timeframe, base_config, days)}

    def iterative_adjust(self, symbol: str, timeframe: str, base_config: Dict, days: int, iterations: int = 3) -> Tuple[Dict, List[Dict]]:
        history = []
        cfg = base_config.copy()
        for _ in range(iterations):
            metrics = self.evaluate_config(symbol, timeframe, cfg, days)
            if "error" in metrics:
                break
            history.append({"config": cfg.copy(), "metrics": metrics})
            cfg = self.suggest_next_config(cfg, metrics)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"m1_iterative_adjust_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"history": history, "final_config": cfg}, f, ensure_ascii=False, indent=2)
        return cfg, history