# %%
import pandas as pd
import numpy as np
import requests

# %%
class RiskControlScorer:
    def __init__(self):
        self.adv_20_min = 10e9
        self.market_cap_min = 1000e9

        self.beta_3m_pct_threshold = 0.80
        self.dd_3m_pct_threshold = 0.80
        self.vol_3m_pct_threshold = 0.80
        
        self.beta_1m_3m_diff_max = 1.0
        
        self.turnover_pct_threshold = 0.30
        self.bid_ask_spread_pct_threshold = 0.70
        
        self.red_flag_vol_threshold = 0.80
        self.red_flag_adv_threshold = 25e12
        
    def check_hard_requirements(self, rec):
        adv_20 = rec.get('adv_20_abs', 0)
        
        if adv_20 < self.adv_20_min:
            return False, "ADV-20 below minimum threshold"
        
        return True, "Hard requirements passed"
    
    def check_market_cap(self, rec):
        """Soft check for market cap"""
        market_cap = rec.get('market_cap', 0)
        if market_cap > 0 and market_cap < self.market_cap_min:
            return False, "Market cap below minimum threshold"
        elif market_cap == 0:
            return True, "Market cap data not available"
        return True, "Market cap check passed"
    
    def check_turnover_ratio(self, rec):
        turnover_pct = rec.get('pct_turnover', None)
        if turnover_pct is not None and turnover_pct < self.turnover_pct_threshold:
            return False, f"Turnover ratio too low: {turnover_pct:.2%}"
        return True, "Turnover ratio check passed"
    
    def check_bid_ask_spread(self, rec):
        spread_pct = rec.get('pct_avg_bid_ask_spread', None)
        if spread_pct is not None and spread_pct > self.bid_ask_spread_pct_threshold:
            return False, f"Bid-ask spread too high: {spread_pct:.2%}"
        return True, "Bid-ask spread check passed"
    
    def check_beta_3m(self, rec):
        beta_pct = rec.get('pct_beta_3m', None)
        if beta_pct is not None and beta_pct > self.beta_3m_pct_threshold:
            return False, f"Beta 3M too high: {beta_pct:.2%}"
        return True, "Beta 3M check passed"
    
    def check_drawdown_3m(self, rec):
        dd_pct = rec.get('pct_dd_3m', None)
        if dd_pct is not None and dd_pct > self.dd_3m_pct_threshold:
            return False, f"Drawdown 3M too high: {dd_pct:.2%}"
        return True, "Drawdown 3M check passed"
    
    def check_beta_volatility_warning(self, rec):
        beta_1m = rec.get('beta_1m', None)
        beta_3m = rec.get('beta_3m', None)
        
        if beta_1m is not None and beta_3m is not None:
            beta_diff = abs(beta_1m - beta_3m)
            if beta_diff > self.beta_1m_3m_diff_max:
                return False, f"Beta difference too high: {beta_diff:.2f}"
        
        return True, "Beta volatility check passed"
    
    def check_volatility_3m(self, rec):
        vol_pct = rec.get('pct_vol_3m', None)
        if vol_pct is not None and vol_pct > self.vol_3m_pct_threshold:
            return False, f"Volatility 3M too high: {vol_pct:.2%}"
        return True, "Volatility 3M check passed"
    
    def check_red_flag_low_volume(self, rec):
        vol_pct = rec.get('pct_vol_3m', None)
        adv_20 = rec.get('adv_20_abs', 0)
        
        if vol_pct is not None and vol_pct > self.red_flag_vol_threshold and adv_20 < self.red_flag_adv_threshold:
            return True, f"RED FLAG: High volatility ({vol_pct:.2%}) + Low liquidity ({adv_20/1e9:.1f} tỷ)"
        
        return False, "No red flag for low volume"
    
    def score_symbol(self, symbol, rec, all_records=None):
        result = {
            'symbol': symbol,
            'score': 0,
            'hard_reject': False,
            'warnings': [],
            'red_flags': []
        }
        
        hard_ok, hard_msg = self.check_hard_requirements(rec)
        if not hard_ok:
            result['hard_reject'] = True
            return result
        
        checks = [
            ('Market Cap', self.check_market_cap),
            ('Turnover', self.check_turnover_ratio),
            ('Bid-Ask Spread', self.check_bid_ask_spread),
            ('Beta 3M', self.check_beta_3m),
            ('Drawdown 3M', self.check_drawdown_3m),
            ('Beta Volatility', self.check_beta_volatility_warning),
            ('Volatility 3M', self.check_volatility_3m),
        ]
        
        passed_count = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            passed, msg = check_func(rec)
            if passed:
                passed_count += 1
            else:
                result['warnings'].append(f"{check_name}: {msg}")
        
        # Check for red flag (D1)
        has_red_flag, red_flag_msg = self.check_red_flag_low_volume(rec)
        if has_red_flag:
            result['red_flags'].append(red_flag_msg)
        
        # CHECKLIST-BASED SCORING: Scale to 5 points
        # Base score = (passed / total) * 5.0
        pass_rate = passed_count / total_checks if total_checks > 0 else 0
        base_score = pass_rate * 5.0
        
        # Red flag penalty (20% reduction instead of fixed deduction)
        if has_red_flag:
            base_score *= 0.80
        
        result['score'] = round(base_score, 2)
        
        return result

# %%
def calculate_risk_metrics(df, symbol=None):
        if df is None or len(df) < 20:
            return {}
    
        close = df['close'].dropna()
        if len(close) < 20:
            return {}
    
        metrics = {}
    
        # 1. MAX DRAWDOWN 3M (60 trading days)
        lookback_3m = min(60, len(close))
        prices_3m = close.iloc[-lookback_3m:]
        running_max = prices_3m.expanding().max()
        drawdown = (prices_3m - running_max) / running_max
        max_dd = drawdown.min()
        metrics['dd_3m'] = abs(max_dd) if max_dd < 0 else 0.0
    
        # 2. ANNUALIZED VOLATILITY 3M (60 trading days) - C4
        if len(prices_3m) >= 2:
            log_returns = np.log(prices_3m / prices_3m.shift(1)).dropna()
            vol_3m = log_returns.std() * np.sqrt(252)  # Annualize by 252 trading days
            metrics['vol_3m'] = vol_3m
    
        # 3. VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR) 1M (20 trading days)
        lookback_1m = min(20, len(close))
        prices_1m = close.iloc[-lookback_1m:]
        if len(prices_1m) >= 2:
            log_returns_1m = np.log(prices_1m / prices_1m.shift(1)).dropna()
            var_95 = np.percentile(log_returns_1m, 5)  # 95% confidence level
            metrics['var_1m'] = abs(var_95) if var_95 < 0 else 0.0
            # CVaR = mean of returns below VaR
            cvar_95 = log_returns_1m[log_returns_1m <= var_95].mean()
            metrics['cvar_1m'] = abs(cvar_95) if cvar_95 < 0 else 0.0
    
        volume_col = None
        for col in ['total_volume', 'volume', 'Volume', 'vol']:
            if col in df.columns:
                volume_col = col
                break
    
        # Try to use total_value directly if available (more accurate)
        if 'total_value' in df.columns:
            lookback_20d = min(20, len(df))
            total_value_20d = df['total_value'].iloc[-lookback_20d:]
            adv_20 = np.mean(total_value_20d)
            metrics['adv_20_abs'] = adv_20
        elif volume_col is not None:
            lookback_20d = min(20, len(df))
            prices_20d = close.iloc[-lookback_20d:]
            volume_20d = df[volume_col].iloc[-lookback_20d:]
        
            daily_value = prices_20d.values * volume_20d.values
            adv_20 = np.mean(daily_value)
            metrics['adv_20_abs'] = adv_20
        else:
            metrics['adv_20_abs'] = 0
    
        if volume_col is not None:
            if 'shares_outstanding' in df.columns or 'shares_out' in df.columns:
                shares_col = 'shares_outstanding' if 'shares_outstanding' in df.columns else 'shares_out'
                lookback_20d = min(20, len(df))
                avg_volume = df[volume_col].iloc[-lookback_20d:].mean()
                shares_out = df[shares_col].iloc[-1]
                if shares_out > 0:
                    metrics['turnover'] = avg_volume / shares_out
            else:
                metrics['turnover'] = None

        if 'bid' in df.columns and 'ask' in df.columns:
            lookback_20d = min(20, len(df))
            bid_ask_data = df[['bid', 'ask']].iloc[-lookback_20d:]
            spread_pct = ((bid_ask_data['ask'] - bid_ask_data['bid']) / 
                         ((bid_ask_data['bid'] + bid_ask_data['ask']) / 2)).mean()
            metrics['avg_bid_ask_spread_pct'] = spread_pct
        else:
            metrics['avg_bid_ask_spread_pct'] = None
    
        return metrics


def calculate_drawdown_3m(df):
    metrics = calculate_risk_metrics(df)
    return metrics.get('dd_3m')


def calculate_risk_metrics_batch(symbol_list, stock_data, start_date=None, end_date=None):
    records = []

    beta_map = {}
    try:
        beta_params = {}
        if start_date:
            beta_params['start_date'] = start_date
        if end_date:
            beta_params['end_date'] = end_date
        resp = requests.get("http://192.168.8.190:8000/MKD/beta_calculation", params=beta_params if beta_params else None, timeout=60)
        if resp.status_code == 200:
            beta_json = resp.json()
            df_beta = pd.DataFrame(beta_json if isinstance(beta_json, list) else [beta_json])

            sym_col = None
            for c in ['symbol', 'Symbol', 'ticker', 'Ticker']:
                if c in df_beta.columns:
                    sym_col = c
                    break
            if sym_col is None:
                df_beta['symbol'] = None
                sym_col = 'symbol'

            for sym in symbol_list:
                row = df_beta[df_beta[sym_col] == sym]
                if not row.empty:
                    beta_map[sym] = row.iloc[-1].to_dict()
    except Exception as e:
        pass

    market_cap_map = {}
    try:
        cap_params = {}
        if end_date:
            cap_params['end_date'] = end_date
        resp = requests.get("http://192.168.8.190:8000/MKD/stock-ratios", params=cap_params if cap_params else None, timeout=60)
        if resp.status_code == 200:
            cap_json = resp.json()
            df_cap = pd.DataFrame(cap_json if isinstance(cap_json, list) else [cap_json])
            
            if 'code' in df_cap.columns:
                df_market_cap = df_cap[df_cap['code'] == 'ryd11'].copy()
                
                sym_col = None
                for c in ['symbol', 'Symbol', 'ticker', 'Ticker']:
                    if c in df_market_cap.columns:
                        sym_col = c
                        break
                
                if sym_col and 'value' in df_market_cap.columns:
                    for sym in symbol_list:
                        rows = df_market_cap[df_market_cap[sym_col] == sym]
                        if not rows.empty:
                            market_cap_value = rows.iloc[-1]['value']
                            if market_cap_value is not None and not pd.isna(market_cap_value):
                                market_cap_map[sym] = {
                                    'market_cap': float(market_cap_value) * 1e9
                                }
                
    except Exception as e:
        pass

    for sym in symbol_list:
        df = stock_data.get(sym)
        if df is None or len(df) == 0:
            continue

        rec = {'symbol': sym}
        rec.update(beta_map.get(sym, {}))
        rec.update(market_cap_map.get(sym, {}))
        
        risk_metrics = calculate_risk_metrics(df, sym)
        
        if 'adv_20_from_api' in rec and rec['adv_20_from_api'] > 0:
            rec['adv_20_abs'] = rec['adv_20_from_api']
        else:
            rec['adv_20_abs'] = risk_metrics.get('adv_20_abs', 0)
        
        risk_metrics.pop('adv_20_abs', None)
        rec.update(risk_metrics)
        
        records.append(rec)

    percentile_metrics = [
        ('beta_3m', 'pct_beta_3m'),
        ('dd_3m', 'pct_dd_3m'),
        ('vol_3m', 'pct_vol_3m'),
        ('turnover', 'pct_turnover'),
        ('avg_bid_ask_spread_pct', 'pct_avg_bid_ask_spread')
    ]
    
    for metric_key, pct_key in percentile_metrics:
        vals = [r[metric_key] for r in records if r.get(metric_key) is not None and not np.isnan(float(r.get(metric_key, np.nan)))]
        if not vals:
            continue
        
        for rec in records:
            val = rec.get(metric_key)
            if val is not None and not np.isnan(float(val)):
                # Special handling for turnover (higher is better)
                if metric_key == 'turnover':
                    # Lower percentile = lower turnover = worse
                    rank = sum(1 for v in vals if v < float(val))
                else:
                    # For risk metrics (higher is worse), higher percentile = higher risk
                    rank = sum(1 for v in vals if v <= float(val))
                rec[pct_key] = rank / len(vals)

    return records