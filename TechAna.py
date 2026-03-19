# %%
import requests
import pandas as pd
import numpy as np
from Bluechip import get_cached_bluechips

START_DATE = "2022-10-01"
END_DATE = "2022-12-30"

# %%
class APIFetcher:    
    @staticmethod
    def fetch_batch(url, symbol_list, start_date, end_date, timeout_secs=60):
        all_data = {}
        try:
            params = {
                "symbols": ",".join(symbol_list),
                "start_date": start_date,
                "end_date": end_date
            }
            resp = requests.get(url, params=params, timeout=timeout_secs)
            if resp.status_code != 200:
                print(f"API Error {resp.status_code}: {url}")
                return {}
                
            batch_data = resp.json()
            
            if isinstance(batch_data, list):
                for item in batch_data:
                    symbol = item.get('symbol', item.get('Symbol'))
                    if symbol:
                        if symbol not in all_data:
                            all_data[symbol] = []
                        all_data[symbol].append(item)
            elif isinstance(batch_data, dict):
                all_data.update(batch_data)
        except Exception as e:
            print(f"Error fetching batch: {e}")
        
        result = {}
        for symbol, data in all_data.items():
            try:
                df = pd.DataFrame(data) if isinstance(data, list) else data
                # Chuẩn hóa cột ngày tháng
                for date_col in ['date', 'Date', 'trading_date', 'TradingDate']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.sort_values(date_col).set_index(date_col)
                        break
                result[symbol] = df
            except Exception:
                pass
        return result

def calculate_technical_indicators_fallback(df):
    if df is None or len(df) == 0:
        return None
    x = df.copy()

    # Normalize column names
    x.columns = [c.lower() for c in x.columns]

    price = None
    for col in ['adj_close', 'close', 'price']:
        if col in x.columns:
            price = x[col]
            break
    if price is None:
        return None

    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    x['close'] = pd.to_numeric(price, errors='coerce')

    # Moving averages
    for w in [10, 20, 50, 100, 200]:
        x[f'ma{w}'] = x['close'].rolling(w).mean()

    # MACD (12,26,9)
    exp12 = x['close'].ewm(span=12, adjust=False).mean()
    exp26 = x['close'].ewm(span=26, adjust=False).mean()
    x['macd12269'] = exp12 - exp26
    x['macds12269'] = x['macd12269'].ewm(span=9, adjust=False).mean()

    # RSI(14)
    delta = x['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    x['rsi14'] = 100 - (100 / (1 + rs))

    # Bollinger(20,2)
    x['bbm2020'] = x['close'].rolling(20).mean()
    std20 = x['close'].rolling(20).std()
    x['bbu2020'] = x['bbm2020'] + 2 * std20
    x['bbl2020'] = x['bbm2020'] - 2 * std20

    return x

#%%
def calculate_trend_strength(df, symbol=None):
    if df is None or len(df) < 60:
        return {'trend_strength': 0, 'perf_3m': 0, 'ma_alignment': 0}
    
    close = df['close']
    
    ma20 = df['ma20'].iloc[-1] if 'ma20' in df.columns else 0
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns else 0
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns else ma50
    price = close.iloc[-1]
    
    ma_score = 0
    if ma20 > 0 and price > ma20: 
        ma_score += 1
    if ma50 > 0 and price > ma50: 
        ma_score += 1
    if ma200 > 0 and price > ma200: 
        ma_score += 1
    if ma20 > 0 and ma50 > 0 and ma20 > ma50: 
        ma_score += 1
    if ma50 > 0 and ma200 > 0 and ma50 > ma200: 
        ma_score += 1
    ma_score = ma_score / 5.0
    
    slope_3m = 0
    if len(df) >= 60 and 'ma50' in df.columns:
        ma50_60d_ago = df['ma50'].iloc[-60]
        if ma50_60d_ago > 0:
            slope_3m = (ma50 - ma50_60d_ago) / ma50_60d_ago
        slope_score = min(max(slope_3m * 2, -1), 1)
    else:
        slope_score = 0
    
    perf_3m = 0
    price_60d_ago = close.iloc[-60] if len(close) >= 60 else close.iloc[0]
    if price_60d_ago > 0:
        perf_3m = (price - price_60d_ago) / price_60d_ago
    perf_score = min(max(perf_3m * 2, -1), 1)
    
    trend_strength = (ma_score * 0.4 + 
                     (slope_score + 1) / 2 * 0.3 +
                     (perf_score + 1) / 2 * 0.3)
    trend_strength = trend_strength * 2 - 1
    
    return {
        'trend_strength': trend_strength,
        'ma_alignment': ma_score,
        'slope_3m': slope_3m,
        'perf_3m': perf_3m
    }


class TechnicalAnalysisScorer:
    def __init__(self):
        self.criteria = [
            # Short-term momentum
            {'id': 1, 'name': 'Price > MA20', 'type': 'compare', 'col1': 'close', 'col2': 'ma20', 'op': '>', 'weight': 0.5},            
            {'id': 2, 'name': 'MA20 > MA50', 'type': 'compare', 'col1': 'ma20', 'col2': 'ma50', 'op': '>', 'weight': 0.5},
            
            # Medium-term trend (20 days = ~1 month)
            {'id': 3, 'name': 'MA50 Slope (20d)', 'type': 'slope', 'col': 'ma50', 'lookback': 20, 'weight': 1.0},
            
            # Momentum indicators
            {'id': 4, 'name': 'MACD Bullish', 'type': 'macd', 'weight': 0.5},
            {'id': 5, 'name': 'RSI Optimal (45-75)', 'type': 'rsi_range', 'low': 45, 'high': 75, 'weight': 0.5},
            
            # Volume & Relative strength
            {'id': 6, 'name': 'Volume Spike', 'type': 'compare', 'col1': 'volume', 'col2': 'vol_ma20', 'op': '>', 'weight': 0.3},
            {'id': 7, 'name': 'RS vs VNI', 'type': 'rs', 'weight': 1.0},
        ]
    
    def evaluate_criterion(self, tech_df, criterion, vni_series=None):
        try:
            if tech_df is None or len(tech_df) < 20: 
                return {'pass': False}
            
            latest = tech_df.iloc[-1]
            crit_type = criterion['type']
            
            if crit_type == 'compare':
                val1 = latest.get(criterion['col1'], 0)
                val2 = latest.get(criterion['col2'], 0)
                if pd.isna(val1) or pd.isna(val2): 
                    return {'pass': False}
                passed = val1 > val2 if criterion['op'] == '>' else val1 < val2
                return {'pass': passed, 'val1': val1, 'val2': val2}

            elif crit_type == 'slope':
                col = criterion['col']
                lookback = criterion.get('lookback', 20)
                if len(tech_df) < lookback + 1: 
                    return {'pass': False, 'slope': 0, 'magnitude': 0}

                curr = tech_df[col].iloc[-1]
                prev = tech_df[col].iloc[-(lookback+1)]

                if pd.isna(curr) or pd.isna(prev) or prev == 0:
                    return {'pass': False, 'slope': 0, 'magnitude': 0}

                pct_change = (curr - prev) / prev
                
                passed = pct_change > 0.02
                
                magnitude_score = min(max((pct_change + 0.1) / 0.2, 0), 1)
                
                return {
                    'pass': passed, 
                    'slope': pct_change,
                    'magnitude': magnitude_score
                }

            # MACD Bullish
            elif crit_type == 'macd':
                macd = latest.get('macd12269', 0)
                signal = latest.get('macds12269', 0)
                return {'pass': macd > signal and macd > 0}

            # RSI
            elif crit_type == 'rsi_range':
                rsi = latest.get('rsi14', 50)
                return {'pass': criterion['low'] <= rsi <= criterion['high']}
            
            # Relative Strength vs VNINDEX
            elif crit_type == 'rs':
                if vni_series is None: 
                    return {'pass': False}
                common = tech_df.index.intersection(vni_series.index)
                if len(common) < 10: 
                    return {'pass': False}
                s_ret = tech_df.loc[common[-1], 'close'] / tech_df.loc[common[0], 'close']
                m_ret = vni_series.loc[common[-1]] / vni_series.loc[common[0]]
                return {'pass': s_ret > m_ret}
                
            return {'pass': False}
        except: 
            return {'pass': False}

    def score_symbol(self, symbol, tech_df, vni_series=None):
        if tech_df is None or len(tech_df) < 20:
            return {'symbol': symbol, 'score': 0, 'trend_strength': 0}
        
        total_score = 0.0
        if 'volume' in tech_df.columns and 'vol_ma20' not in tech_df.columns:
            tech_df['vol_ma20'] = tech_df['volume'].rolling(20).mean()

        details = {}
        for crit in self.criteria:
            res = self.evaluate_criterion(tech_df, crit, vni_series)
            details[f"C{crit['id']}"] = res
            
            if res.get('pass', False):
                if 'magnitude' in res:
                    total_score += crit['weight'] * res['magnitude']
                else:
                    total_score += crit['weight']
        
        trend_metrics = calculate_trend_strength(tech_df, symbol)
        trend_strength = trend_metrics['trend_strength']
        
        trend_multiplier = 1.0 + (trend_strength * 0.8)
        trend_multiplier = max(trend_multiplier, 0.2)
        
        final_score = total_score * trend_multiplier
        
        return {
            'symbol': symbol, 
            'score': final_score,
            'base_score': total_score,
            'trend_strength': trend_strength,
            'trend_metrics': trend_metrics,
            'details': details
        }

#%%
def fetch_outstanding_shares(symbol_list, start_date, end_date):
    shares_map = {}
    for sym in symbol_list:
        try:
            params = {
                "symbol": sym,
                "start_date": start_date,
                "end_date": end_date
            }
            r = requests.get(
                "http://192.168.8.190:8000/MKD/outstanding_shares",
                params=params,
                timeout=15
            )
            if r.status_code != 200:
                continue

            data = r.json()

            if isinstance(data, list) and len(data) > 0:
                # Pick the latest by date
                try:
                    latest = max(data, key=lambda x: x.get('date', ''))
                except Exception:
                    latest = data[-1]
                shares = latest.get('outstanding_shares', 0)
            elif isinstance(data, dict):
                shares = data.get('outstanding_shares', 0)
            else:
                shares = 0

            if shares and shares > 0:
                shares_map[sym] = shares
        except Exception:
            continue

    return shares_map

#%%    
def calculate_foreign_investor_metrics(symbol_list, start_date, end_date):
    all_data = {}

    outstanding_map = fetch_outstanding_shares(symbol_list, start_date, end_date)

    price_map = {}
    stock_data_temp = APIFetcher.fetch_batch(
        "http://192.168.8.190:8000/MKD/stock_daily",
        symbol_list, end_date, end_date, timeout_secs=30
    )
    for sym, df in stock_data_temp.items():
        if df is not None and len(df) > 0:
            price_map[sym] = df['close'].iloc[-1] if 'close' in df.columns else 10000

    for sym in symbol_list:
        try:
            params = {"symbol": sym, "start_date": start_date, "end_date": end_date}
            r = requests.get("http://192.168.8.190:8000/MKD/foreign_investors", params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.sort_values('date')
                    elif 'trading_date' in df.columns:
                        df['trading_date'] = pd.to_datetime(df['trading_date'])
                        df = df.sort_values('trading_date')

                    df['nff_vol'] = df['buy_volume'] - df['sell_volume']
                    df['f_activity'] = df['buy_volume'] + df['sell_volume']

                    if len(df) >= 5:
                        recent = df.tail(5)
                        nff_5d_vol = recent['nff_vol'].sum()
                        total_5d_buy_value = recent['buy_value'].sum()
                        total_5d_sell_value = recent['sell_value'].sum()
                        avg_activity = recent['f_activity'].mean()
                    else:
                        nff_5d_vol = df['nff_vol'].sum()
                        total_5d_buy_value = df['buy_value'].sum()
                        total_5d_sell_value = df['sell_value'].sum()
                        avg_activity = df['f_activity'].mean()

                    latest = df.iloc[-1]

                    shares = outstanding_map.get(sym, 1e9)
                    price = price_map.get(sym, 10000)

                    nff_5d_value = nff_5d_vol * price
                    activity_value = avg_activity * price
                    market_cap = shares * price
                    
                    # Use absolute buy/sell values to avoid negative bias
                    total_5d_activity_value = total_5d_buy_value + total_5d_sell_value

                    all_data[sym] = {
                        'nff_1d_vol': latest['nff_vol'],
                        'nff_5d_vol': nff_5d_vol,
                        'avg_f_activity': avg_activity,
                        'nff_5d_value': nff_5d_value,
                        'activity_value': activity_value,
                        'market_cap': market_cap,
                        'nff_5d_pct': (nff_5d_vol / shares * 100) if shares > 0 else 0,
                        'activity_pct': (avg_activity / shares * 100) if shares > 0 else 0,
                        'total_5d_activity_value': total_5d_activity_value,
                    }
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    # Value metrics: use absolute total activity instead of net NFF
    value_metrics = ['total_5d_activity_value', 'activity_value']
    for m in value_metrics:
        vals = [d[m] for d in all_data.values() if d.get(m) is not None]
        if not vals:
            continue

        for sym in all_data:
            val = all_data[sym].get(m, 0)
            rank = sum(1 for v in vals if val > v)
            all_data[sym][f'score_val_{m}'] = rank / len(vals) if len(vals) > 0 else 0

    # Absolute volume metrics (using abs value for NFF)
    abs_metrics = [('abs_nff_5d_vol', 'nff_5d_vol'), ('avg_f_activity', 'avg_f_activity')]
    for score_name, metric in abs_metrics:
        vals = [abs(d[metric]) for d in all_data.values() if d.get(metric) is not None]
        if not vals:
            continue

        for sym in all_data:
            val = abs(all_data[sym].get(metric, 0))
            rank = sum(1 for v in vals if val > v)
            all_data[sym][f'score_abs_{score_name}'] = rank / len(vals) if len(vals) > 0 else 0

    # Percentage metrics (allow negative for net flow direction)
    pct_metrics = ['nff_5d_pct', 'activity_pct']
    for m in pct_metrics:
        vals = [d[m] for d in all_data.values() if d.get(m) is not None]
        if not vals:
            continue

        for sym in all_data:
            val = all_data[sym].get(m, 0)
            if m.startswith('nff') and val < 0:
                all_data[sym][f'score_pct_{m}'] = 0
            else:
                rank = sum(1 for v in vals if val > v)
                all_data[sym][f'score_pct_{m}'] = rank / len(vals) if len(vals) > 0 else 0

    return all_data

#%%
class ForeignInvestorScorer:
    def __init__(self, mega_caps=None, large_caps=None):
        self.weights = {
            'val_total_activity': 2.0,  # Absolute trading activity (buy+sell)
            'val_activity': 1.5,         # Average daily activity
            'pct_nff_5d': 0.3,          # Net flow percentage (small weight)
        }
        # Use provided bluechips or defaults
        self.mega_caps = mega_caps if mega_caps is not None else ['VIC', 'VHM', 'VRE']
        self.large_caps = large_caps if large_caps is not None else ['NVL', 'KDH', 'DXG', 'BCM', 'NLG', 'KBC', 'HDG']
        
    def score_symbol(self, symbol, fi_dict):
        if not fi_dict: 
            return {'symbol': symbol, 'score': 0}
        
        score = 0.0

        # Score based on absolute trading activity (not net flow)
        val_score = 0.0
        if 'score_val_total_5d_activity_value' in fi_dict:
            val_score += fi_dict.get('score_val_total_5d_activity_value', 0) * self.weights['val_total_activity']
        
        val_score += fi_dict.get('score_val_activity_value', 0) * self.weights['val_activity']

        # Smaller weight on percentage metrics (net flow direction)
        pct_score = 0.0
        pct_score += fi_dict.get('score_pct_nff_5d_pct', 0) * self.weights['pct_nff_5d']

        mcap = fi_dict.get('market_cap', 0)
        
        # For large caps: emphasize absolute activity
        if mcap >= 5e12:
            score = val_score * 1.5
        elif mcap >= 1e12:
            score = val_score * 1.2
        else:
            # For small caps: also primarily use activity, but allow pct modifier
            score = val_score + pct_score * 0.3

        # Positive NFF bonus for bluechips and large caps
        nff = fi_dict.get('nff_5d_vol', 0)
        if symbol in self.mega_caps:
            if nff > 0:
                score *= 1.3
            elif nff < 0:
                score *= 0.8  # Reduced penalty for temporary selling
        elif symbol in self.large_caps:
            if nff > 0:
                score *= 1.2
            elif nff < 0:
                score *= 0.85

        return {'symbol': symbol, 'score': round(min(score, 5.0), 3)}
    
# %%
class RiskControlScorer:
    def __init__(self):
        self.max_dd_threshold = 0.4
        self.min_adv_threshold = 5e9
        self.min_price_threshold = 10000
        
    def score_symbol(self, symbol, rec):
        if not rec: 
            return {'symbol': symbol, 'score': 0, 'status': 'No Data'}
        
        base_score = 5.0
        penalties = {}

        adv = rec.get('adv_20', 0)
        if adv < self.min_adv_threshold:
            penalty = 2.0 if adv < (self.min_adv_threshold / 2) else 1.0
            base_score -= penalty
            penalties['Liquidity'] = f"-{penalty} (ADV: {adv/1e9:.1f}B)"

        price = rec.get('close_last', 0)
        if price < self.min_price_threshold:
            base_score -= 1.5
            penalties['Penny'] = "-1.5 (Price < 10k)"

        dd = rec.get('dd_3m', 0)
        if dd > self.max_dd_threshold:
            penalty = round((dd / self.max_dd_threshold) * 1.0, 2)
            base_score -= penalty
            penalties['Drawdown'] = f"-{penalty} (DD: {dd:.1%})"

        pct_beta = rec.get('pct_beta_3m', 0.5)
        if pct_beta > 0.8:
            base_score -= 0.5
            penalties['High Beta'] = "-0.5"

        final_score = max(0, round(base_score, 2))
        
        return {
            'symbol': symbol, 
            'score': final_score, 
            'hard_reject': False,
            'penalties': penalties
        }
    
# %%
def calculate_risk_metrics(df, symbol=None):
        if df is None or len(df) < 20:
            return {}
    
        close = df['close'].dropna()
        if len(close) < 20:
            return {}
    
        metrics = {}
        metrics['close_last'] = close.iloc[-1]
    
        lookback_3m = min(60, len(close))
        prices_3m = close.iloc[-lookback_3m:]
        running_max = prices_3m.expanding().max()
        drawdown = (prices_3m - running_max) / running_max
        max_dd = drawdown.min()
        metrics['dd_3m'] = abs(max_dd) if max_dd < 0 else 0.0
    
        if len(prices_3m) >= 2:
            log_returns = np.log(prices_3m / prices_3m.shift(1)).dropna()
            vol_3m = log_returns.std() * np.sqrt(252)  # Annualize by 252 trading days
            metrics['vol_3m'] = vol_3m
    
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
        for col in ['volume', 'Volume', 'vol']:
            if col in df.columns:
                volume_col = col
                break
    
        if volume_col is not None:
            lookback_20d = min(20, len(df))
            prices_20d = close.iloc[-lookback_20d:]
            volume_20d = df[volume_col].iloc[-lookback_20d:]
        
            # Daily value traded = price * volume
            daily_value = prices_20d.values * volume_20d.values
            adv_20 = np.mean(daily_value)  # Average daily value (VND)
            metrics['adv_20'] = adv_20
        
        return metrics


def calculate_drawdown_3m(df):
    metrics = calculate_risk_metrics(df)
    return metrics.get('dd_3m')


def calculate_risk_metrics_batch(symbol_list, stock_data, start_date=None, end_date=None):
    records = []

    # Attempt to fetch beta metrics in one call
    beta_map = {}
    try:
        beta_params = {}
        if start_date:
            beta_params['start_date'] = start_date
        if end_date:
            beta_params['end_date'] = end_date
        resp = requests.get(
            "http://192.168.8.190:8000/MKD/beta_calculation",
            params=beta_params if beta_params else None,
            timeout=60
        )
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
        print(f"Risk beta fetch error: {e}")

    for sym in symbol_list:
        df = stock_data.get(sym)
        if df is None or len(df) == 0:
            continue

        rec = {'symbol': sym}
        rec.update(beta_map.get(sym, {}))
        rec.update(calculate_risk_metrics(df, sym))
        records.append(rec)

    # Add peer percentiles (lower is better for these metrics)
    for key, pct_key in [('beta_3m', 'pct_beta_3m'), ('dd_3m', 'pct_dd_3m'), ('vol_3m', 'pct_vol_3m')]:
        vals = [r[key] for r in records if r.get(key) is not None and not np.isnan(r.get(key))]
        if not vals:
            continue
        for rec in records:
            val = rec.get(key)
            if val is not None and not np.isnan(val):
                rank = sum(1 for v in vals if v < val)
                rec[pct_key] = rank / len(vals)

    return records

# %%
def analyze_symbols(symbol_list, start_date, end_date, industry='Real Estate', year=2025, quarter=3):    
    bluechip_data = get_cached_bluechips(industry, symbol_list, year, quarter)
    mega_caps = bluechip_data.get('mega_caps', [])
    large_caps = bluechip_data.get('large_caps', [])
    
    # 1. Fetch Stock Data
    stock_data = APIFetcher.fetch_batch(
        "http://192.168.8.190:8000/MKD/stock_daily", 
        symbol_list, start_date, end_date
    )
    
    # 2. Fetch VNINDEX
    vni_dict = APIFetcher.fetch_batch(
        "http://192.168.8.190:8000/MKD/stock_index_daily", 
        ["VNINDEX"], start_date, end_date
    )
    vni_series = None
    if "VNINDEX" in vni_dict:
        vni_df = vni_dict["VNINDEX"]
        col = 'close' if 'close' in vni_df else 'adj_close'
        vni_series = vni_df[col]

    # 3. Technical Scoring
    tech_scorer = TechnicalAnalysisScorer()
    tech_scores = {}
    for sym in symbol_list:
        df = stock_data.get(sym)
        if df is not None:
            df_tech = calculate_technical_indicators_fallback(df)
            if df_tech is not None:
                tech_scores[sym] = tech_scorer.score_symbol(sym, df_tech, vni_series)

    # 4. Foreign Scoring
    fi_data = calculate_foreign_investor_metrics(symbol_list, start_date, end_date)
    fi_scorer = ForeignInvestorScorer(mega_caps=mega_caps, large_caps=large_caps)
    foreign_scores = {}
    for sym, data in fi_data.items():
        foreign_scores[sym] = fi_scorer.score_symbol(sym, data)

    # 5. Risk Scoring
    risk_scorer = RiskControlScorer()
    risk_records = calculate_risk_metrics_batch(symbol_list, stock_data, start_date, end_date)
    risk_map = {r['symbol']: r for r in risk_records}
    
    risk_scores = {}
    for sym in symbol_list:
        if sym in risk_map:
            risk_scores[sym] = risk_scorer.score_symbol(sym, risk_map[sym])
        else:
            risk_scores[sym] = {'symbol': sym, 'score': 0}

    return tech_scores, foreign_scores, risk_scores