# %%
import requests
import pandas as pd
import numpy as np

START_DATE = "2025-09-01"
END_DATE = "2025-12-31"

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
class TechnicalAnalysisScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'MA order uptrend', 'type': 'ma_order', 'must_have': True},
            {'id': 2, 'name': 'MACD positive', 'type': 'macd_status', 'must_have': True},
            {'id': 3, 'name': 'RSI strength', 'type': 'rsi_status', 'must_have': False},
            {'id': 4, 'name': 'Price above MA20', 'type': 'price_vs_ma20', 'must_have': False},
            {'id': 5, 'name': 'Price stability', 'type': 'price_in_bands', 'must_have': False},
            {'id': 6, 'name': 'Relative strength', 'type': 'relative_strength', 'must_have': False},
        ]
    
    def evaluate_criterion(self, tech_df, criterion, vni_series=None):
        try:
            latest = tech_df.iloc[-1]
            crit_type = criterion['type']
            
            if crit_type == 'ma_order':
                return {'pass': latest.get('ma20') > latest.get('ma50') > latest.get('ma200')}
            elif crit_type == 'macd_status':
                return {'pass': latest.get('macd12269') > latest.get('macds12269') and latest.get('macd12269') > 0}
            elif crit_type == 'rsi_status':
                return {'pass': latest.get('rsi14') > 50}
            elif crit_type == 'price_vs_ma20':
                return {'pass': (tech_df['close'].tail(20) > tech_df['ma20'].tail(20)).sum() >= 15}
            elif crit_type == 'price_in_bands':
                return {'pass': latest.get('bbl2020') < latest.get('close') < latest.get('bbu2020')}
            elif crit_type == 'relative_strength':
                if vni_series is None: return {'pass': False}
                common = tech_df.index.intersection(vni_series.index)
                if len(common) < 10: return {'pass': False}
                s_valid = tech_df.loc[common, 'close']
                v_valid = vni_series.loc[common]
                rs = np.log(s_valid) - np.log(v_valid)
                return {'pass': rs.iloc[-1] > rs.iloc[0]}
            return {'pass': False}
        except: return {'pass': False}

    def score_symbol(self, symbol, tech_df, vni_series=None):
        points = 0
        results = {}
        must_haves = [c for c in self.criteria if c['must_have']]
        passed_must = []
        for c in must_haves:
            res = self.evaluate_criterion(tech_df, c, vni_series)
            passed_must.append(res['pass'])
            results[f"C{c['id']}"] = res
        
        if all(passed_must): points += 2
        
        optionals = [c for c in self.criteria if not c['must_have']]
        for c in optionals:
            res = self.evaluate_criterion(tech_df, c, vni_series)
            results[f"C{c['id']}"] = res
            if res['pass']: points += 3/len(optionals)
        return {'symbol': symbol, 'score': min(points, 5), 'details': results}

# %%
class ForeignInvestorScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'NFF supportive (5D)', 'code': 'pct_nff_5d', 'direction': 'higher_better', 'must_have': True, 'threshold': 0.70},
            {'id': 2, 'name': 'FFR buy bias (5D)', 'code': 'pct_ffr_5d', 'direction': 'higher_better', 'must_have': True, 'threshold': 0.70},
            {'id': 3, 'name': 'Room decreasing (5D)', 'code': 'pct_room_delta_5d', 'direction': 'lower_better', 'must_have': False, 'threshold': 0.30},
            {'id': 4, 'name': 'NFF strong (1D)', 'code': 'pct_nff_1d', 'direction': 'higher_better', 'must_have': False, 'threshold': 0.70},
            {'id': 5, 'name': 'FFR buy bias (1D)', 'code': 'pct_ffr_1d', 'direction': 'higher_better', 'must_have': False, 'threshold': 0.70},
            {'id': 6, 'name': 'Room decreasing (1D)', 'code': 'pct_room_delta_1d', 'direction': 'lower_better', 'must_have': False, 'threshold': 0.30},
        ]
        
    def score_symbol(self, symbol, fi_dict):
        if not fi_dict: return {'symbol': symbol, 'score': 0}
        points = 0
        must_pass = True
        
        for c in self.criteria:
            val = fi_dict.get(c['code']) # Lấy giá trị Percentile đã tính
            passed = False
            if val is not None:
                passed = val >= c['threshold'] if c['direction'] == 'higher_better' else val <= c['threshold']
            
            if c['must_have'] and not passed: must_pass = False
            if not c['must_have'] and passed: points += 3/4
            
        if must_pass: points += 2
        return {'symbol': symbol, 'score': min(points, 5)}

# %%
class RiskControlScorer:
    def __init__(self):
        self.dd_hard = 0.35
        self.beta_hard = 2.5
        
    def percentile_to_score(self, percentile):
        if percentile <= 30: return 1.0
        elif percentile <= 60: return 0.7
        elif percentile <= 80: return 0.4
        else: return 0.1 # Rủi ro cao

    def score_symbol(self, symbol, rec, all_records=None):
        dd = rec.get('dd_3m', 0)
        beta = rec.get('beta_3m', 1)
        if dd > self.dd_hard or beta > self.beta_hard:
            return {'symbol': symbol, 'score': 0, 'hard_reject': True}
        
        scores = []
        for key in ['pct_beta_3m', 'pct_dd_3m', 'pct_vol_3m']:
            val = rec.get(key)
            if val is not None:
                scores.append(self.percentile_to_score(val * 100)) # val là 0-1, function nhận 0-100
        
        base_score = np.mean(scores) * 5.0 if scores else 2.5 # Quy về thang 5
        
        if dd > 0.25: base_score = min(base_score, 3.0)
        if beta > 2.0: base_score = min(base_score, 3.0)
        
        return {'symbol': symbol, 'score': max(0, round(base_score, 2)), 'hard_reject': False}

# %%
def calculate_foreign_investor_metrics(symbol_list, start_date, end_date):
    all_data = {}
    
    for sym in symbol_list:
        try:
            params = {"symbol": sym, "start_date": start_date, "end_date": end_date}
            r = requests.get("http://192.168.8.190:8000/MKD/foreign_investors", params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data:
                    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
                    for dc in ['date', 'trading_date']: 
                        if dc in df.columns: 
                            df[dc] = pd.to_datetime(df[dc]); df=df.sort_values(dc); break
                    
                    df['nff'] = df['buy_value'] - df['sell_value']
                    df['ffr'] = df['buy_value'] / (df['buy_value'] + df['sell_value']).replace(0, np.nan)
                    df['room_delta'] = df['current_room'].diff()
                    
                    if len(df) >= 5:
                        nff_5d = df['nff'].tail(5).sum()
                        ffr_5d = df['ffr'].tail(5).mean()
                        room_delta_5d = df['current_room'].iloc[-1] - df['current_room'].iloc[-5]
                    else:
                        nff_5d = df['nff'].sum()
                        ffr_5d = df['ffr'].mean()
                        room_delta_5d = 0
                        
                    latest = {
                        'nff_1d': df['nff'].iloc[-1],
                        'ffr_1d': df['ffr'].iloc[-1],
                        'room_delta_1d': df['room_delta'].iloc[-1],
                        'nff_5d': nff_5d,
                        'ffr_5d': ffr_5d,
                        'room_delta_5d': room_delta_5d
                    }
                    all_data[sym] = latest
        except: pass
        
    metrics = ['nff_5d', 'ffr_5d', 'room_delta_5d', 'nff_1d', 'ffr_1d', 'room_delta_1d']
    for m in metrics:
        vals = [d[m] for d in all_data.values() if d.get(m) is not None and not np.isnan(d.get(m))]
        if not vals: continue
        
        for sym in all_data:
            val = all_data[sym].get(m)
            if val is not None and not np.isnan(val):
                rank = sum(1 for v in vals if v < val)
                pct = rank / len(vals) if len(vals) > 0 else 0.5
                all_data[sym][f'pct_{m}'] = pct
                
    return all_data

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
    
        # 2. ANNUALIZED VOLATILITY 3M (60 trading days)
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
    
        # 4. AVERAGE DAILY VALUE TRADED 20D (ADV)
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


def calculate_risk_metrics_batch(symbol_list, stock_data):
    records = []

    # Attempt to fetch beta metrics in one call
    beta_map = {}
    try:
        resp = requests.get("http://192.168.8.190:8000/MKD/beta_calculation", timeout=60)
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
def analyze_symbols(symbol_list, start_date, end_date):
    print(f"Starting Technical & Risk Analysis for {len(symbol_list)} symbols...")
    
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
    print("Calculating Technical Scores...")
    tech_scorer = TechnicalAnalysisScorer()
    tech_scores = {}
    for sym in symbol_list:
        df = stock_data.get(sym)
        if df is not None:
            df_tech = calculate_technical_indicators_fallback(df)
            if df_tech is not None:
                tech_scores[sym] = tech_scorer.score_symbol(sym, df_tech, vni_series)

    # 4. Foreign Scoring
    print("Calculating Foreign Scores...")
    fi_data = calculate_foreign_investor_metrics(symbol_list, start_date, end_date)
    fi_scorer = ForeignInvestorScorer()
    foreign_scores = {}
    for sym, data in fi_data.items():
        foreign_scores[sym] = fi_scorer.score_symbol(sym, data)

    # 5. Risk Scoring
    print("Calculating Risk Scores...")
    risk_scorer = RiskControlScorer()
    risk_records = calculate_risk_metrics_batch(symbol_list, stock_data)
    # Map record theo symbol để dễ tra cứu
    risk_map = {r['symbol']: r for r in risk_records}
    
    risk_scores = {}
    for sym in symbol_list:
        if sym in risk_map:
            risk_scores[sym] = risk_scorer.score_symbol(sym, risk_map[sym], risk_records)
        else:
            risk_scores[sym] = {'symbol': sym, 'score': 0}

    return tech_scores, foreign_scores, risk_scores