# %%
import requests
import pandas as pd
import numpy as np

START_DATE = "2025-09-01"
END_DATE = "2025-12-31"

tech_scores = {}
foreign_scores = {}
risk_scores = {}

r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
r.raise_for_status()
all_stocks = r.json()
banks = [s['symbol'] for s in all_stocks if s.get('industry_lv1') == 'Banks']

# %% API FETCHER ==============================================
class APIFetcher:    
    @staticmethod
    def fetch_batch(url, symbol_list, start_date, end_date, batch_size=None, 
                    delay_between_batches=None, max_retries=3, timeout_secs=60):
        all_data = {}
        total_symbols = len(symbol_list)
        print(f"Fetching {total_symbols} symbols in one request...")

        try:
            params = {
                "symbols": ",".join(symbol_list),
                "start_date": start_date,
                "end_date": end_date
            }

            resp = requests.get(url, params=params, timeout=timeout_secs)
            resp.raise_for_status()
            batch_data = resp.json()
            print(f"Got {len(batch_data)} records")

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
            print(f"Error fetching batch: {type(e).__name__}: {e}")
        
        result = {}
        for symbol, data in all_data.items():
            try:
                df = pd.DataFrame(data) if isinstance(data, list) else data
                
                for date_col in ['date', 'Date', 'trading_date', 'TradingDate']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.sort_values(date_col)
                        df = df.set_index(date_col)  # Set date as index for proper alignment
                        break
                
                result[symbol] = df
            except Exception as e:
                print(f"Warning: Could not convert {symbol} to DataFrame: {e}")
        
        return result

def calculate_technical_indicators_fallback(df, symbol=None):
    if df is None or len(df) == 0:
        return None

    close_col = None
    for c in ['adj_close', 'close', 'Close']:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        return None

    x = df.copy()
    x['close'] = pd.to_numeric(x[close_col], errors='coerce')

    # Moving averages
    x['ma10'] = x['close'].rolling(10).mean()
    x['ma20'] = x['close'].rolling(20).mean()
    x['ma50'] = x['close'].rolling(50).mean()
    x['ma100'] = x['close'].rolling(100).mean()
    x['ma200'] = x['close'].rolling(200).mean()

    # MACD (12,26,9)
    exp12 = x['close'].ewm(span=12, adjust=False).mean()
    exp26 = x['close'].ewm(span=26, adjust=False).mean()
    x['macd12269'] = exp12 - exp26
    x['macds12269'] = x['macd12269'].ewm(span=9, adjust=False).mean()
    x['macdh12269'] = x['macd12269'] - x['macds12269']

    # RSI(14)
    delta = x['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    x['rsi14'] = 100 - (100 / (1 + rs))

    # Bollinger(20, 2)
    x['bbm2020'] = x['close'].rolling(20).mean()
    std20 = x['close'].rolling(20).std()
    x['bbu2020'] = x['bbm2020'] + 2 * std20
    x['bbl2020'] = x['bbm2020'] - 2 * std20

    return x

# %% 
# TECHNICAL ANALYSIS SCORING ==================================================
class TechnicalAnalysisScorer:
    def __init__(self):
        self.criteria = [
            # MUST-HAVE: Trend confirmation (2 points if all pass)
            {'id': 1, 'name': 'MA order uptrend', 'type': 'ma_order', 'rule': 'MA20 > MA50 > MA200', 'must_have': True},
            {'id': 2, 'name': 'MACD positive', 'type': 'macd_status', 'rule': 'MACD > Signal and MACD > 0', 'must_have': True},
            
            # OPTIONAL: Quality indicators (distribute 3 points)
            {'id': 3, 'name': 'RSI strength', 'type': 'rsi_status', 'rule': 'RSI > 50', 'must_have': False},
            {'id': 4, 'name': 'Price above MA20', 'type': 'price_vs_ma20', 'rule': 'Recent price > MA20', 'must_have': False},
            {'id': 5, 'name': 'Price stability', 'type': 'price_in_bands', 'rule': 'Price within Bollinger Bands', 'must_have': False},
            {'id': 6, 'name': 'Relative strength', 'type': 'relative_strength', 'rule': 'RS outperforming index', 'must_have': False},
        ]
    
    def evaluate_criterion(self, tech_df, criterion, vni_series=None):
        try:
            latest = tech_df.iloc[-1]
            crit_type = criterion['type']
            
            # MA order criterion - use actual API column names: ma20, ma50, ma200
            if crit_type == 'ma_order':
                ma20 = latest.get('ma20')
                ma50 = latest.get('ma50')
                ma200 = latest.get('ma200')
                if pd.notna(ma20) and pd.notna(ma50) and pd.notna(ma200):
                    return {'pass': ma20 > ma50 > ma200}
                return {'pass': False, 'reason': 'Missing MA values'}
            
            # MACD status criterion - use actual API column names: macd12269, macds12269
            elif crit_type == 'macd_status':
                macd = latest.get('macd12269')  # Macd line (12, 26, 9)
                signal = latest.get('macds12269')  # Macd signal line (12, 26, 9)
                if pd.notna(macd) and pd.notna(signal):
                    return {'pass': macd > signal and macd > 0}
                return {'pass': False, 'reason': 'Missing MACD values'}
            
            # RSI strength criterion - use API column: rsi14
            elif crit_type == 'rsi_status':
                rsi = latest.get('rsi14')
                if pd.notna(rsi):
                    return {'pass': rsi > 50}
                return {'pass': False, 'reason': 'Missing RSI'}
            
            # Price above MA20 criterion
            elif crit_type == 'price_vs_ma20':
                price = latest.get('close')
                ma20 = latest.get('ma20')
                if pd.notna(price) and pd.notna(ma20):
                    recent_above = (tech_df['close'].tail(20) > tech_df['ma20'].tail(20)).sum()
                    return {'pass': recent_above >= 15}  # 75% of recent period
                return {'pass': False, 'reason': 'Missing data'}
            
            # Price within bands criterion - use API columns: bbu2020, bbl2020
            elif crit_type == 'price_in_bands':
                price = latest.get('close')
                bb_upper = latest.get('bbu2020')  # Bollinger upper (20, 2.0)
                bb_lower = latest.get('bbl2020')  # Bollinger lower (20, 2.0)
                if pd.notna(price) and pd.notna(bb_upper) and pd.notna(bb_lower):
                    return {'pass': bb_lower < price < bb_upper}
                return {'pass': False, 'reason': 'Missing band data'}
            
            # Relative strength criterion
            elif crit_type == 'relative_strength':
                if vni_series is None or len(vni_series) == 0:
                    return {'pass': False, 'reason': 'No index data'}
                
                stock_close = tech_df['close']
                
                common_dates = stock_close.index.intersection(vni_series.index)
                if len(common_dates) < 10:
                    return {'pass': False, 'reason': f'Insufficient overlap: {len(common_dates)} dates'}
                
                stock_aligned = stock_close[common_dates]
                vni_aligned = vni_series[common_dates]
                
                # Filter out zero/negative values before log
                valid_mask = (stock_aligned > 0) & (vni_aligned > 0)
                if valid_mask.sum() < 10:
                    return {'pass': False, 'reason': 'Insufficient valid data for RS'}
                
                stock_valid = stock_aligned[valid_mask]
                vni_valid = vni_aligned[valid_mask]
                rs = (np.log(stock_valid) - np.log(vni_valid)).dropna()
                
                if len(rs) > 1:
                    rs_trend = rs.iloc[-1] > rs.iloc[0]
                    return {'pass': rs_trend, 'rs_change': float(rs.iloc[-1] - rs.iloc[0])}
                return {'pass': False, 'reason': 'Insufficient RS data'}
            
            return {'pass': False, 'reason': 'Unknown criterion'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_bank(self, symbol, tech_df, vni_series=None):
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(tech_df, crit, vni_series)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(tech_df, crit, vni_series)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results
        }


# %%
# FOREIGN INVESTOR SCORING ==================================================
class ForeignInvestorScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'NFF supportive (5D) – Top 30%', 'code': 'nff_5d', 'type': 'peer_percentile', 'direction': 'higher_better', 'must_have': True, 'threshold': 0.70},
            {'id': 2, 'name': 'FFR buy bias (5D) – Top 30%', 'code': 'pct_ffr_5d', 'type': 'peer_percentile', 'direction': 'higher_better', 'must_have': True, 'threshold': 0.70},
            {'id': 3, 'name': 'Room decreasing (5D) – Bottom 30%', 'code': 'pct_room_delta_5d', 'type': 'peer_percentile', 'direction': 'lower_better', 'must_have': False, 'threshold': 0.30},
            {'id': 4, 'name': 'NFF strong (1D) – Top 30%', 'code': 'pct_nff_1d', 'type': 'peer_percentile', 'direction': 'higher_better', 'must_have': False, 'threshold': 0.70},
            {'id': 5, 'name': 'FFR buy bias (1D) – Top 30%', 'code': 'pct_ffr_1d', 'type': 'peer_percentile', 'direction': 'higher_better', 'must_have': False, 'threshold': 0.70},
            {'id': 6, 'name': 'Room decreasing (1D) – Bottom 30%', 'code': 'pct_room_delta_1d', 'type': 'peer_percentile', 'direction': 'lower_better', 'must_have': False, 'threshold': 0.30},
        ]

    def evaluate_criterion(self, fi_dict, criterion):
        try:
            code = criterion['code']
            pct_value = fi_dict.get(code)

            if pct_value is None:
                return {'pass': False, 'reason': f"No {code} data"}

            threshold = criterion.get('threshold')
            direction = criterion.get('direction')

            if direction == 'higher_better':
                pass_check = pct_value >= threshold
            elif direction == 'lower_better':
                pass_check = pct_value <= threshold
            else:
                return {'pass': False, 'reason': 'Unknown direction'}

            return {'pass': pass_check, 'value': pct_value, 'threshold': threshold}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score_bank(self, symbol, fi_dict):
        if not fi_dict or not isinstance(fi_dict, dict):
            return {'symbol': symbol, 'score': 0, 'reason': 'Invalid data'}
    
        results = {}
        points = 0
    
        # Must-have criteria
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(fi_dict, crit)
                must_have_results.append(result.get('pass', False))
                results[f"C{crit['id']}"] = result
    
        if must_have_results and all(must_have_results):
            points += 2
    
        # Optional criteria
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        if optional_criteria:
            for crit in optional_criteria:
                result = self.evaluate_criterion(fi_dict, crit)
                results[f"C{crit['id']}"] = result
                if result.get('pass', False):
                    points += 3 / len(optional_criteria)
    
        return {
            'symbol': symbol,
            'score': round(min(points, 5), 2),
            'details': results
        }


# %%
# Calculate foreign investor metrics
def calculate_foreign_investor_metrics(symbol_list, start_date, end_date):
    all_symbol_data = {}
    max_retries = 3
    skipped = []
    errors = []
    
    for idx, symbol in enumerate(symbol_list):
        success = False
    
        for attempt in range(max_retries):
            try:
                params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
                resp = requests.get("http://192.168.8.190:8000/MKD/foreign_investors", 
                                   params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            
                if not data:
                    print(f"  {symbol}: Empty response")
                    skipped.append((symbol, "empty response"))
                    break
            
                # Convert to DataFrame
                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
            
                # Ensure we have required columns
                required_cols = ['buy_value', 'sell_value', 'current_room']
                if not all(col in df.columns for col in required_cols):
                    reason = "missing required columns"
                    print(f"  {symbol}: {reason}")
                    skipped.append((symbol, reason))
                    break
            
                # Parse date column
                for date_col in ['date', 'Date', 'trading_date', 'TradingDate']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.sort_values(date_col).reset_index(drop=True)
                        break
            
                # Calculate 1D metrics
                df['nff_1d'] = df['buy_value'] - df['sell_value']
                df['ffr_1d'] = df.apply(
                    lambda row: row['buy_value'] / (row['buy_value'] + row['sell_value']) 
                    if (row['buy_value'] + row['sell_value']) != 0 else None, 
                    axis=1
                )
                df['room_delta_1d'] = df['current_room'].diff()
            
                # Calculate 5D aggregates (only if enough data)
                if len(df) >= 5:
                    df['nff_5d'] = df['nff_1d'].rolling(window=5, min_periods=5).sum()
                    df['ffr_5d'] = df['ffr_1d'].rolling(window=5, min_periods=5).mean()
                    df['room_delta_5d'] = df['current_room'] - df['current_room'].shift(5)
                else:
                    df['nff_5d'] = None
                    df['ffr_5d'] = None
                    df['room_delta_5d'] = None
            
                # Store latest record for this symbol
                latest = df.iloc[-1].to_dict()
                latest['symbol'] = symbol
                all_symbol_data[symbol] = latest
                success = True
                break
            
            except requests.HTTPError as e:
                if '429' in str(e):
                    # Rate limit error - wait longer and retry
                    if attempt < max_retries - 1:
                        print(f"  {symbol}: Rate limited. Retrying {attempt+1}/{max_retries-1}...")
                    else:
                        print(f"  {symbol}: Rate limit exceeded after {max_retries} retries")
                        errors.append((symbol, "rate limited"))
                        break
                else:
                    # Other HTTP errors - don't retry
                    print(f"  {symbol}: HTTP Error {str(e)}")
                    errors.append((symbol, f"HTTPError {str(e)}"))
                    break
                
            except Exception as e:
                print(f"  {symbol}: Error calculating - {type(e).__name__}: {str(e)}")
                errors.append((symbol, f"Exception {str(e)}"))
                break
    
    # Brief summary of coverage
    missing = set(symbol_list) - set(all_symbol_data.keys())
    if missing:
        print(f"  Note: {len(missing)} symbols skipped: {', '.join(sorted(missing))}")
    if skipped:
        print("  Skipped details:")
        for sym, msg in skipped:
            print(f"    - {sym}: {msg}")
    if errors:
        print("  Error details:")
        for sym, msg in errors:
            print(f"    - {sym}: {msg}")
    
    # Calculate peer percentiles across all symbols
    if not all_symbol_data:
        return {}

    metrics = ['nff_1d', 'ffr_1d', 'room_delta_1d', 'nff_5d', 'ffr_5d', 'room_delta_5d']

    for metric in metrics:
        values = []
        symbols = []
        for symbol, data in all_symbol_data.items():
            val = data.get(metric)
            if val is not None and pd.notna(val):
                values.append(val)
                symbols.append(symbol)
    
        if not values:
            continue
    
        # Calculate percentile for each symbol
        for symbol in symbols:
            val = all_symbol_data[symbol].get(metric)
            if val is not None and pd.notna(val):
                # Percentile = (rank - 1) / (n - 1) where rank is position in sorted list
                rank = sum(1 for v in values if v < val) + 1
                percentile = (rank - 1) / (len(values) - 1) if len(values) > 1 else 0.5
                all_symbol_data[symbol][f"pct_{metric}"] = percentile

    return all_symbol_data

def run_analysis():
    global tech_scores, foreign_scores, risk_scores, banks
    
    print(f"Analyzing {len(banks)} banks: {', '.join(banks)}")
    print(f"\nFetching data for period: {START_DATE} to {END_DATE}\n")

    stock_data = APIFetcher.fetch_batch(
        url="http://192.168.8.190:8000/MKD/stock_daily",
        symbol_list=banks,
        start_date=START_DATE,
        end_date=END_DATE
    )
    print(f"Stock data fetched: {len(stock_data)} symbols\n")

    # Fetch VNINDEX for relative strength calculation
    print("Fetching VNINDEX...")
    vni_data = None
    vni_series = None  # Initialize as None by default
    try:
        vni_batch = APIFetcher.fetch_batch(
            url="http://192.168.8.190:8000/MKD/stock_index_daily",
            symbol_list=["VNINDEX"],
            start_date=START_DATE,
            end_date=END_DATE,
            batch_size=1
        )
        if "VNINDEX" in vni_batch:
            vni_data = vni_batch["VNINDEX"]
            close_col = 'close' if 'close' in vni_data.columns else 'adj_close'
        
            # vni_data already has DatetimeIndex from fetch_batch, use it directly
            if close_col in vni_data.columns:
                vni_series = pd.Series(
                    vni_data[close_col].values,
                    index=vni_data.index  # Use the DatetimeIndex already set
                )
            print(f"VNINDEX: {len(vni_data)} records\n")
    except Exception as e:
        print(f"VNINDEX error: {e}\n")

    # %%
    # Calculate indicators - FETCH FROM API, fallback to local calculation
    technical_data = {}
    print("Fetching technical indicators from API...")
    
    # Also fetch daily data in case we need fallback
    print("Fetching daily stock data (for fallback)...")
    stock_data = APIFetcher.fetch_batch(
        url="http://192.168.8.190:8000/MKD/stock_daily",
        symbol_list=banks,
        start_date=START_DATE,
        end_date=END_DATE
    )
    print(f"Daily data: {len(stock_data)} symbols\n")
    
    skipped = []
    api_count = 0
    fallback_count = 0
    
    for symbol in banks:
        try:
            params = {"symbol": symbol, "start_date": START_DATE, "end_date": END_DATE}
            resp = requests.get(
                "http://192.168.8.190:8000/MKD/technical_indicators",
                params=params, timeout=20,
            )
            resp.raise_for_status()
            ti_json = resp.json()
            
            if ti_json and len(ti_json) > 0:
                df = pd.DataFrame(ti_json)
                
                # Parse date column and set as index
                for date_col in ['date', 'Date', 'trading_date', 'TradingDate']:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.sort_values(date_col)
                        df = df.set_index(date_col)
                        break
                
                # Ensure 'close' column exists
                if 'close' not in df.columns:
                    for close_col in ['adj_close', 'Close', 'price']:
                        if close_col in df.columns:
                            df['close'] = pd.to_numeric(df[close_col], errors='coerce')
                            break
                
                if len(df) > 0:
                    technical_data[symbol] = df
                    api_count += 1
                    continue
        except Exception as e:
            pass
        
        try:
            src = stock_data.get(symbol)
            if src is not None and len(src) > 0:
                tech_df = calculate_technical_indicators_fallback(src, symbol)
                if tech_df is not None and len(tech_df) > 0:
                    technical_data[symbol] = tech_df
                    fallback_count += 1
                else:
                    skipped.append(symbol)
            else:
                skipped.append(symbol)
        except Exception as e:
            skipped.append(symbol)
    
    if skipped:
        print(f"Skipped: {', '.join(skipped)}\n")

    # %%
    # Score technical indicators
    scorer = TechnicalAnalysisScorer()
    tech_scores.clear()
    
    try:
        _ = vni_series
    except NameError:
        vni_series = None

    for symbol in sorted(technical_data.keys()):
        try:
            tech_df = technical_data[symbol]
            score_result = scorer.score_bank(symbol, tech_df, vni_series)
            tech_scores[symbol] = score_result
        except Exception as e:
            print(f"{symbol}: Scoring error - {str(e)}")

    # %%
    # Fetch and score foreign investor data
    foreign_scores.clear()
    try:
        print("Fetching foreign investor data for all banks...")
        fi_data_dict = calculate_foreign_investor_metrics(banks, START_DATE, END_DATE)

        if not fi_data_dict:
            print("Warning: No foreign investor data calculated\n")
        else:
            print(f"  Calculated metrics for {len(fi_data_dict)} banks (requested {len(banks)})\n")

            scorer_fi = ForeignInvestorScorer()
            for symbol, fi_record in fi_data_dict.items():
                try:
                    score_result = scorer_fi.score_bank(symbol, fi_record)
                    foreign_scores[symbol] = score_result
                except Exception as e:
                    print(f"  {symbol}: Scoring error - {type(e).__name__}: {str(e)}")

            print(f"Foreign Investor scores calculated for {len(foreign_scores)} banks\n")

    except requests.Timeout:
        print("Timeout: Foreign investor request timed out\n")
    except Exception as e:
        print(f"Foreign Investor error: {type(e).__name__}: {e}\n")

    # %% CALCULATE RISK METRICS FROM STOCK DATA
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

    # %%
    # RISK CONTROL SCORING ===================================================
    class RiskControlScorer:    
        def __init__(self, 
                     # Hard reject thresholds
                     dd_3m_hard_reject=0.35, beta_3m_hard_reject=2.5, vol_3m_hard_reject=0.90, adv_20_min=None,
                     # Soft cap thresholds
                     dd_3m_soft_cap_lo=0.25, dd_3m_soft_cap_hi=0.35,
                     beta_3m_soft_cap_lo=2.0, beta_3m_soft_cap_hi=2.5,
                     beta_1m_spike_cap=2.8):
            # Hard reject
            self.dd_3m_hard_reject = dd_3m_hard_reject
            self.beta_3m_hard_reject = beta_3m_hard_reject
            self.vol_3m_hard_reject = vol_3m_hard_reject
            self.adv_20_min = adv_20_min  # Minimum ADV in VND (e.g., 10e9 for 10B/day)
        
            # Soft cap
            self.dd_3m_soft_cap_lo = dd_3m_soft_cap_lo
            self.dd_3m_soft_cap_hi = dd_3m_soft_cap_hi
            self.beta_3m_soft_cap_lo = beta_3m_soft_cap_lo
            self.beta_3m_soft_cap_hi = beta_3m_soft_cap_hi
            self.beta_1m_spike_cap = beta_1m_spike_cap

        @staticmethod
        def percentile_to_score(percentile):
            if percentile <= 30:
                return 1.0
            elif percentile <= 60:
                return 0.7
            elif percentile <= 80:
                return 0.4
            else:
                return 0.1

        def evaluate(self, rec, all_records=None):
            # Extract metrics
            b1 = rec.get('beta_1m') or rec.get('beta_1M')
            b3 = rec.get('beta_3m') or rec.get('beta_3M')
            b12 = rec.get('beta_12m') or rec.get('beta_12M')
            dd3 = rec.get('dd_3m') or rec.get('drawdown_3m')
            vol3 = rec.get('vol_3m') or rec.get('volatility_3m')
            var1 = rec.get('var_1m')
            cvar1 = rec.get('cvar_1m')
            adv = rec.get('adv_20')

            # Ensure dd3 is positive
            if dd3 is not None and dd3 < 0:
                dd3 = abs(dd3)

            results = {}

            # ===== HARD REJECT CHECKS =====
            hard_reject_reasons = []
        
            if dd3 is not None and dd3 > self.dd_3m_hard_reject:
                hard_reject_reasons.append(f"dd_3m={dd3:.2%} > {self.dd_3m_hard_reject:.2%}")
        
            if b3 is not None and b3 > self.beta_3m_hard_reject:
                hard_reject_reasons.append(f"beta_3m={b3:.2f} > {self.beta_3m_hard_reject:.2f}")
        
            if vol3 is not None and vol3 > self.vol_3m_hard_reject:
                hard_reject_reasons.append(f"vol_3m={vol3:.2%} > {self.vol_3m_hard_reject:.2%}")
        
            if self.adv_20_min is not None and adv is not None and adv < self.adv_20_min:
                hard_reject_reasons.append(f"adv_20={adv:.0f} < {self.adv_20_min:.0f}")

            results['hard_reject'] = len(hard_reject_reasons) > 0
            results['hard_reject_reasons'] = hard_reject_reasons

            # ===== SOFT CAP CHECKS =====
            soft_cap_score = 5.0  # Start at max
            soft_cap_reasons = []

            if dd3 is not None and self.dd_3m_soft_cap_lo < dd3 <= self.dd_3m_soft_cap_hi:
                soft_cap_score = min(soft_cap_score, 3.0)
                soft_cap_reasons.append(f"dd_3m={dd3:.2%} in ({self.dd_3m_soft_cap_lo:.2%}, {self.dd_3m_soft_cap_hi:.2%}] → cap 3.0")

            if b3 is not None and self.beta_3m_soft_cap_lo < b3 <= self.beta_3m_soft_cap_hi:
                soft_cap_score = min(soft_cap_score, 3.0)
                soft_cap_reasons.append(f"beta_3m={b3:.2f} in ({self.beta_3m_soft_cap_lo:.2f}, {self.beta_3m_soft_cap_hi:.2f}] → cap 3.0")

            if b1 is not None and b1 > self.beta_1m_spike_cap:
                soft_cap_score = min(soft_cap_score, 3.5)
                soft_cap_reasons.append(f"beta_1m={b1:.2f} > {self.beta_1m_spike_cap:.2f} → cap 3.5")

            results['soft_cap_score'] = soft_cap_score
            results['soft_cap_reasons'] = soft_cap_reasons

            # ===== PERCENTILE SCORING (if all_records provided) =====
            percentile_score = 0.0
            if all_records is not None and len(all_records) > 1:
                # Collect risk features from all records
                b3_vals = [r.get('beta_3m') or r.get('beta_3M') for r in all_records if r.get('beta_3m') or r.get('beta_3M')]
                dd3_vals = [abs(r.get('dd_3m') or r.get('drawdown_3m') or 0) for r in all_records if r.get('dd_3m') or r.get('drawdown_3m')]
                vol3_vals = [r.get('vol_3m') or r.get('volatility_3m') for r in all_records if r.get('vol_3m') or r.get('volatility_3m')]

                scores = []

                # Beta 3M percentile (lower = better)
                if b3 is not None and len(b3_vals) > 0:
                    pct_b3 = sum(1 for v in b3_vals if v < b3) / len(b3_vals) * 100
                    score_b3 = self.percentile_to_score(pct_b3)
                    scores.append(score_b3)

                # Drawdown 3M percentile (lower = better)
                if dd3 is not None and len(dd3_vals) > 0:
                    pct_dd3 = sum(1 for v in dd3_vals if v < dd3) / len(dd3_vals) * 100
                    score_dd3 = self.percentile_to_score(pct_dd3)
                    scores.append(score_dd3)

                # Volatility 3M percentile (lower = better)
                if vol3 is not None and len(vol3_vals) > 0:
                    pct_vol3 = sum(1 for v in vol3_vals if v < vol3) / len(vol3_vals) * 100
                    score_vol3 = self.percentile_to_score(pct_vol3)
                    scores.append(score_vol3)

                # Average percentile score (0-1)
                if scores:
                    percentile_score = np.mean(scores)

            results['percentile_score'] = percentile_score

            return results

        def score_bank(self, symbol, rec, all_records=None):
            res = self.evaluate(rec, all_records)

            # If hard reject, score = 0
            if res['hard_reject']:
                final_score = 0.0
                reason = "; ".join(res['hard_reject_reasons'])
            else:
                # Base score from percentile (0-1 -> 0-5)
                base_score = res['percentile_score'] * 5.0
            
                # Apply soft cap
                final_score = min(base_score, res['soft_cap_score'])

            return {
                'symbol': symbol,
                'score': round(final_score, 2),
                'details': res,
                'hard_reject': res['hard_reject'],
                'soft_cap': res['soft_cap_score'] < 5.0
            }

    #%%
    risk_scores.clear()

    try:
        print("Fetching risk control metrics (beta/drawdown/vol/var/adv)...")
        risk_json = requests.get("http://192.168.8.190:8000/MKD/beta_calculation", timeout=60).json()
        recs = risk_json if isinstance(risk_json, list) else [risk_json]
        df_risk = pd.DataFrame(recs)

        sym_col = None
        for c in ['symbol', 'Symbol', 'ticker', 'Ticker']:
            if c in df_risk.columns:
                sym_col = c
                break
        if sym_col is None:
            df_risk['symbol'] = None
            sym_col = 'symbol'

        # First pass: collect all records with calculated metrics
        all_risk_records = []
        for sym in banks:
            if sym not in stock_data or stock_data[sym] is None:
                continue
        
            # Get beta data from API
            row = df_risk[df_risk[sym_col] == sym]
            if row.empty:
                continue
            rec = row.iloc[-1].to_dict()
        
            # Calculate all risk metrics from stock data
            stock_metrics = calculate_risk_metrics(stock_data[sym], sym)
            rec.update(stock_metrics)
        
            all_risk_records.append(rec)

        # Second pass: score with percentile context
        scorer_rc = RiskControlScorer()
        for rec in all_risk_records:
            sym = rec.get('symbol') or rec.get('Symbol')
            if sym:
                risk_scores[sym] = scorer_rc.score_bank(sym, rec, all_records=all_risk_records)
    
        print(f"Risk control scores calculated for {len(risk_scores)} banks\n")
    
        # Print summary of hard rejects and soft caps
        hard_reject_count = sum(1 for s in risk_scores.values() if s['hard_reject'])
        soft_cap_count = sum(1 for s in risk_scores.values() if s['soft_cap'])
        if hard_reject_count > 0:
            print(f"  Hard rejects: {hard_reject_count} banks")
        if soft_cap_count > 0:
            print(f"  Soft caps (score reduced): {soft_cap_count} banks\n")
    
    except Exception as e:
        print(f"Risk Control error: {type(e).__name__}: {e}\n")


# %% RUN ANALYSIS IF MAIN MODULE
if __name__ == "__main__":
    run_analysis()
