# %%
import requests
import pandas as pd
import os
import TechAna_DRAFT as TechAna
from Filtering_Stock import filter_stocks_by_price_and_liquidity

START_DATE = TechAna.START_DATE
END_DATE = TechAna.END_DATE

all_stocks = []
bank_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
bank_ratios_data = {}
as_of_date = os.getenv("Fund_AsOf_Date")
as_of_ts = None

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all bank symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    filtered_stocks = [s for s in all_stocks if s.get('industry_lv1') == 'Banks']
    filtered_stocks = filter_stocks_by_price_and_liquidity(
        filtered_stocks,
        min_price=10000,
        min_volume_threshold=20000,
        min_pass_rate=0.50,
        start_date=getattr(TechAna, 'START_DATE', None),
        end_date=getattr(TechAna, 'END_DATE', None),
    )
    bank_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(bank_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }
    
    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    # Convert to DataFrame
    df_ratios = pd.DataFrame(all_ratios_data)

    if not as_of_date:
        try:
            end_dt = pd.to_datetime(getattr(TechAna, 'END_DATE', None), errors = "coerce")
            if pd.notna(end_dt):
                prev_q_end = (end_dt.to_period('Q') - 1).end_time
                as_of_date = prev_q_end.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Guard against look-ahead bias by cutting fundamental data at as_of_date.
    if as_of_date:
        try:
            as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
            if pd.notna(as_of_ts) and not df_ratios.empty and 'date' in df_ratios.columns:
                ratio_dates = pd.to_datetime(df_ratios['date'], errors='coerce')
                df_ratios = df_ratios[ratio_dates.notna() & (ratio_dates <= as_of_ts)].copy()
        except Exception:
            pass

    # Organize by bank
    if 'symbol' in df_ratios.columns:
        for symbol in sorted(df_ratios['symbol'].unique()):
            bank_ratios_data[symbol] = df_ratios[df_ratios['symbol'] == symbol].copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_BANK] Warning: failed to fetch bank ratios: {e}")

# %%
# Liquidity Scoring System (LIQ)
CODE_MAPPING = {
    'ryq57': 'LDR',           # Loan-to-Deposit Ratio
    'casa': 'CASA',           # Current Account Saving Account
    'rtq51': 'Deposit_Growth' # Deposit Growth Rate
}

class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'LDR safety', 'code': 'ryq57', 'type': 'absolute', 'rule': '≤ 1.00', 'threshold': 1.00, 'direction': 'lower', 'must_have': True},
            {'id': 2, 'name': 'CASA minimum', 'code': 'casa', 'type': 'absolute', 'rule': '≥ 0.10', 'threshold': 0.10, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'LDR peer (healthy)', 'code': 'ryq57', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.40', 'cut': 0.40, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'CASA peer (strong)', 'code': 'casa', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Deposit growth level', 'code': 'rtq51', 'type': 'trend_level', 'rule': 'mean_12q ≥ 0 and ≥ peer_median', 'window': 12, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'Deposit growth stability', 'code': 'rtq51', 'type': 'trend_vol', 'rule': 'std_12q ≤ peer_median_std', 'window': 12, 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_value(self, bank_df, code):
        """Extract the latest value for a metric code"""
        metric_data = bank_df[bank_df['code'] == code]
        if len(metric_data) > 0:
            return metric_data['value'].dropna().iloc[-1] if len(metric_data.dropna()) > 0 else None
        return None
    
    def evaluate_criterion(self, bank_df, criterion, all_banks_data, peer_data_cache=None):
        """Evaluate a single criterion for a bank"""
        try:
            code = criterion['code']
            crit_type = criterion['type']
            
            latest_value = self.get_metric_value(bank_df, code)
            
            if latest_value is None:
                return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
            
            # Absolute criterion
            if crit_type == 'absolute':
                if criterion['direction'] == 'lower':
                    pass_check = latest_value <= criterion['threshold']
                else:
                    pass_check = latest_value >= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion['threshold']}
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                if peer_data_cache and code in peer_data_cache:
                    peer_values = peer_data_cache[code]
                else:
                    peer_values = [self.get_metric_value(df, code) for df in all_banks_data.values()]
                    peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = (latest_value <= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (latest_value >= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            # Trend level criterion
            elif crit_type == 'trend_level':
                metric_data = bank_df[bank_df['code'] == code]
                recent_data = metric_data['value'].dropna().tail(criterion['window'])
                if len(recent_data) > 0:
                    mean_val = recent_data.mean()
                    pass_check = mean_val >= 0
                    return {'pass': pass_check, 'mean': mean_val}
            
            # Trend volatility criterion
            elif crit_type == 'trend_vol':
                metric_data = bank_df[bank_df['code'] == code]
                recent_data = metric_data['value'].dropna().tail(criterion['window'])
                if len(recent_data) > 0:
                    std_val = recent_data.std()
                    return {'pass': True, 'std': std_val}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_bank(self, symbol, bank_df, all_banks_data, peer_data_cache=None):
        """Calculate liquidity score for a bank (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': min(points, 5),  # Cap at 5
            'details': results
        }

# %%
# Calculate LIQ scores for all banks
scorer = LiquidityScorer()

liq_peer_data_cache = {}
liq_metric_codes = ['ryq57', 'casa']
for code in liq_metric_codes:
    liq_peer_data_cache[code] = [scorer.get_metric_value(df, code) for df in bank_ratios_data.values()]
    liq_peer_data_cache[code] = [v for v in liq_peer_data_cache[code] if v is not None]

liq_scores = {}

for idx, (symbol, bank_df) in enumerate(bank_ratios_data.items()):
    score_result = scorer.score_bank(symbol, bank_df, bank_ratios_data, liq_peer_data_cache)
    liq_scores[symbol] = score_result

# %%
# Profitability Scoring System (PROF)
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'NIM minimum', 'code': 'ryq44', 'type': 'absolute', 'rule': '≥ 0.025', 'threshold': 0.025, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'CIR max', 'code': 'ryq48', 'type': 'absolute', 'rule': '≤ 0.60', 'threshold': 0.60, 'direction': 'lower', 'must_have': True},
            {'id': 3, 'name': 'NIM peer strength', 'code': 'ryq44', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'CIR peer efficiency', 'code': 'ryq48', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.30', 'cut': 0.30, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'ROE healthy', 'code': 'ryq12', 'type': 'peer_or_abs', 'rule': 'value ≥ peer_median OR value ≥ 0.12', 'abs_threshold': 0.12, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'COF advantage', 'code': 'ryq46', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.40', 'cut': 0.40, 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_value(self, bank_df, code):
        """Extract the latest value for a metric code"""
        metric_data = bank_df[bank_df['code'] == code]
        if len(metric_data) > 0:
            return metric_data['value'].dropna().iloc[-1] if len(metric_data.dropna()) > 0 else None
        return None
    
    def evaluate_criterion(self, bank_df, criterion, all_banks_data, peer_data_cache=None):
        """Evaluate a single criterion for a bank"""
        try:
            code = criterion['code']
            crit_type = criterion['type']
            
            latest_value = self.get_metric_value(bank_df, code)
            
            if latest_value is None:
                return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
            
            # Absolute criterion
            if crit_type == 'absolute':
                if criterion['direction'] == 'lower':
                    pass_check = latest_value <= criterion['threshold']
                else:
                    pass_check = latest_value >= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion['threshold']}
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                if peer_data_cache and code in peer_data_cache:
                    peer_values = peer_data_cache[code]
                else:
                    peer_values = [self.get_metric_value(df, code) for df in all_banks_data.values()]
                    peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = sum(1 for v in peer_values if latest_value <= v) / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = sum(1 for v in peer_values if latest_value >= v) / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            # Peer or Absolute criterion (ROE: either >= peer_median OR >= abs_threshold)
            elif crit_type == 'peer_or_abs':
                if peer_data_cache and code in peer_data_cache:
                    peer_values = peer_data_cache[code]
                else:
                    peer_values = [self.get_metric_value(df, code) for df in all_banks_data.values()]
                    peer_values = [v for v in peer_values if v is not None]
                
                # Check absolute threshold
                abs_pass = latest_value >= criterion['abs_threshold']
                
                # Check peer median
                if len(peer_values) > 0:
                    sorted_peers = sorted(peer_values)
                    peer_median = sorted_peers[len(sorted_peers)//2]
                    peer_pass = latest_value >= peer_median
                    pass_check = abs_pass or peer_pass
                    return {'pass': pass_check, 'value': latest_value, 'abs_threshold': criterion['abs_threshold'], 
                           'peer_median': peer_median, 'abs_pass': abs_pass, 'peer_pass': peer_pass}
                else:
                    # If no peer data, just check absolute
                    return {'pass': abs_pass, 'value': latest_value, 'abs_threshold': criterion['abs_threshold']}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_bank(self, symbol, bank_df, all_banks_data, peer_data_cache=None):
        """Calculate profitability score for a bank (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': min(points, 5),  # Cap at 5
            'details': results
        }

# %%
# Calculate PROF scores for all banks
prof_scorer = ProfitabilityScorer()

prof_peer_data_cache = {}
prof_metric_codes = ['ryq44', 'ryq48', 'ryq46', 'ryq12']
for code in prof_metric_codes:
    prof_peer_data_cache[code] = [prof_scorer.get_metric_value(df, code) for df in bank_ratios_data.values()]
    prof_peer_data_cache[code] = [v for v in prof_peer_data_cache[code] if v is not None]

prof_scores = {}

for idx, (symbol, bank_df) in enumerate(bank_ratios_data.items()):
    score_result = prof_scorer.score_bank(symbol, bank_df, bank_ratios_data, prof_peer_data_cache)
    prof_scores[symbol] = score_result

# %%
# Solvency Scoring System (SOLV)
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'NPL ceiling', 'code': 'ryq58', 'type': 'absolute', 'rule': '≤ 0.04', 'threshold': 0.04, 'direction': 'lower', 'must_have': True},
            {'id': 2, 'name': 'Coverage minimum (LLR/NPL)', 'code': 'ryq59', 'type': 'absolute', 'rule': '≥ 1.00', 'threshold': 1.00, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'NPL peer (clean)', 'code': 'ryq58', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.30', 'cut': 0.30, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'Coverage peer (strong)', 'code': 'ryq59', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Equity buffer proxy', 'code': 'ryq56', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'Provisioning intensity', 'code': 'ryq60', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 7, 'name': 'Credit growth not reckless', 'credit_code': 'rtq50', 'npl_code': 'ryq58', 'type': 'conditional', 'rule': '(credit_growth ≥ peer_median) AND (NPL not worsening)', 'window': 4, 'must_have': False},
        ]
    
    def get_metric_value(self, bank_df, code):
        """Extract the latest value for a metric code"""
        metric_data = bank_df[bank_df['code'] == code]
        if len(metric_data) > 0:
            return metric_data['value'].dropna().iloc[-1] if len(metric_data.dropna()) > 0 else None
        return None
    
    def get_metric_series(self, bank_df, code):
        """Extract all values for a metric code as a series"""
        metric_data = bank_df[bank_df['code'] == code]
        if len(metric_data) > 0:
            return metric_data['value'].dropna()
        return pd.Series()
    
    def evaluate_criterion(self, bank_df, criterion, all_banks_data, peer_data_cache=None):
        """Evaluate a single criterion for a bank"""
        try:
            crit_type = criterion['type']
            
            # Absolute criterion
            if crit_type == 'absolute':
                code = criterion['code']
                latest_value = self.get_metric_value(bank_df, code)
                
                if latest_value is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
                
                if criterion['direction'] == 'lower':
                    pass_check = latest_value <= criterion['threshold']
                else:
                    pass_check = latest_value >= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion['threshold']}
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                code = criterion['code']
                latest_value = self.get_metric_value(bank_df, code)
                
                if latest_value is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
                
                if peer_data_cache and code in peer_data_cache:
                    peer_values = peer_data_cache[code]
                else:
                    peer_values = [self.get_metric_value(df, code) for df in all_banks_data.values()]
                    peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    peer_series = pd.Series(peer_values)
                    if criterion['direction'] == 'lower':
                        pct = sum(1 for v in peer_values if latest_value <= v) / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = sum(1 for v in peer_values if latest_value >= v) / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            # Conditional criterion (Credit growth + NPL stability)
            elif crit_type == 'conditional':
                credit_code = criterion['credit_code']
                npl_code = criterion['npl_code']
                window = criterion['window']
                
                # Check 1: Credit growth >= peer median
                credit_latest = self.get_metric_value(bank_df, credit_code)
                if credit_latest is None:
                    return {'pass': False, 'reason': f'Code {credit_code} not found'}
                
                if peer_data_cache and credit_code in peer_data_cache:
                    peer_credit = peer_data_cache[credit_code]
                else:
                    peer_credit = [self.get_metric_value(df, credit_code) for df in all_banks_data.values()]
                    peer_credit = [v for v in peer_credit if v is not None]
                
                credit_median = None
                credit_pass = False
                if len(peer_credit) > 0:
                    sorted_credit = sorted(peer_credit)
                    credit_median = sorted_credit[len(sorted_credit)//2] if len(sorted_credit) > 0 else None
                    credit_pass = credit_latest >= credit_median
                
                # Check 2: NPL not worsening (current NPL <= 4q average)
                npl_series = self.get_metric_series(bank_df, npl_code)
                npl_pass = True
                npl_current = None
                npl_avg = None
                
                if len(npl_series) >= window + 1:
                    npl_current = npl_series.iloc[-1]
                    npl_avg = npl_series.iloc[-window-1:-1].mean()
                    npl_pass = npl_current <= npl_avg
                elif len(npl_series) > 0:
                    npl_current = npl_series.iloc[-1]
                    npl_avg = npl_series.mean()
                    npl_pass = npl_current <= npl_avg
                
                pass_check = credit_pass and npl_pass
                return {
                    'pass': pass_check, 
                    'credit_pass': credit_pass, 
                    'credit_value': credit_latest,
                    'npl_pass': npl_pass, 
                    'npl_current': npl_current,
                    'npl_avg': npl_avg
                }
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_bank(self, symbol, bank_df, all_banks_data, peer_data_cache=None):
        """Calculate solvency score for a bank (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': min(points, 5),  # Cap at 5
            'details': results
        }

# %%
# Calculate SOLV scores for all banks
solv_scorer = SolvencyScorer()

solv_peer_data_cache = {}
solv_metric_codes = ['ryq58', 'ryq59', 'ryq56', 'ryq60']
for code in solv_metric_codes:
    solv_peer_data_cache[code] = [solv_scorer.get_metric_value(df, code) for df in bank_ratios_data.values()]
    solv_peer_data_cache[code] = [v for v in solv_peer_data_cache[code] if v is not None]

solv_scores = {}

for idx, (symbol, bank_df) in enumerate(bank_ratios_data.items()):
    score_result = solv_scorer.score_bank(symbol, bank_df, bank_ratios_data, solv_peer_data_cache)
    solv_scores[symbol] = score_result

# %%
class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE quality gate', 'code': 'ryq12', 'type': 'absolute', 'rule': 'ROE ≥ 10%', 'threshold': 0.10, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'P/B cheap vs peer (bottom 30%)', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.30', 'cut': 0.30,'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'P/E cheap vs peer (bottom 30%)', 'code': 'ryd21', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.30', 'cut': 0.30, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'P/B below own history (avg_5y excl latest)', 'code': 'ryd25', 'type': 'history_vs_avg_exlatest', 'rule': 'value ≤ avg_5y(excl latest)', 'window_years': 5, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'P/E below own history (avg_5y excl latest)', 'code': 'ryd21', 'type': 'history_vs_avg_exlatest', 'rule': 'value ≤ avg_5y(excl latest)', 'window_years': 5, 'direction': 'lower', 'must_have': False},
            {'id': 6, 'name': 'Justified PB proxy: (P/B)/ROE ≤ peer_median', 'pb_code': 'ryd25', 'roe_code': 'ryq12', 'type': 'ratio_threshold', 'rule': '(P/B)/ROE ≤ peer_median', 'fallback_abs': 20.0, 'direction': 'lower', 'must_have': False},
        ]

    def get_metric_series(self, bank_df, code):
        metric = bank_df[bank_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def get_metric_value(self, bank_df, code):
        s = self.get_metric_series(bank_df, code)
        return None if s.empty else s.iloc[-1]

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, bank_df, criterion, all_banks_data, peer_data_cache=None):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(bank_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(bank_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_banks_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(bank_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(bank_df, code)
                window_q = int(criterion.get('window_years', 5) * 4)

                if len(s) < 2:
                    return {'pass': False, 'reason': 'Not enough history', 'value': latest}

                hist = s.tail(window_q + 1)
                hist_ex = hist.iloc[:-1]
                if hist_ex.empty:
                    return {'pass': False, 'reason': 'No history excl latest', 'value': latest}

                avg = float(hist_ex.mean())
                pass_check = (latest <= avg) if criterion['direction'] == 'lower' else (latest >= avg)

                return {'pass': pass_check, 'value': latest, 'hist_avg_exlatest': avg, 'window_q': window_q}

            if crit_type == 'ratio_threshold':
                pb = self.get_metric_value(bank_df, criterion['pb_code'])
                roe = self.get_metric_value(bank_df, criterion['roe_code'])

                if pb is None or roe is None or roe == 0:
                    return {'pass': False, 'reason': 'Missing PB or ROE or ROE=0', 'pb': pb, 'roe': roe}

                if roe < 0.10:
                    return {'pass': False, 'reason': 'ROE below gate (<10%)', 'pb': pb, 'roe': roe, 'ratio': pb/roe}

                ratio = pb / roe

                peer_ratios = []
                for df in all_banks_data.values():
                    pb_i = self.get_metric_value(df, criterion['pb_code'])
                    roe_i = self.get_metric_value(df, criterion['roe_code'])
                    if pb_i is None or roe_i is None or roe_i == 0:
                        continue
                    if roe_i < 0.10:
                        continue
                    peer_ratios.append(pb_i / roe_i)

                if len(peer_ratios) > 0:
                    peer_median = float(pd.Series(peer_ratios).median())
                    pass_check = ratio <= peer_median
                    return {'pass': pass_check, 'ratio': ratio, 'peer_median': peer_median, 'pb': pb, 'roe': roe}
                else:
                    fallback = float(criterion.get('fallback_abs', 20.0))
                    pass_check = ratio <= fallback
                    return {'pass': pass_check, 'ratio': ratio, 'fallback_abs': fallback, 'pb': pb, 'roe': roe}

            return {'pass': False, 'reason': 'Unknown criterion type'}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score_bank(self, symbol, bank_df, all_banks_data, peer_data_cache=None):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(bank_df, crit, all_banks_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': round(min(points, 5.0), 2),  # Cap at 5
            'details': results
        }


# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, bank_df in bank_ratios_data.items():
        val_scores[symbol] = val_scorer.score_bank(symbol, bank_df, bank_ratios_data, None)    
    
    # Build combined scores (align symbols across all scorers)
    symbols = sorted(set(liq_scores.keys()) | set(prof_scores.keys()) | set(solv_scores.keys()) | set(val_scores.keys()))
    combined_scores_draft = pd.DataFrame({
        'Symbol': symbols,
        'LIQ_Score': [liq_scores.get(s, {}).get('score', 0) for s in symbols],
        'PROF_Score': [prof_scores.get(s, {}).get('score', 0) for s in symbols],
        'SOLV_Score': [solv_scores.get(s, {}).get('score', 0) for s in symbols],
        'VAL_Score': [val_scores.get(s, {}).get('score', 0) for s in symbols]
    })
    combined_scores_draft['Total_Score'] = combined_scores_draft['LIQ_Score'] + combined_scores_draft['PROF_Score'] + combined_scores_draft['SOLV_Score'] + combined_scores_draft['VAL_Score']
    combined_scores_draft = combined_scores_draft.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    combined_scores_draft['Rank'] = combined_scores_draft['Total_Score'].rank(method='min', ascending=False).astype(int)
    combined_scores_draft = combined_scores_draft[['Rank','Symbol','LIQ_Score','PROF_Score','SOLV_Score','VAL_Score','Total_Score']].sort_values(['Rank','Symbol']).reset_index(drop=True)

except Exception as e:
    print(f"[Fund_BANK] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()