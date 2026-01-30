# %%
import requests
import pandas as pd

all_stocks = []
RealEstate_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
RealEstate_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all real estate symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    RealEstate_symbols = [
        s.get('symbol')
        for s in all_stocks
        if isinstance(s, dict)
        and s.get('industry_lv1') == 'Financials'
        and s.get('industry_lv2') == 'Real Estate'
        and s.get('symbol')
    ]
    if not RealEstate_symbols:
        RealEstate_symbols = [s for s in all_stocks if isinstance(s, str)]
    
    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(RealEstate_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }

    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    url_daily = "http://192.168.8.190:8000/MKD/stock-ratios-daily"
    daily_params = {
        "symbols": ",".join(RealEstate_symbols),
        "codes": "30022",
        "start_date": "2025-11-01",
        "end_date": "2025-12-31"
    }
    r = requests.get(url_daily, params=daily_params, headers={"accept": "application/json"})
    r.raise_for_status()
    daily_payload = r.json()
    if isinstance(daily_payload, dict):
        daily_data = daily_payload.get('data', [])
    else:
        daily_data = daily_payload

    matching_volume_sum = {}
    matching_volume_count = {}
    for record in daily_data:
        if not isinstance(record, dict):
            continue
        if str(record.get('code')) != '30022':
            continue
        symbol = record.get('symbol')
        value = record.get('value')
        if symbol and value is not None:
            matching_volume_sum[symbol] = matching_volume_sum.get(symbol, 0.0) + float(value)
            matching_volume_count[symbol] = matching_volume_count.get(symbol, 0) + 1

    matching_volume_12_2025 = {
        symbol: matching_volume_sum[symbol] / matching_volume_count[symbol]
        for symbol in matching_volume_sum
        if matching_volume_count.get(symbol, 0) > 0
    }

    # Sort by average matching volume December 2025 and take top 100
    filtered_stocks = sorted(
        RealEstate_symbols,
        key=lambda s: float(matching_volume_12_2025.get(s, float('-inf'))),
        reverse=True
    )
    top_50_stocks = filtered_stocks[:50]
    RealEstate_symbols = top_50_stocks

    df_ratios = pd.DataFrame(all_ratios_data)

    if 'symbol' in df_ratios.columns:
        for symbol in RealEstate_symbols:
            symbol_data = df_ratios[df_ratios['symbol'] == symbol]
            if not symbol_data.empty:
                RealEstate_ratios_data[symbol] = symbol_data.copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_RealEstate] Warning: failed to fetch real estate ratios: {e}")

# %%
print(RealEstate_symbols)

# %%
import numpy as np

class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Cash Ratio', 'code': 'ryq1', 'type': 'absolute', 'rule': 'value ≥ 0.10', 'threshold': 0.10, 'direction': 'higher', 'weight': 1.0},
            {'id': 2, 'name': 'Prepayments/ST Debt', 'type': 'calculated_absolute', 'formula': 'bsa58/bsa56', 'codes': ['bsa58', 'bsa56'], 'rule': 'value ≥ 0.5', 'threshold': 0.5, 'direction': 'higher', 'weight': 1.5},
            {'id': 3, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 1.5', 'threshold': 1.5, 'direction': 'higher', 'weight': 1.0},
            {'id': 4, 'name': 'Current Ratio', 'code': 'ryq3', 'type': 'absolute', 'rule': 'value ≥ 1.5', 'threshold': 1.5, 'direction': 'higher', 'weight': 0.5},
            {'id': 5, 'name': 'Inventory Days Percentile', 'code': 'ryq18', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.60', 'cut': 0.60, 'direction': 'lower', 'weight': 1.0},
        ]
    
    def get_metric_series(self, realestate_df, code, periods=12):
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return []

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        if periods is not None:
            metric_data = metric_data.tail(periods)

        return list(metric_data[['period_key', 'value']].itertuples(index=False, name=None))

    def get_metric_period_dict(self, realestate_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return {}

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        return dict(metric_data[['period_key', 'value']].itertuples(index=False, name=None))

    def calculate_ratio_series(self, realestate_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(realestate_df, numerator_code)
        denominator_map = self.get_metric_period_dict(realestate_df, denominator_code)

        common_periods = [p for p in numerator_map.keys() if p in denominator_map]
        if not common_periods:
            return []

        if all(isinstance(p, tuple) for p in common_periods):
            common_periods = sorted(common_periods, key=lambda x: (x[0], x[1]))
        else:
            common_periods = sorted(common_periods)

        if periods is not None:
            common_periods = common_periods[-periods:]

        series = []
        for p in common_periods:
            denom = denominator_map.get(p)
            num = numerator_map.get(p)
            if denom is None or num is None or denom == 0:
                continue
            series.append((p, num / denom))

        return series
    
    def evaluate_criterion(self, realestate_df, criterion, all_realestate_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': f"Code {criterion['code']} not found", 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    period_passes = [v <= criterion['threshold'] for v in values]
                else:
                    period_passes = [v >= criterion['threshold'] for v in values]
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'calculated_absolute':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(realestate_df, numerator, denominator, periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    period_passes = [v <= criterion['threshold'] for v in values]
                else:
                    period_passes = [v >= criterion['threshold'] for v in values]
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'peer_percentile':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}
                period_values = {p: v for p, v in series}

                # Use pre-computed peer_data_cache if available
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_realestate_data.values()]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = np.array([pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None])
                    if len(peer_values) == 0:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= peer_values).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= peer_values).sum() / len(peer_values)
                        pass_check = pct >= criterion['cut']

                    period_passes.append(pass_check)
                    percentiles.append(pct)
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': list(period_values.values()),
                    'percentiles': percentiles,
                    'cut': criterion['cut'],
                    'periods_used': len(period_values)
                }

            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_realestate(self, symbol, realestate_df, all_realestate_data, peer_data_cache=None):
        points = 0.0
        details = {}
                
        for crit in self.criteria:
            result = self.evaluate_criterion(realestate_df, crit, all_realestate_data, peer_data_cache)
            details[f"C{crit['id']}"] = result
            
            if result['pass']:
                points += crit.get('weight', 0)
        
        return {
            'symbol': symbol,
            'score': round(points, 2),
            'details': details
        }

# %%
scorer = LiquidityScorer()

# Setup peer_data_cache for LiquidityScorer
peer_data_cache = {}
metric_codes = ['ryq1', 'ryq77', 'ryq3', 'ryq18', 'bsa58', 'bsa56']

for code in metric_codes:
    peer_data_cache[code] = [scorer.get_metric_period_dict(df, code) for df in RealEstate_ratios_data.values()]

liq_scores = {}

for idx, (symbol, realestate_df) in enumerate(RealEstate_ratios_data.items()):
    score_result = scorer.score_realestate(symbol, realestate_df, RealEstate_ratios_data, peer_data_cache)
    liq_scores[symbol] = score_result

# %%
'''for symbol in sorted(liq_scores.keys()):
    print(f"{symbol}: {liq_scores[symbol].get('score')}")'''

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Gross Margin Percentile', 'code': 'ryq25', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'weight': 1.0},
            {'id': 2, 'name': 'Net Margin Percentile', 'code': 'ryq29', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'weight': 1.0},
            {'id': 3, 'name': 'ROE', 'code': 'ryq12', 'type': 'absolute', 'rule': 'value ≥ 0.10', 'threshold': 0.10, 'direction': 'higher', 'weight': 1.0},
            {'id': 4, 'name': 'Profit Growth YoY', 'code': 'ryq39', 'type': 'absolute', 'rule': 'value > 0.0', 'threshold': 0.0, 'direction': 'higher', 'weight': 1.0},
            {'id': 5, 'name': 'EBIT Margin', 'code': 'ryq27', 'type': 'absolute', 'rule': 'value ≥ 0.15', 'threshold': 0.15, 'direction': 'higher', 'weight': 1.0},
        ]
    
    def get_metric_series(self, realestate_df, code, periods=12):
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return []

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        if periods is not None:
            metric_data = metric_data.tail(periods)

        return list(metric_data[['period_key', 'value']].itertuples(index=False, name=None))
    
    def get_metric_period_dict(self, realestate_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return {}

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        return dict(metric_data[['period_key', 'value']].itertuples(index=False, name=None))
    
    def evaluate_criterion(self, realestate_df, criterion, all_realestate_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': f"Code {criterion['code']} not found", 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    period_passes = [v <= criterion['threshold'] for v in values]
                else:
                    period_passes = [v >= criterion['threshold'] for v in values]
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'peer_percentile':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                
                # Use pre-computed peer_data_cache if available
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, criterion['code']) for df in all_realestate_data.values()]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = np.array([pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None])
                    if len(peer_values) == 0:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= peer_values).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= peer_values).sum() / len(peer_values)
                        pass_check = pct >= criterion['cut']

                    period_passes.append(pass_check)
                    percentiles.append(pct)
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': list(period_values.values()),
                    'percentiles': percentiles,
                    'cut': criterion['cut'],
                    'periods_used': len(period_values)
                }

            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_realestate(self, symbol, realestate_df, all_realestate_data, peer_data_cache=None):
        points = 0.0
        details = {}
        
        for crit in self.criteria:
            result = self.evaluate_criterion(realestate_df, crit, all_realestate_data, peer_data_cache)
            details[f"C{crit['id']}"] = result
            
            if result['pass']:
                points += crit.get('weight', 0.0)
        
        return {
            'symbol': symbol,
            'score': round(points, 2),
            'details': details
        }

# %%
prof_scorer = ProfitabilityScorer()

# Setup peer_data_cache for ProfitabilityScorer
prof_peer_data_cache = {}
prof_metric_codes = ['ryq25', 'ryq29', 'ryq12', 'ryq39', 'ryq27']

for code in prof_metric_codes:
    prof_peer_data_cache[code] = [prof_scorer.get_metric_period_dict(df, code) for df in RealEstate_ratios_data.values()]

prof_scores = {}

for idx, (symbol, realestate_df) in enumerate(RealEstate_ratios_data.items()):
    score_result = prof_scorer.score_realestate(symbol, realestate_df, RealEstate_ratios_data, prof_peer_data_cache)
    prof_scores[symbol] = score_result

# %%
'''for symbol in sorted(prof_scores.keys()):
    print(f"{symbol}: {prof_scores[symbol].get('score')}")'''

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 1.5', 'threshold': 1.5, 'direction': 'higher', 'weight': 1.5},
            {'id': 2, 'name': 'Fin. Debt/Equity', 'type': 'calculated_ratio', 'formula': '(bsa56 + bsa71) / bsa78', 'codes': ['bsa56', 'bsa71', 'bsa78'], 'rule': 'value ≤ 1.0', 'threshold': 1.0, 'direction': 'lower', 'weight': 1.0},
            {'id': 3, 'name': 'Adj. Liab/Assets', 'type': 'calculated_ratio', 'formula': '(bsa54 - bsa58) / bsa53', 'codes': ['bsa54', 'bsa58', 'bsa53'], 'rule': 'value ≤ 0.60', 'threshold': 0.60, 'direction': 'lower', 'weight': 1.0},
            {'id': 4, 'name': 'Inventory/Fin. Debt', 'type': 'calculated_ratio', 'formula': 'bsa15 / (bsa56 + bsa71)', 'codes': ['bsa15', 'bsa56', 'bsa71'], 'rule': 'value ≥ 1.5', 'threshold': 1.5, 'direction': 'higher', 'weight': 1.0},
            {'id': 5, 'name': 'Cash/Fin. Debt', 'type': 'calculated_ratio', 'formula': 'bsa2 / (bsa56 + bsa71)', 'codes': ['bsa2', 'bsa56', 'bsa71'], 'rule': 'value ≥ 0.05', 'threshold': 0.05, 'direction': 'higher', 'weight': 0.5},
        ]
    
    def get_metric_series(self, realestate_df, code, periods=12):
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return []

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        if periods is not None:
            metric_data = metric_data.tail(periods)

        return list(metric_data[['period_key', 'value']].itertuples(index=False, name=None))
    
    def get_metric_period_dict(self, realestate_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = realestate_df[realestate_df['code'] == code].copy()
        if metric_data.empty:
            return {}

        if 'date' in metric_data.columns:
            metric_data = metric_data.sort_values('date')
            metric_data['period_key'] = metric_data['date']
        elif {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        return dict(metric_data[['period_key', 'value']].itertuples(index=False, name=None))
    
    def calculate_formula_ratio_series(self, realestate_df, formula, codes, periods=12):
        """Calculate ratio from formula like '(bsa56 + bsa71) / bsa78'"""
        # Extract all series for the codes
        series_dict = {}
        for code in codes:
            series = self.get_metric_series(realestate_df, code, periods=periods)
            if not series:
                return []
            series_dict[code] = {p: v for p, v in series}
        
        # Find common periods
        common_periods = set(series_dict[codes[0]].keys())
        for code in codes[1:]:
            common_periods &= set(series_dict[code].keys())
        
        if not common_periods:
            return []
        
        # Evaluate formula for each period
        result = []
        for period in sorted(common_periods):
            try:
                # Create namespace with values for this period
                values = {code: series_dict[code][period] for code in codes}
                # Evaluate the formula
                ratio_value = eval(formula, {"__builtins__": {}}, values)
                if ratio_value is not None:
                    result.append((period, ratio_value))
            except:
                continue
        
        return result
    
    def evaluate_criterion(self, realestate_df, criterion, all_realestate_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']
            
            # Absolute criterion
            if crit_type == 'absolute':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': f"Code {criterion['code']} not found", 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    period_passes = [v <= criterion['threshold'] for v in values]
                else:
                    period_passes = [v >= criterion['threshold'] for v in values]
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }
            
            # Calculated ratio criterion
            elif crit_type == 'calculated_ratio':
                series = self.calculate_formula_ratio_series(realestate_df, criterion['formula'], criterion['codes'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    period_passes = [v <= criterion['threshold'] for v in values]
                else:
                    period_passes = [v >= criterion['threshold'] for v in values]
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                series = self.get_metric_series(realestate_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                
                # Use pre-computed peer_data_cache if available
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, criterion['code']) for df in all_realestate_data.values()]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = np.array([pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None])
                    if len(peer_values) == 0:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= peer_values).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= peer_values).sum() / len(peer_values)
                        pass_check = pct >= criterion['cut']

                    period_passes.append(pass_check)
                    percentiles.append(pct)
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': list(period_values.values()),
                    'percentiles': percentiles,
                    'cut': criterion['cut'],
                    'periods_used': len(period_values)
                }
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_realestate(self, symbol, realestate_df, all_realestate_data, peer_data_cache=None):
        points = 0.0
        details = {}
        
        for crit in self.criteria:
            result = self.evaluate_criterion(realestate_df, crit, all_realestate_data, peer_data_cache)
            details[f"C{crit['id']}"] = result
            
            if result['pass']:
                points += crit.get('weight', 0)
        
        return {
            'symbol': symbol,
            'score': round(points, 2),
            'details': details
        }

# %%
solv_scorer = SolvencyScorer()

# Setup peer_data_cache for SolvencyScorer
solv_peer_data_cache = {}
solv_metric_codes = ['ryq77', 'bsa56', 'bsa71', 'bsa78', 'bsa54', 'bsa58', 'bsa53', 'bsa15', 'bsa2']

for code in solv_metric_codes:
    solv_peer_data_cache[code] = [solv_scorer.get_metric_period_dict(df, code) for df in RealEstate_ratios_data.values()]

solv_scores = {}

for idx, (symbol, realestate_df) in enumerate(RealEstate_ratios_data.items()):
    score_result = solv_scorer.score_realestate(symbol, realestate_df, RealEstate_ratios_data, solv_peer_data_cache)
    solv_scores[symbol] = score_result

# %%
for symbol in sorted(solv_scores.keys()):
    print(f"{symbol}: {solv_scores[symbol].get('score')}")

# %%
class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE >= 10%', 'code': 'ryq12', 'type': 'absolute', 'rule': 'value >= 0.10', 'threshold': 0.10, 'direction': 'higher', 'weight': 0.5},
            {'id': 2, 'name': 'ROCE >= 8%', 'type': 'calculated_ratio', 'formula': 'rqq72 / (bsa78 + bsa56 + bsa71)', 'codes': ['rqq72', 'bsa78', 'bsa56', 'bsa71'],'rule': 'value >= 0.08', 'threshold': 0.08, 'direction': 'higher', 'weight': 0.5},            
            {'id': 3, 'name': 'P/B < Median', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct <= 0.50', 'cut': 0.50, 'direction': 'lower', 'weight': 1.5},
            {'id': 4, 'name': 'P/B Deep Value', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct <= 0.20', 'cut': 0.20, 'direction': 'lower', 'weight': 0.5},
            {'id': 5, 'name': 'P/E < Median', 'code': 'ryd21', 'type': 'peer_percentile', 'rule': 'pct <= 0.50', 'cut': 0.50, 'direction': 'lower', 'weight': 1.0},
            {'id': 6, 'name': 'P/E vs History', 'code': 'ryd21', 'type': 'history_vs_avg_exlatest', 'rule': 'value <= avg', 'window_years': 5, 'direction': 'lower', 'weight': 1.0},
        ]

    def get_metric_value(self, realestate_df, code):
        """Get latest (most recent) value for a metric code"""
        metric = realestate_df[realestate_df['code'] == code]
        if len(metric) == 0:
            return None
        s = metric['value'].dropna()
        return None if s.empty else s.iloc[-1]

    def get_metric_series(self, realestate_df, code):
        """Get all historical values for a metric code"""
        metric = realestate_df[realestate_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def calculate_formula_value(self, realestate_df, formula, codes):
        """Calculate single value from formula using latest values"""
        values = {}
        for code in codes:
            val = self.get_metric_value(realestate_df, code)
            if val is None:
                return None
            values[code] = val
        try:
            return eval(formula, {"__builtins__": {}}, values)
        except:
            return None

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, realestate_df, criterion, all_realestate_data):
        """Evaluate a single criterion using latest (most recent) value only"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(realestate_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'calculated_ratio':
                # Use latest values only for calculation
                latest = self.calculate_formula_value(realestate_df, criterion['formula'], criterion['codes'])
                if latest is None:
                    return {'pass': False, 'reason': 'Missing data for formula calculation', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(realestate_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_realestate_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(realestate_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(realestate_df, code)
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

            return {'pass': False, 'reason': 'Unknown criterion type'}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score_realestate(self, symbol, realestate_df, all_realestate_data):
        results = {}
        points = 0.0

        for crit in self.criteria:
            result = self.evaluate_criterion(realestate_df, crit, all_realestate_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += crit.get('weight', 0)
        
        return {
            'symbol': symbol,
            'score': round(points, 2),
            'details': results
        }

# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, realestate_df in RealEstate_ratios_data.items():
        val_scores[symbol] = val_scorer.score_realestate(symbol, realestate_df, RealEstate_ratios_data)    
    
    # Build combined scores
    combined_scores_draft = pd.DataFrame({
        'Symbol': sorted(liq_scores.keys()),
        'LIQ_Score': [liq_scores[s]['score'] for s in sorted(liq_scores.keys())],
        'PROF_Score': [prof_scores[s]['score'] for s in sorted(prof_scores.keys())],
        'SOLV_Score': [solv_scores[s]['score'] for s in sorted(solv_scores.keys())],
        'VAL_Score': [val_scores[s]['score'] for s in sorted(val_scores.keys())]
    })
    combined_scores_draft['Total_Score'] = combined_scores_draft['LIQ_Score'] + combined_scores_draft['PROF_Score'] + combined_scores_draft['SOLV_Score'] + combined_scores_draft['VAL_Score']
    combined_scores_draft = combined_scores_draft.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    combined_scores_draft['Rank'] = combined_scores_draft['Total_Score'].rank(method='min', ascending=False).astype(int)
    combined_scores_draft = combined_scores_draft[['Rank','Symbol','LIQ_Score','PROF_Score','SOLV_Score','VAL_Score','Total_Score']].sort_values(['Rank','Symbol']).reset_index(drop=True)

    print(f"Real Estate Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_REAL_ESTATE] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()