# %%
import requests
import pandas as pd

all_stocks = []
goods_and_services_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
goods_and_services_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all goods and services symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    goods_and_services_symbols = [
        s.get('symbol')
        for s in all_stocks
        if isinstance(s, dict)
        and s.get('industry_lv1') == 'Industrials'
        and s.get('industry_lv2') == 'Industrial Goods & Services'
        and s.get('symbol')
    ]
    if not goods_and_services_symbols:
        goods_and_services_symbols = [s for s in all_stocks if isinstance(s, str)]
    
    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(goods_and_services_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }

    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    url_daily = "http://192.168.8.190:8000/MKD/stock-ratios-daily"
    daily_params = {
        "symbols": ",".join(goods_and_services_symbols),
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
        goods_and_services_symbols,
        key=lambda s: float(matching_volume_12_2025.get(s, float('-inf'))),
        reverse=True
    )
    top_75_stocks = filtered_stocks[:75]
    goods_and_services_symbols = top_75_stocks

    df_ratios = pd.DataFrame(all_ratios_data)

    if 'symbol' in df_ratios.columns:
        for symbol in goods_and_services_symbols:
            symbol_data = df_ratios[df_ratios['symbol'] == symbol]
            if not symbol_data.empty:
                goods_and_services_ratios_data[symbol] = symbol_data.copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_INDUSTRIALS_G&S] Warning: failed to fetch goods and services ratios: {e}")

# %%
print(goods_and_services_symbols)

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Quick Ratio', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value ≥ 1.0', 'threshold': 1.0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'CCC', 'type': 'calculated_peer_percentile', 'formula': 'ryq18+ryq16-ryq20', 'codes': ['ryq18', 'ryq16', 'ryq20'], 'rule': 'pct ≤ 0.40', 'cut': 0.40, 'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'DIO (Days Inventory)', 'code': 'ryq18', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.80', 'cut': 0.80, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'DPO (Days Payables)', 'code': 'ryq20', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Inventory Turnover', 'type': 'calculated_peer', 'formula': 'rev/bsa15', 'codes': ['rev', 'bsa15'], 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'Short-term Inv/Cash', 'type': 'calculated_ratio', 'formula': 'bsa5/bsa2', 'codes': ['bsa5', 'bsa2'], 'rule': 'value ≥ 0.4', 'threshold': 0.4, 'direction': 'higher', 'must_have': False},
        ]
        self.peer_cache = {}  # Cache for peer computations

    def get_metric_series(self, goods_and_services_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = goods_and_services_df[goods_and_services_df['code'] == code].copy()
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

    def get_metric_period_dict(self, goods_and_services_df, code):
        metric_data = goods_and_services_df[goods_and_services_df['code'] == code].copy()
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

    def calculate_ratio_series(self, goods_and_services_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(goods_and_services_df, numerator_code)
        denominator_map = self.get_metric_period_dict(goods_and_services_df, denominator_code)

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
    
    def calculate_ccc_series(self, goods_and_services_df, codes, periods=12):
        """Calculate CCC series (ryq18+ryq16-ryq20) for last N periods"""
        # codes = ['ryq18', 'ryq16', 'ryq20']
        dio_map = self.get_metric_period_dict(goods_and_services_df, codes[0])  # ryq18
        dso_map = self.get_metric_period_dict(goods_and_services_df, codes[1])  # ryq16
        dpo_map = self.get_metric_period_dict(goods_and_services_df, codes[2])  # ryq20

        common_periods = [p for p in dio_map.keys() if p in dso_map and p in dpo_map]
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
            dio = dio_map.get(p)
            dso = dso_map.get(p)
            dpo = dpo_map.get(p)
            if dio is None or dso is None or dpo is None:
                continue
            ccc = dio + dso - dpo
            series.append((p, ccc))

        return series
    
    def calculate_ccc_dict(self, goods_and_services_df, codes):
        """Calculate CCC dict for all periods"""
        dio_map = self.get_metric_period_dict(goods_and_services_df, codes[0])  # ryq18
        dso_map = self.get_metric_period_dict(goods_and_services_df, codes[1])  # ryq16
        dpo_map = self.get_metric_period_dict(goods_and_services_df, codes[2])  # ryq20

        result = {}
        common_periods = [p for p in dio_map.keys() if p in dso_map and p in dpo_map]
        
        for p in common_periods:
            dio = dio_map.get(p)
            dso = dso_map.get(p)
            dpo = dpo_map.get(p)
            if dio is None or dso is None or dpo is None:
                continue
            ccc = dio + dso - dpo
            result[p] = ccc

        return result
    
    def evaluate_criterion(self, goods_and_services_df, criterion, all_companies_data, peer_data_cache=None, peer_ccc_cache=None):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            import numpy as np
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(goods_and_services_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': f"Code {criterion['code']} not found", 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    pass_check = all(v <= criterion['threshold'] for v in values)
                else:
                    pass_check = all(v >= criterion['threshold'] for v in values)
                return {
                    'pass': pass_check,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'calculated_ratio':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(goods_and_services_df, numerator, denominator, periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    pass_check = all(v <= criterion['threshold'] for v in values)
                else:
                    pass_check = all(v >= criterion['threshold'] for v in values)
                return {
                    'pass': pass_check,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type in {'peer_percentile', 'calculated_peer', 'calculated_peer_percentile'}:
                if crit_type == 'calculated_peer':
                    numerator, denominator = criterion['codes']
                    series = self.calculate_ratio_series(goods_and_services_df, numerator, denominator, periods=12)
                elif crit_type == 'calculated_peer_percentile':
                    series = self.calculate_ccc_series(goods_and_services_df, criterion['codes'], periods=12)
                else:
                    series = self.get_metric_series(goods_and_services_df, criterion['code'], periods=12)

                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}

                # Use pre-computed peer_data_cache if available
                if crit_type == 'calculated_peer':
                    if peer_data_cache and criterion['codes'][0] in peer_data_cache:
                        peer_maps = peer_data_cache[criterion['codes'][0]]
                    else:
                        peer_maps = [self.calculate_ratio_series(df, criterion['codes'][0], criterion['codes'][1], periods=None) for df in all_companies_data.values()]
                elif crit_type == 'calculated_peer_percentile':
                    if peer_ccc_cache:
                        peer_maps = peer_ccc_cache
                    else:
                        peer_maps = [self.calculate_ccc_dict(df, criterion['codes']) for df in all_companies_data.values()]
                else:
                    code = criterion['code']
                    if peer_data_cache and code in peer_data_cache:
                        peer_maps = peer_data_cache[code]
                    else:
                        peer_maps = [self.get_metric_period_dict(df, code) for df in all_companies_data.values()]

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
    
    def score_goods_and_services(self, symbol, goods_and_services_df, all_companies_data, peer_data_cache=None, peer_ccc_cache=None):
        """Calculate liquidity score for a goods and services (0-5 points)"""
        is_pure_service = False
        
        try:
            inv_series = self.get_metric_series(goods_and_services_df, 'bsa15', periods=4) 
            asset_series = self.get_metric_series(goods_and_services_df, 'bsa53', periods=4)
            
            if asset_series:
                inv_vals = [v for _, v in inv_series]
                asset_vals = [v for _, v in asset_series]
                
                if not inv_vals or sum(inv_vals) == 0:
                    is_pure_service = True
                else:
                    avg_inv = sum(inv_vals) / len(inv_vals)
                    avg_asset = sum(asset_vals) / len(asset_vals)
                    if avg_asset > 0 and (avg_inv / avg_asset) < 0.07:
                        is_pure_service = True
        except Exception:
            is_pure_service = False

        active_criteria = []
        for crit in self.criteria:
            if is_pure_service and crit['id'] in [2, 3, 5]:
                continue
            if is_pure_service and crit['id'] == 1:
                crit_copy = crit.copy()
                crit_copy['threshold'] = 0.75
                crit_copy['rule'] = 'value ≥ 0.75 (Service Adjusted)'
                active_criteria.append(crit_copy)
            else:
                active_criteria.append(crit)
        
        results = {}
        points = 0

        must_have_list = [c for c in active_criteria if c['must_have']]
        must_have_results = []
        
        for crit in must_have_list:
            result = self.evaluate_criterion(goods_and_services_df, crit, all_companies_data, peer_data_cache, peer_ccc_cache)
            must_have_results.append(result['pass'])
            results[f"C{crit['id']}"] = result
        
        if not must_have_list or all(must_have_results):
            points += 1.5
        
        optional_criteria = [c for c in active_criteria if not c['must_have']]

        if optional_criteria:
            point_per_crit = 3.5 / len(optional_criteria)
            
            for crit in optional_criteria:
                result = self.evaluate_criterion(goods_and_services_df, crit, all_companies_data, peer_data_cache, peer_ccc_cache)
                results[f"C{crit['id']}"] = result
                if result['pass']:
                    points += point_per_crit
        
        if is_pure_service:
            results['note'] = 'Pure Service Detected: Skipped Inventory Criteria (C2, C3, C5)'

        return {
            'symbol': symbol,
            'score': float(min(points, 5)), 
            'details': results
        }

#%%
scorer = LiquidityScorer()

# Pre-compute all peer data once (OPTIMIZATION)
peer_data_cache = {}
for code in ['ryq2', 'ryq18', 'ryq16', 'ryq20', 'rev', 'bsa15', 'bsa5', 'bsa2']:
    peer_data_cache[code] = [scorer.get_metric_period_dict(df, code) for df in goods_and_services_ratios_data.values()]

# Pre-compute CCC for all peers
ccc_codes = ['ryq18', 'ryq16', 'ryq20']
peer_ccc_cache = [scorer.calculate_ccc_dict(df, ccc_codes) for df in goods_and_services_ratios_data.values()]

liq_scores = {}
for symbol in goods_and_services_symbols:
    if symbol in goods_and_services_ratios_data:
        goods_and_services_df = goods_and_services_ratios_data[symbol]
        score_result = scorer.score_goods_and_services(symbol, goods_and_services_df, goods_and_services_ratios_data, peer_data_cache, peer_ccc_cache)
        liq_scores[symbol] = score_result

# %%
'''for symbol in sorted(liq_scores.keys()):
    print(f"{symbol}: {liq_scores[symbol].get('score')}")'''

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Gross Profit Margin', 'code': 'ryq25', 'type': 'absolute', 'rule': 'value ≥ 0.1', 'threshold': 0.1, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'EBIT Margin', 'code': 'ryq27', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.30', 'cut': 0.30, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Asset Turnover', 'code': 'ryq31', 'type': 'absolute', 'rule': 'value ≥ 0.7', 'threshold': 0.7, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Fixed Asset Turnover', 'code': 'ryq91', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Operating Leverage', 'type': 'calculated_ratio', 'formula': 'ryq27_growth/ryq34_growth', 'codes': ['ryq27', 'ryq34'], 'rule': 'value ≥ 1.2', 'threshold': 1.2, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'ROA', 'code': 'ryq14', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.60', 'cut': 0.60, 'direction': 'higher', 'must_have': False},
        ]

    def get_metric_series(self, goods_and_services_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = goods_and_services_df[goods_and_services_df['code'] == code].copy()
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

    def get_metric_period_dict(self, goods_and_services_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = goods_and_services_df[goods_and_services_df['code'] == code].copy()
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

    def calculate_growth_rate_series(self, goods_and_services_df, code, periods=12):
        """Calculate YoY growth rate for a metric over last N periods"""
        metric_data = goods_and_services_df[goods_and_services_df['code'] == code].copy()
        if metric_data.empty:
            return []

        if {'year', 'quarter'}.issubset(metric_data.columns):
            metric_data = metric_data.sort_values(['year', 'quarter'])
            metric_data['period_key'] = list(zip(metric_data['year'], metric_data['quarter']))
        else:
            metric_data = metric_data.reset_index().rename(columns={'index': 'period_key'})

        metric_data = metric_data.dropna(subset=['value'])
        
        if len(metric_data) < 5:  # Need at least 5 quarters for YoY growth
            return []
        
        values = metric_data['value'].values
        period_keys = metric_data['period_key'].values
        
        growth_series = []
        for i in range(4, len(values)):
            current = values[i]
            prior_year = values[i - 4]  # 4 quarters back = 1 year
            if prior_year != 0:
                growth = (current - prior_year) / abs(prior_year)
                growth_series.append((period_keys[i], growth))
        
        if periods is not None and len(growth_series) > periods:
            growth_series = growth_series[-periods:]
        
        return growth_series

    def calculate_operating_leverage_series(self, goods_and_services_df, codes, periods=12):
        """Calculate Operating Leverage (EBIT margin growth / Revenue growth) for last N periods"""
        # codes = ['ryq27', 'ryq34']
        ebit_growth = self.calculate_growth_rate_series(goods_and_services_df, codes[0], periods=None)
        revenue_growth = self.calculate_growth_rate_series(goods_and_services_df, codes[1], periods=None)
        
        if not ebit_growth or not revenue_growth:
            return []
        
        ebit_dict = {p: v for p, v in ebit_growth}
        revenue_dict = {p: v for p, v in revenue_growth}
        
        common_periods = [p for p in ebit_dict.keys() if p in revenue_dict]
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
            ebit_g = ebit_dict.get(p)
            revenue_g = revenue_dict.get(p)
            if ebit_g is None or revenue_g is None or revenue_g == 0:
                continue
            leverage = ebit_g / revenue_g
            series.append((p, leverage))
        
        return series

    def evaluate_criterion(self, goods_and_services_df, criterion, all_companies_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            import numpy as np
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(goods_and_services_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': f"Code {criterion['code']} not found", 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    pass_check = all(v <= criterion['threshold'] for v in values)
                else:
                    pass_check = all(v >= criterion['threshold'] for v in values)
                return {
                    'pass': pass_check,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'calculated_ratio':
                # Operating Leverage: growth ratio
                series = self.calculate_operating_leverage_series(goods_and_services_df, criterion['codes'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'Insufficient data for growth calculation', 'values': []}

                values = [v for _, v in series]
                if criterion['direction'] == 'lower':
                    pass_check = all(v <= criterion['threshold'] for v in values)
                else:
                    pass_check = all(v >= criterion['threshold'] for v in values)
                return {
                    'pass': pass_check,
                    'values': values,
                    'threshold': criterion['threshold'],
                    'periods_used': len(values)
                }

            if crit_type == 'peer_percentile':
                series = self.get_metric_series(goods_and_services_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}

                # Use pre-computed peer_data_cache if available
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_companies_data.values()]

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

    def score_goods_and_services(self, symbol, goods_and_services_df, all_companies_data, peer_data_cache=None):
        """Calculate profitability score for a goods and services (0-5 points)"""
        is_low_margin_biz = False
        avg_gpm = 0.0
        try:
            gpm_series = self.get_metric_series(goods_and_services_df, 'ryq25', periods=4)
            if gpm_series:
                vals = [v for _, v in gpm_series]
                avg_gpm = sum(vals) / len(vals)
                if avg_gpm < 0.15: 
                    is_low_margin_biz = True
        except Exception:
            pass
            
        active_criteria = []
        for crit in self.criteria:
            crit_copy = crit.copy()
            
            if crit['id'] == 1:
                if is_low_margin_biz:
                    crit_copy['threshold'] = 0.05
                    crit_copy['rule'] = 'value ≥ 0.05 (Low Margin Adj)'
                else:
                    crit_copy['threshold'] = 0.15
            
            active_criteria.append(crit_copy)

        results = {}
        points = 0
        
        # Must-have criteria (1.5 points if all pass)
        must_have_list = [c for c in active_criteria if c['must_have']]
        must_have_results = []
        
        for crit in must_have_list:
            result = self.evaluate_criterion(goods_and_services_df, crit, all_companies_data, peer_data_cache)
            must_have_results.append(result['pass'])
            results[f"C{crit['id']}"] = result
        
        if not must_have_list or all(must_have_results):
            points += 1.5 
        
        # Optional criteria (distribute remaining 3.5 points)
        optional_criteria = [c for c in active_criteria if not c['must_have']]
        if optional_criteria:
            point_per_crit = 3.5 / len(optional_criteria)
            for crit in optional_criteria:
                result = self.evaluate_criterion(goods_and_services_df, crit, all_companies_data, peer_data_cache)
                results[f"C{crit['id']}"] = result
                if result['pass']:
                    points += point_per_crit
        
        if is_low_margin_biz:
            results['note'] = f'Low Margin Biz Detected (Avg GPM={avg_gpm:.1%}): Threshold lowered to 8%'

        return {
            'symbol': symbol,
            'score': float(min(points, 5)), 
            'details': results
        }

#%%
prof_scorer = ProfitabilityScorer()

# Pre-compute all peer data once (OPTIMIZATION)
prof_peer_data_cache = {}
for code in ['ryq25', 'ryq27', 'ryq31', 'ryq91', 'ryq14']:
    prof_peer_data_cache[code] = [prof_scorer.get_metric_period_dict(df, code) for df in goods_and_services_ratios_data.values()]

prof_scores = {}
for symbol in goods_and_services_symbols:
    if symbol in goods_and_services_ratios_data:
        goods_and_services_df = goods_and_services_ratios_data[symbol]
        score_result = prof_scorer.score_goods_and_services(symbol, goods_and_services_df, goods_and_services_ratios_data, prof_peer_data_cache)
        prof_scores[symbol] = score_result

# %%
'''for symbol in sorted(prof_scores.keys()):
    print(f"{symbol}: {prof_scores[symbol].get('score')}")'''

# %%
import numpy as np

class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 2.5', 'threshold': 2.5, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Debt/EBITDA Proxy', 'type': 'calculated_custom_debt', 'codes': ['ryq10', 'rqq72', 'bsa53'], 'rule': 'pct ≤ 3.0', 'cut': 3.0, 'direction': 'lower', 'must_have': True},
            {'id': 3, 'name': 'Fixed Assets/Total Assets', 'type': 'calculated_ratio', 'formula': 'bsa29/bsa53', 'codes': ['bsa29', 'bsa53'], 'rule': 'pct ≤ 0.70', 'cut': 0.70, 'direction': 'lower_peer', 'must_have': False},
            {'id': 4, 'name': 'Capex Efficiency', 'type': 'growth_ratio', 'codes': ['ryq34', 'bsa29'], 'rule': 'value ≥ 1.0', 'threshold': 1.0, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Working Capital/Rev', 'type': 'calculated_wc', 'codes': ['bsa1', 'bsa55', 'rev'], 'rule': 'pct ≤ 0.70', 'cut': 0.70, 'direction': 'lower_peer', 'must_have': False},
        ]

    def get_metric_series(self, df, code, periods=12):
        data = df[df['code'] == code].copy()
        if data.empty: return []
        if 'date' in data.columns:
            data = data.sort_values('date')
            data['period_key'] = data['date']
        else:
            data = data.sort_values(['year', 'quarter'])
            data['period_key'] = list(zip(data['year'], data['quarter']))
        data = data.dropna(subset=['value'])
        if periods: data = data.tail(periods)
        return list(data[['period_key', 'value']].itertuples(index=False, name=None))

    def get_metric_period_dict(self, df, code):
        series = self.get_metric_series(df, code, periods=None)
        return {p: v for p, v in series}

    def calculate_growth_rate_series(self, df, code, periods=12):
        series = self.get_metric_series(df, code, periods=None)
        if len(series) < 5: return []
        
        values = [v for _, v in series]
        keys = [k for k, _ in series]
        growth_data = []
        
        for i in range(4, len(values)):
            curr, prev = values[i], values[i-4]
            if prev != 0:
                growth = (curr - prev) / abs(prev)
                growth_data.append((keys[i], growth))
            else:
                growth_data.append((keys[i], 0))
                
        if periods and len(growth_data) > periods:
            growth_data = growth_data[-periods:]
        return growth_data

    
    def calculate_custom_debt_proxy(self, df, codes, periods=12):
        debt = self.get_metric_period_dict(df, codes[0])
        profit = self.get_metric_period_dict(df, codes[1])
        assets = self.get_metric_period_dict(df, codes[2])
        
        common = [p for p in debt if p in profit and p in assets]
        common.sort()
        if periods: common = common[-periods:]
        
        series = []
        for p in common:
            d, pr, a = debt[p], profit[p], assets[p]
            if a != 0 and pr != 0:
                val = d / (pr / a)
                series.append((p, val))
        return series

    def calculate_wc_revenue(self, df, codes, periods=12):
        ca = self.get_metric_period_dict(df, codes[0])
        cl = self.get_metric_period_dict(df, codes[1])
        rev = self.get_metric_period_dict(df, codes[2])
        
        common = [p for p in ca if p in cl and p in rev]
        common.sort()
        if periods: common = common[-periods:]
        
        series = []
        for p in common:
            r = rev[p]
            if r != 0:
                wc = (ca[p] - cl[p]) / r
                series.append((p, wc))
        return series

    def evaluate_criterion(self, df, criterion, all_companies_data, peer_cache=None):
        try:
            import numpy as np
            crit_type = criterion['type']
            series = []

            # 1. Get Series Data based on Type
            if crit_type == 'absolute':
                series = self.get_metric_series(df, criterion['code'], periods=12)
            elif crit_type == 'calculated_custom_debt':
                series = self.calculate_custom_debt_proxy(df, criterion['codes'], periods=12)
            elif crit_type == 'calculated_ratio':
                num = self.get_metric_period_dict(df, criterion['codes'][0])
                den = self.get_metric_period_dict(df, criterion['codes'][1])
                keys = sorted([k for k in num if k in den])[-12:]
                series = [(k, num[k]/den[k]) for k in keys if den[k] != 0]
            elif crit_type == 'calculated_wc':
                series = self.calculate_wc_revenue(df, criterion['codes'], periods=12)
            elif crit_type == 'growth_ratio':
                # Capex Efficiency logic
                rev_g = dict(self.calculate_growth_rate_series(df, criterion['codes'][0], periods=12))
                asset_g = dict(self.calculate_growth_rate_series(df, criterion['codes'][1], periods=12))
                keys = sorted([k for k in rev_g if k in asset_g])
                series = []
                for k in keys:
                    rg, ag = rev_g[k], asset_g[k]
                    if ag <= 0 and rg > 0:
                        series.append((k, 999.0)) 
                    elif ag == 0:
                        series.append((k, 0.0))
                    else:
                        series.append((k, rg/ag))
            
            if not series: return {'pass': False, 'values': []}
            
            values = [v for _, v in series]

            # 2. Check Rule (Absolute vs Peer)
            if 'peer' in criterion.get('direction', '') or 'cut' in criterion:
                # Peer Comparison Logic - OPTIMIZED
                current_period = series[-1][0]
                current_val = values[-1]
                
                # Build peer values from cache instead of recalculating
                peer_vals = []
                
                if crit_type == 'calculated_custom_debt':
                    # Pre-compute peer debt values
                    cache_key = ('calculated_custom_debt', tuple(criterion['codes']))
                    if peer_cache is None:
                        peer_cache = {}
                    
                    if cache_key not in peer_cache:
                        peer_cache[cache_key] = {}
                        for comp_code, comp_df in all_companies_data.items():
                            s = self.calculate_custom_debt_proxy(comp_df, criterion['codes'], periods=None)
                            if s:
                                for period, val in s:
                                    if period not in peer_cache[cache_key]:
                                        peer_cache[cache_key][period] = []
                                    peer_cache[cache_key][period].append(val)
                    
                    if current_period in peer_cache[cache_key]:
                        peer_vals = peer_cache[cache_key][current_period]
                
                elif crit_type == 'calculated_ratio':
                    # Use metric cache for ratio
                    code1, code2 = criterion['codes']
                    if peer_cache and code1 in peer_cache and code2 in peer_cache:
                        peer_maps1 = peer_cache[code1]
                        peer_maps2 = peer_cache[code2]
                        for i in range(len(peer_maps1)):
                            n = peer_maps1[i].get(current_period)
                            d = peer_maps2[i].get(current_period)
                            if n is not None and d and d != 0:
                                peer_vals.append(n / d)
                
                elif crit_type == 'calculated_wc':
                    # Pre-compute peer WC values
                    cache_key = ('calculated_wc', tuple(criterion['codes']))
                    if peer_cache is None:
                        peer_cache = {}
                    
                    if cache_key not in peer_cache:
                        peer_cache[cache_key] = {}
                        for comp_code, comp_df in all_companies_data.items():
                            s = self.calculate_wc_revenue(comp_df, criterion['codes'], periods=None)
                            if s:
                                for period, val in s:
                                    if period not in peer_cache[cache_key]:
                                        peer_cache[cache_key][period] = []
                                    peer_cache[cache_key][period].append(val)
                    
                    if current_period in peer_cache[cache_key]:
                        peer_vals = peer_cache[cache_key][current_period]
                
                peer_vals = np.array(peer_vals)
                if len(peer_vals) == 0: return {'pass': False, 'values': values}

                if 'lower' in criterion['direction']:
                    pct = (current_val <= peer_vals).sum() / len(peer_vals)
                    is_pass = pct <= criterion['cut']
                else:
                    pct = (current_val >= peer_vals).sum() / len(peer_vals)
                    is_pass = pct >= criterion['cut']
                
                return {'pass': is_pass, 'values': values, 'percentile': pct}

            else:
                threshold = criterion['threshold']
                pass_check = all(v >= threshold for v in values)
                return {'pass': pass_check, 'values': values}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score_goods_and_services(self, symbol, goods_and_services_df, all_companies_data, peer_cache=None):
        """Chấm điểm Solvency với logic công bằng cho Asset Heavy/Light"""
        if peer_cache is None:
            peer_cache = {}
        
        is_asset_heavy = False
        avg_fa_ratio = 0.0
        try:
            fa = self.get_metric_period_dict(goods_and_services_df, 'bsa29')
            ta = self.get_metric_period_dict(goods_and_services_df, 'bsa53')
            common = [k for k in fa if k in ta]
            if common:
                ratios = [fa[k]/ta[k] for k in common[-4:] if ta[k] != 0] # 4 quý gần nhất
                if ratios:
                    avg_fa_ratio = sum(ratios) / len(ratios)
                    if avg_fa_ratio > 0.30:
                        is_asset_heavy = True
        except: pass

        results = {}
        points = 0
                
        adjusted_must_have = []
        optional_passed = 0
        optional_count = 0

        for crit in self.criteria:
            if crit['id'] == 3:
                if is_asset_heavy:
                    results[f"C{crit['id']}"] = {'pass': True, 'reason': f"Skipped: Asset Heavy (FA/TA={avg_fa_ratio:.1%})"}
                    continue
            
            result = self.evaluate_criterion(goods_and_services_df, crit, all_companies_data, peer_cache)
            results[f"C{crit['id']}"] = result
            
            if crit['id'] == 1 and not result['pass']:
                vals = result.get('values', [])
                if vals and min(vals) > 100:
                    result['pass'] = True
                    results[f"C{crit['id']}"]['note'] = "Passed by Zero-Debt Logic"

            if crit['must_have']:
                adjusted_must_have.append(result['pass'])
            else:
                optional_count += 1
                if result['pass']:
                    optional_passed += 1

        # Check Must Have
        if not adjusted_must_have or all(adjusted_must_have):
            points += 2.0
        
        # Check Optional
        if optional_count > 0:
            points += (optional_passed / optional_count) * 3.0
            
        if is_asset_heavy:
            results['note'] = f"Asset Heavy Mode (FA > 30%): Fixed Asset criteria skipped."

        return {
            'symbol': symbol,
            'score': float(min(points, 5)),
            'details': results
        }

# %%
solv_scorer = SolvencyScorer()

# Pre-compute all peer data once (OPTIMIZATION)
solv_peer_data_cache = {}

# Pre-compute basic metrics
for code in ['ryq77', 'ryq10', 'rqq72', 'bsa53', 'bsa29', 'bsa1', 'bsa55', 'rev']:
    solv_peer_data_cache[code] = [solv_scorer.get_metric_period_dict(df, code) for df in goods_and_services_ratios_data.values()]

# Pre-compute calculated_custom_debt for all periods (for faster peer comparison)
debt_codes = ['ryq10', 'rqq72', 'bsa53']
solv_peer_data_cache[('calculated_custom_debt', tuple(debt_codes))] = {}
for period_key in set().union(*[set(d.keys()) for d in solv_peer_data_cache['ryq10']]):
    solv_peer_data_cache[('calculated_custom_debt', tuple(debt_codes))][period_key] = []
    for comp_df in goods_and_services_ratios_data.values():
        s = solv_scorer.calculate_custom_debt_proxy(comp_df, debt_codes, periods=None)
        if s:
            for p, val in s:
                if p == period_key:
                    solv_peer_data_cache[('calculated_custom_debt', tuple(debt_codes))][period_key].append(val)

# Pre-compute calculated_wc for all periods
wc_codes = ['bsa1', 'bsa55', 'rev']
solv_peer_data_cache[('calculated_wc', tuple(wc_codes))] = {}
for period_key in set().union(*[set(d.keys()) for d in solv_peer_data_cache['bsa1']]):
    solv_peer_data_cache[('calculated_wc', tuple(wc_codes))][period_key] = []
    for comp_df in goods_and_services_ratios_data.values():
        s = solv_scorer.calculate_wc_revenue(comp_df, wc_codes, periods=None)
        if s:
            for p, val in s:
                if p == period_key:
                    solv_peer_data_cache[('calculated_wc', tuple(wc_codes))][period_key].append(val)

solv_scores = {}
for symbol in goods_and_services_symbols:
    if symbol in goods_and_services_ratios_data:
        goods_and_services_df = goods_and_services_ratios_data[symbol]
        score_result = solv_scorer.score_goods_and_services(symbol, goods_and_services_df, goods_and_services_ratios_data, solv_peer_data_cache)
        solv_scores[symbol] = score_result

# %%
'''for symbol in sorted(solv_scores.keys()):
    print(f"{symbol}: {solv_scores[symbol].get('score')}")'''

# %%
class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE quality gate', 'code': 'ryq12', 'type': 'absolute', 'rule': 'ROE ≥ 8%', 'threshold': 0.08, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'P/B < Median', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.50', 'cut': 0.50,'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'P/B Deep Value', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.20', 'cut': 0.20,'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'P/E < Median', 'code': 'ryd21', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'P/E Deep Value', 'code': 'ryd21', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.20', 'cut': 0.20, 'direction': 'lower', 'must_have': False},
            {'id': 6, 'name': 'P/E vs History', 'code': 'ryd21', 'type': 'history_vs_avg_exlatest', 'rule': 'value ≤ avg', 'window_years': 5, 'direction': 'lower', 'must_have': False},
            {'id': 7, 'name': 'Justified PB', 'pb_code': 'ryd25', 'roe_code': 'ryq12', 'type': 'ratio_threshold', 'rule': 'Better than peer', 'fallback_abs': 15.0, 'direction': 'lower', 'must_have': False},
        ]

    def get_metric_series(self, goods_and_services_df, code):
        metric = goods_and_services_df[goods_and_services_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def get_metric_value(self, goods_and_services_df, code):
        s = self.get_metric_series(goods_and_services_df, code)
        return None if s.empty else s.iloc[-1]

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, goods_and_services_df, criterion, all_goods_and_services_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(goods_and_services_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(goods_and_services_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_goods_and_services_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(goods_and_services_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(goods_and_services_df, code)
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
                pb = self.get_metric_value(goods_and_services_df, criterion['pb_code'])
                roe = self.get_metric_value(goods_and_services_df, criterion['roe_code'])

                if pb is None or roe is None or roe == 0:
                    return {'pass': False, 'reason': 'Missing PB or ROE or ROE=0', 'pb': pb, 'roe': roe}

                if roe < 0.10:
                    return {'pass': False, 'reason': 'ROE below gate (<10%)', 'pb': pb, 'roe': roe, 'ratio': pb/roe}

                ratio = pb / roe

                peer_ratios = []
                for df in all_goods_and_services_data.values():
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

    def score_goods_and_services(self, symbol, goods_and_services_df, all_goods_and_services_data):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(goods_and_services_df, crit, all_goods_and_services_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(goods_and_services_df, crit, all_goods_and_services_data)
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

    for symbol, goods_and_services_df in goods_and_services_ratios_data.items():
        val_scores[symbol] = val_scorer.score_goods_and_services(symbol, goods_and_services_df, goods_and_services_ratios_data)    
    
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

    print(f"Goods and Services Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_GOODS_AND_SERVICES] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()