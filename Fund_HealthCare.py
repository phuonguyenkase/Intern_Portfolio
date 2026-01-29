# %%
import requests
import pandas as pd

all_stocks = []
healthcare_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
healthcare_ratios_data = {}
combined_scores_draft = pd.DataFrame()

# %%
try:
    # Get all symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    filtered_stocks = [
        s for s in all_stocks
        if s.get('industry_lv1') == 'Health Care'
    ]
    healthcare_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(healthcare_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }
    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    # Extract ryd11 values for 2025 Q3 directly from ratios data
    ryd11_2025_q3 = {}
    for record in all_ratios_data:
        if (record.get('code') == 'ryd11' and 
            record.get('year') == 2025 and 
            record.get('quarter') == 3):
            symbol = record.get('symbol')
            value = record.get('value')
            if symbol and value is not None:
                ryd11_2025_q3[symbol] = float(value)

    filtered_stocks = sorted(
        filtered_stocks,
        key=lambda s: float(ryd11_2025_q3.get(s.get('symbol'), float('-inf'))),
        reverse=True
    )
    top_50_stocks = filtered_stocks[:50]
    healthcare_symbols = [s['symbol'] for s in top_50_stocks]

    # Convert to DataFrame
    df_ratios = pd.DataFrame(all_ratios_data)

    if 'symbol' in df_ratios.columns:
        for symbol in healthcare_symbols:
            symbol_data = df_ratios[df_ratios['symbol'] == symbol]
            if not symbol_data.empty:
                healthcare_ratios_data[symbol] = symbol_data.copy()
    
    missing_symbols = [s for s in healthcare_symbols if s not in healthcare_ratios_data]
    if missing_symbols:
        print(f"WARNING: {len(missing_symbols)} symbols have NO ratio data and will be excluded:")
        print(missing_symbols)
    print(f"Symbols with data: {len(healthcare_ratios_data)} / {len(healthcare_symbols)}")

except Exception as e:
    print(f"[Fund_Healthcare] Warning: failed to fetch healthcare ratios: {e}")

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Quick Ratio', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value ≥ 1.2', 'threshold': 1.2, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Cash Ratio', 'code': 'ryq1', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 3, 'name': 'Days Sales Outstanding (DSO)', 'code': 'ryq16', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.30', 'cut': 0.30, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'Days Inventory Outstanding (DIO)', 'code': 'ryq18', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'Inventory/Current Assets', 'type': 'calculated_peer', 'formula': 'bsa15 / bsa1', 'codes': ['bsa15', 'bsa1'], 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
        ]

    def get_metric_series(self, healthcare_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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

    def get_metric_period_dict(self, healthcare_df, code):
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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

    def calculate_ratio_series(self, healthcare_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(healthcare_df, numerator_code)
        denominator_map = self.get_metric_period_dict(healthcare_df, denominator_code)

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
    
    def evaluate_criterion(self, healthcare_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(healthcare_df, criterion['code'], periods=12)
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

            if crit_type in {'peer_percentile', 'calculated_peer'}:
                if crit_type == 'calculated_peer':
                    numerator, denominator = criterion['codes']
                    series = self.calculate_ratio_series(healthcare_df, numerator, denominator, periods=12)
                else:
                    series = self.get_metric_series(healthcare_df, criterion['code'], periods=12)

                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}

                if crit_type == 'calculated_peer':
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_ratio_series(df, numerator, denominator, periods=None)
                        }
                        for df in all_companies_data.values()
                    ]
                else:
                    peer_maps = [self.get_metric_period_dict(df, criterion['code']) for df in all_companies_data.values()]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = [pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None]
                    if not peer_values:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= pd.Series(peer_values)).sum() / len(peer_values)
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
    
    def score_healthcare(self, symbol, healthcare_df, all_companies_data):
        """Calculate liquidity score for a healthcare (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (1.5 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 1.5  # 1.5 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3.5 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3.5 / len(optional_criteria)  # Distribute 3.5 points among optional
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),  # Cap at 5, force float
            'details': results
        }

# %%
scorer = LiquidityScorer()
liq_scores = {}

for idx, (symbol, healthcare_df) in enumerate(healthcare_ratios_data.items()):
    score_result = scorer.score_healthcare(symbol, healthcare_df, healthcare_ratios_data)
    liq_scores[symbol] = score_result

#%%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Gross Profit Margin', 'code': 'ryq25', 'type': 'absolute', 'rule': 'value ≥ 0.2', 'threshold': 0.2, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'ROIC', 'code': 'ryq76', 'type': 'absolute', 'rule': 'value ≥ 0.1', 'threshold': 0.1, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'ROA', 'code': 'ryq14', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'EBIT Margin Stability', 'code': 'ryq27', 'type': 'trend_vol', 'rule': 'std_8q ≤ peer_median_std', 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'Fixed Asset Turnover', 'code': 'ryq91', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]
    
    def get_metric_series(self, healthcare_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, healthcare_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, healthcare_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (or 8 for trend_vol)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(healthcare_df, criterion['code'], periods=12)
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
            
            if crit_type == 'trend_vol':
                series = self.get_metric_series(healthcare_df, criterion['code'], periods=8)
                if len(series) < 3:
                    return {'pass': False, 'reason': 'Insufficient data for std calculation (need ≥3 periods)', 'values': []}
                
                values = [v for _, v in series]
                company_std = pd.Series(values).std()
                
                peer_stds = []
                for df in all_companies_data.values():
                    peer_series = self.get_metric_series(df, criterion['code'], periods=8)
                    if len(peer_series) >= 3:
                        peer_values = [v for _, v in peer_series]
                        peer_stds.append(pd.Series(peer_values).std())
                
                if not peer_stds:
                    return {'pass': False, 'reason': 'No peer std available', 'value': company_std}
                
                peer_median_std = pd.Series(peer_stds).median()
                pass_check = company_std <= peer_median_std
                
                return {
                    'pass': pass_check,
                    'value': company_std,
                    'peer_median_std': peer_median_std,
                    'periods_used': len(values)
                }

            if crit_type == 'peer_percentile':
                series = self.get_metric_series(healthcare_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                peer_maps = [self.get_metric_period_dict(df, criterion['code']) for df in all_companies_data.values()]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = [pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None]
                    if not peer_values:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= pd.Series(peer_values)).sum() / len(peer_values)
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
    
    def score_healthcare(self, symbol, healthcare_df, all_companies_data):
        """Calculate profitability score for a healthcare (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),  # Cap at 5, force float
            'details': results
        }

# %%
prof_scorer = ProfitabilityScorer()
prof_scores = {}

for idx, (symbol, healthcare_df) in enumerate(healthcare_ratios_data.items()):
    score_result = prof_scorer.score_healthcare(symbol, healthcare_df, healthcare_ratios_data)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 8.0', 'threshold': 8.0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Equity/Total Assets', 'type': 'calculated_ratio', 'formula': 'bsa78 / bsa53', 'codes': ['bsa78', 'bsa53'], 'rule': 'value ≥ 0.50', 'threshold': 0.50, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Short-term investments/Cash', 'type': 'calculated_ratio', 'formula': 'bsa5 / bsa2', 'codes': ['bsa5', 'bsa2'], 'rule': 'value ≥ 0.3', 'threshold': 0.30, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Long-term Assets/Equity', 'type': 'calculated_peer', 'formula': 'bsa23 / bsa78', 'codes': ['bsa23', 'bsa78'], 'rule': 'pct ≤ 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_series(self, healthcare_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, healthcare_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = healthcare_df[healthcare_df['code'] == code].copy()
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
    
    def calculate_ratio_series(self, healthcare_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(healthcare_df, numerator_code)
        denominator_map = self.get_metric_period_dict(healthcare_df, denominator_code)

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
    
    def evaluate_criterion(self, healthcare_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(healthcare_df, criterion['code'], periods=12)
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
            
            if crit_type == 'calculated_ratio':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(healthcare_df, numerator, denominator, periods=12)
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

            if crit_type == 'calculated_peer':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(healthcare_df, numerator, denominator, periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                peer_maps = [
                    {
                        p: v for p, v in self.calculate_ratio_series(df, numerator, denominator, periods=None)
                    }
                    for df in all_companies_data.values()
                ]

                period_passes = []
                percentiles = []
                for period_key, value in period_values.items():
                    peer_values = [pm.get(period_key) for pm in peer_maps if pm.get(period_key) is not None]
                    if not peer_values:
                        period_passes.append(False)
                        percentiles.append(None)
                        continue

                    if criterion['direction'] == 'lower':
                        pct = (value <= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (value >= pd.Series(peer_values)).sum() / len(peer_values)
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
    
    def score_healthcare(self, symbol, healthcare_df, all_companies_data):
        """Calculate solvency score for a healthcare (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(healthcare_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),  # Cap at 5, force float
            'details': results
        }

# %%
solv_scorer = SolvencyScorer()
solv_scores = {}

for idx, (symbol, healthcare_df) in enumerate(healthcare_ratios_data.items()):
    score_result = solv_scorer.score_healthcare(symbol, healthcare_df, healthcare_ratios_data)
    solv_scores[symbol] = score_result

# %%
import numpy as np

class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'PEG Ratio', 'type': 'calculated_absolute', 'formula': 'pe / growth', 'components': {'pe': 'ryd21', 'growth': 'ryq34'}, 'rule': 'value <= 0.8', 'threshold': 0.8, 'direction': 'lower', 'must_have': False},
            {'id': 2, 'name': 'EV/EBITDA relative to ROIC', 'type': 'ratio_vs_peer', 'numerator': 'ryd30', 'denominator': 'ryq76', 'rule': 'ratio <= peer_median', 'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'P/B vs ROE Spread', 'type': 'calculated_absolute', 'formula': '(roe - 0.15) / pb', 'components': {'roe': 'ryq12', 'pb': 'ryd25'},'rule': 'value >= 1.0', 'threshold': 1.0, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Revenue Growth Consistency', 'code': 'ryq34', 'type': 'trend_positive', 'window': 4, 'rule': 'all_positive', 'must_have': False}
        ]

    def get_latest_value(self, df, code):
        """Get the most recent non-null value"""
        if df is None or df.empty: return None
        series = df[df['code'] == code]['value'].dropna()
        return series.iloc[-1] if not series.empty else None

    def get_recent_series(self, df, code, periods=4):
        """Get list of values for trend analysis"""
        if df is None or df.empty: return []
        series = df[df['code'] == code].sort_values('date' if 'date' in df.columns else ['year','quarter'])
        return series['value'].dropna().tail(periods).tolist()

    def evaluate_criterion(self, symbol, healthcare_df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'calculated_absolute':
                comp_values = {}
                for key, code in criterion['components'].items():
                    val = self.get_latest_value(healthcare_df, code)
                    if val is None:
                        return {'pass': False, 'reason': f'Missing {key} ({code})', 'val': None}
                    comp_values[key] = val
                
                val = None
                if criterion['id'] == 1: # PEG
                    pe = comp_values['pe']
                    growth = comp_values['growth'] # Giả sử growth dạng số thập phân (0.2 là 20%)
                    if growth <= 0: # Tránh chia cho số âm hoặc 0
                        return {'pass': False, 'reason': 'Growth non-positive', 'value': f'PE={pe}, G={growth}'}
                    val = pe / (growth * 100) # Thường PEG tính growth là số nguyên (20 not 0.2) -> tùy data của bạn
                    
                elif criterion['id'] == 3: # Spread
                    roe = comp_values['roe']
                    pb = comp_values['pb']
                    if pb <= 0: return {'pass': False, 'reason': 'Negative PB'}
                    val = (roe - 0.15) / pb

                if val is None: return {'pass': False}
                
                pass_check = (val <= criterion['threshold']) if criterion['direction'] == 'lower' else (val >= criterion['threshold'])
                return {'pass': pass_check, 'value': val, 'threshold': criterion['threshold']}

            if crit_type == 'ratio_vs_peer':
                num = self.get_latest_value(healthcare_df, criterion['numerator'])
                den = self.get_latest_value(healthcare_df, criterion['denominator'])
                
                if num is None or den is None or den == 0:
                    return {'pass': False, 'reason': 'Missing data or Zero Div', 'ratio': None}
                
                my_ratio = num / den
                
                peer_ratios = []
                for df in all_companies_data.values():
                    p_num = self.get_latest_value(df, criterion['numerator'])
                    p_den = self.get_latest_value(df, criterion['denominator'])
                    if p_num is not None and p_den is not None and p_den != 0:
                        peer_ratios.append(p_num / p_den)
                
                if not peer_ratios: return {'pass': False}
                
                peer_median = np.median(peer_ratios)
                pass_check = my_ratio <= peer_median # Lower is better
                return {'pass': pass_check, 'value': my_ratio, 'peer_median': peer_median}

            if crit_type == 'trend_positive':
                vals = self.get_recent_series(healthcare_df, criterion['code'], periods=criterion['window'])
                if len(vals) < criterion['window']:
                    return {'pass': False, 'reason': 'Not enough history'}
                
                pass_check = all(v > 0 for v in vals)
                return {'pass': pass_check, 'values': vals}

            return {'pass': False, 'reason': 'Unknown type'}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score(self, symbol, healthcare_df, all_companies_data):
        results = {}
        points = 0
        total_criteria = len(self.criteria) # Ở đây là 4 (đã bỏ 1 cái trùng)
        
        weight = 5.0 / total_criteria 

        for crit in self.criteria:
            res = self.evaluate_criterion(symbol, healthcare_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = res
            if res['pass']:
                points += weight
        
        return {'symbol': symbol, 'score': round(points, 2), 'details': results}

# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, healthcare_df in healthcare_ratios_data.items():
        val_scores[symbol] = val_scorer.score(symbol, healthcare_df, healthcare_ratios_data)    
    
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

    print(f"Healthcare Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_HEALTHCARE] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()