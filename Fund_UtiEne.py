# %%
import requests
import pandas as pd

uti_ene = ['Utilities', 'Oil & Gas']
all_stocks = []
uti_ene_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
uti_ene_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    filtered_stocks = [
        s for s in all_stocks
        if s.get('industry_lv1') in uti_ene]
    uti_ene_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(uti_ene_symbols),
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

    # Sort by ryd11 2025 Q3 and take top 100
    filtered_stocks = sorted(
        filtered_stocks,
        key=lambda s: float(ryd11_2025_q3.get(s.get('symbol'), float('-inf'))),
        reverse=True
    )
    top_50_stocks = filtered_stocks[:50]
    uti_ene_symbols = [s['symbol'] for s in top_50_stocks]

    # Convert to DataFrame
    df_ratios = pd.DataFrame(all_ratios_data)

    if 'symbol' in df_ratios.columns:
        for symbol in uti_ene_symbols:
            symbol_data = df_ratios[df_ratios['symbol'] == symbol]
            if not symbol_data.empty:
                uti_ene_ratios_data[symbol] = symbol_data.copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_UtiEne] Warning: failed to fetch uti_ene ratios: {e}")

# %%
print(uti_ene_symbols)

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Current Ratio', 'code': 'ryq3', 'type': 'absolute', 'rule': 'value ≥ 1.2', 'threshold': 1.2, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Quick Ratio', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value ≥ 0.5', 'threshold': 0.5, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'DSO < 180 Days', 'code': 'ryq16', 'type': 'absolute', 'rule': 'value ≤ 180', 'threshold': 180, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'Cash Ratio', 'code': 'ryq1', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.60', 'cut': 0.60, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Net Debt / Equity', 'type': 'calculated_absolute', 'formula': '(bsl1 + bsl2 - bsa2) / bse1', 'codes': ['bsl1', 'bsl2', 'bsa2', 'bse1'], 'rule': 'value ≤ 1.5', 'threshold': 1.5, 'direction': 'lower', 'must_have': False},
        ]

    def get_metric_series(self, uti_ene_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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

    def get_metric_period_dict(self, uti_ene_df, code):
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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

    def calculate_ratio_series(self, uti_ene_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(uti_ene_df, numerator_code)
        denominator_map = self.get_metric_period_dict(uti_ene_df, denominator_code)

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
    
    def evaluate_criterion(self, uti_ene_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(uti_ene_df, criterion['code'], periods=12)
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

            if crit_type == 'calculated_absolute':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(uti_ene_df, numerator, denominator, periods=12)
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

            if crit_type in {'peer_percentile', 'calculated_peer'}:
                if crit_type == 'calculated_peer':
                    numerator, denominator = criterion['codes']
                    series = self.calculate_ratio_series(uti_ene_df, numerator, denominator, periods=12)
                else:
                    series = self.get_metric_series(uti_ene_df, criterion['code'], periods=12)

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
    
    def score_uti_ene(self, symbol, uti_ene_df, all_companies_data):
        """Calculate liquidity score for a uti_ene (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),  # Cap at 5, force float
            'details': results
        }

# %%
scorer = LiquidityScorer()
liq_scores = {}

for idx, (symbol, uti_ene_df) in enumerate(uti_ene_ratios_data.items()):
    score_result = scorer.score_uti_ene(symbol, uti_ene_df, uti_ene_ratios_data)
    liq_scores[symbol] = score_result

# %%
print(f"LIQ_Score: {[liq_scores[s]['score'] for s in sorted(liq_scores.keys())]}")

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Positive Net Profit', 'code': 'ryq29', 'type': 'absolute', 'rule': 'value > 0', 'threshold': 0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'ROIC >= 8%', 'code': 'ryq76', 'type': 'absolute', 'rule': 'value >= 0.08', 'threshold': 0.08, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Gross Profit Margin Stability', 'code': 'ryq25', 'type': 'peer_percentile', 'rule': 'pct >= 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Fixed Asset Turnover', 'code': 'ryq91', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.60', 'cut': 0.60, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Margin Stability', 'code': 'ryq25', 'type': 'trend_vol', 'rule': 'std_8q <= peer_median_std', 'direction': 'lower', 'must_have': False},]
    
    def get_metric_series(self, uti_ene_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, uti_ene_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, uti_ene_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (or 8 for trend_vol)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(uti_ene_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(uti_ene_df, criterion['code'], periods=8)
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
                series = self.get_metric_series(uti_ene_df, criterion['code'], periods=12)
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
    
    def score_uti_ene(self, symbol, uti_ene_df, all_companies_data):
        """Calculate profitability score for a utilities/energy (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
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

for idx, (symbol, uti_ene_df) in enumerate(uti_ene_ratios_data.items()):
    score_result = prof_scorer.score_uti_ene(symbol, uti_ene_df, uti_ene_ratios_data)
    prof_scores[symbol] = score_result

# %%
print(f"Prof_Score: {[prof_scores[s]['score'] for s in sorted(prof_scores.keys())]}")

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 4.0', 'threshold': 4.0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': '(ST+LT borrowings)/Total Assets', 'type': 'calculated_absolute', 'formula': '(bsa56+bsa71)/bsa53', 'codes': ['bsa56', 'bsa71', 'bsa53'], 'rule': 'value ≤ 0.6', 'threshold': 0.6, 'direction': 'lower', 'must_have': True},
            {'id': 3, 'name': 'Long-term Assets/Long-term Debt', 'type': 'calculated_absolute', 'formula': 'bsa23/bsa71', 'codes': ['bsa23', 'bsa71'], 'rule': 'value ≥ 1.5', 'threshold': 1.5, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Equity/Fixed Assets', 'type': 'calculated_peer', 'formula': 'bsa78/bsa29', 'codes': ['bsa78', 'bsa29'], 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Debt/EBITDA Proxy', 'type': 'calculated_peer', 'formula': '(bsa56+bsa71)/rqq72', 'codes': ['bsa56', 'bsa71', 'rqq72'], 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_series(self, uti_ene_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, uti_ene_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = uti_ene_df[uti_ene_df['code'] == code].copy()
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
    
    def calculate_ratio_series(self, uti_ene_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(uti_ene_df, numerator_code)
        denominator_map = self.get_metric_period_dict(uti_ene_df, denominator_code)

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
    
    def calculate_complex_formula_series(self, uti_ene_df, codes, formula, periods=12):
        """Calculate formula-based series for complex expressions like (bsa56+bsa71)/bsa53"""
        all_maps = {code: self.get_metric_period_dict(uti_ene_df, code) for code in codes}
        
        # Find common periods
        common_periods = set(all_maps[codes[0]].keys())
        for code in codes[1:]:
            common_periods &= set(all_maps[code].keys())
        
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
            try:
                # Build local context for eval
                local_context = {code: all_maps[code][p] for code in codes}
                result = eval(formula, {"__builtins__": {}}, local_context)
                if result is not None:
                    series.append((p, result))
            except:
                continue
        
        return series
    
    def evaluate_criterion(self, uti_ene_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(uti_ene_df, criterion['code'], periods=12)
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
                series = self.calculate_complex_formula_series(uti_ene_df, criterion['codes'], criterion['formula'], periods=12)
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
                # Check if it's a complex formula (multiple codes)
                if len(criterion['codes']) > 2:
                    series = self.calculate_complex_formula_series(uti_ene_df, criterion['codes'], criterion['formula'], periods=12)
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_complex_formula_series(df, criterion['codes'], criterion['formula'], periods=None)
                        }
                        for df in all_companies_data.values()
                    ]
                else:
                    numerator, denominator = criterion['codes']
                    series = self.calculate_ratio_series(uti_ene_df, numerator, denominator, periods=12)
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_ratio_series(df, numerator, denominator, periods=None)
                        }
                        for df in all_companies_data.values()
                    ]
                
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}

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
    
    def score_uti_ene(self, symbol, uti_ene_df, all_companies_data):
        """Calculate solvency score for a uti_ene (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(uti_ene_df, crit, all_companies_data)
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

for idx, (symbol, uti_ene_df) in enumerate(uti_ene_ratios_data.items()):
    score_result = solv_scorer.score_uti_ene(symbol, uti_ene_df, uti_ene_ratios_data)
    solv_scores[symbol] = score_result

# %%
print(f"Solv_Score: {[solv_scores[s]['score'] for s in sorted(solv_scores.keys())]}")

# %%
import numpy as np

class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'EV/EBITDA', 'code': 'ryd30', 'type': 'peer_percentile', 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 2, 'name': 'P/B vs ROE valuation', 'type': 'ratio_vs_peer','numerator': 'ryd25', 'denominator': 'ryq12','rule': 'ratio <= peer_median', 'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'Earnings Yield', 'type': 'calculated_absolute', 'formula': 'roe / pb', 'components': {'roe': 'ryq12', 'pb': 'ryd25'}, 'rule': 'value >= 0.06', 'threshold': 0.06, 'direction': 'higher', 'must_have': False}, 
            {'id': 4, 'name': 'P/E Ratio', 'code': 'ryd21', 'type': 'peer_percentile','rule': 'pct <= 0.40', 'cut': 0.40, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'P/B Absolute Safe Zone', 'code': 'ryd25', 'type': 'absolute', 'rule': 'value <= 1.5', 'threshold': 1.5, 'direction': 'lower', 'must_have': False}
        ]

    def get_latest_value(self, df, code):
        if df is None or df.empty: return None
        if 'date' in df.columns:
            series = df[df['code'] == code].sort_values('date')['value']
        else:
            series = df[df['code'] == code]['value']
        return series.iloc[-1] if not series.empty else None

    def evaluate_criterion(self, symbol, df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'peer_percentile':
                val = self.get_latest_value(df, criterion['code'])
                if val is None: return {'pass': False, 'reason': 'Missing data'}

                peer_vals = []
                for d in all_companies_data.values():
                    v = self.get_latest_value(d, criterion['code'])
                    if v is not None: peer_vals.append(v)
                
                if not peer_vals: return {'pass': False}
                
                pct = (val >= pd.Series(peer_vals)).mean()

                if criterion['direction'] == 'lower':
                    pass_check = pct <= criterion['cut']
                else:
                    pass_check = pct >= criterion['cut']

                return {'pass': pass_check, 'value': val, 'percentile': pct}

            if crit_type == 'ratio_vs_peer':
                num = self.get_latest_value(df, criterion['numerator'])
                den = self.get_latest_value(df, criterion['denominator'])
                
                if num is None or den is None or den <= 0: 
                    return {'pass': False, 'reason': 'Missing or Neg Denom'}
                
                my_ratio = num / den

                peer_ratios = []
                for d in all_companies_data.values():
                    n = self.get_latest_value(d, criterion['numerator'])
                    d_val = self.get_latest_value(d, criterion['denominator'])
                    if n is not None and d_val is not None and d_val > 0:
                        peer_ratios.append(n / d_val)
                
                if not peer_ratios: return {'pass': False}
                
                peer_median = np.median(peer_ratios)
                pass_check = my_ratio <= peer_median # Lower is better
                return {'pass': pass_check, 'value': my_ratio, 'peer_median': peer_median}

            if crit_type == 'calculated_absolute':
                # Earnings Yield = ROE / PB
                roe = self.get_latest_value(df, criterion['components']['roe'])
                pb = self.get_latest_value(df, criterion['components']['pb'])
                
                if roe is None or pb is None or pb <= 0:
                    return {'pass': False, 'reason': 'Missing data'}
                
                val = roe / pb 
                if criterion['direction'] == 'higher':
                    pass_check = val >= criterion['threshold']
                else:
                    pass_check = val <= criterion['threshold']
                    
                return {'pass': pass_check, 'value': val}

            return {'pass': False}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score(self, symbol, df, all_companies_data):
        results = {}
        points = 0
        weight = 5.0 / len(self.criteria)

        for crit in self.criteria:
            res = self.evaluate_criterion(symbol, df, crit, all_companies_data)
            results[f"C{crit['id']}"] = res
            if res['pass']:
                points += weight
        
        return {'symbol': symbol, 'score': round(points, 2), 'details': results}

# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, uti_ene_df in uti_ene_ratios_data.items():
        val_scores[symbol] = val_scorer.score(symbol, uti_ene_df, uti_ene_ratios_data)    
    
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

    print(f"UtiEne Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_UTIENE] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()