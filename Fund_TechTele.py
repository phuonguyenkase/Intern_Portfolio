# %%
import requests
import pandas as pd

all_stocks = []
tech_tele_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
tech_tele_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all tech and telecommunication symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    tech_tele_symbols = [s['symbol'] for s in all_stocks if s.get('industry_lv1') in ['Technology', 'Telecommunications']]
    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(tech_tele_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }
    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    # Convert to DataFrame
    df_ratios = pd.DataFrame(all_ratios_data)

    # Handle date columns - stock-ratios may use year/quarter instead of date
    if 'date' in df_ratios.columns:
        df_ratios['date'] = pd.to_datetime(df_ratios['date'], errors='coerce')
    elif 'year' in df_ratios.columns and 'quarter' in df_ratios.columns:
        # Create date from year and quarter (end of quarter)
        quarter_months = {1: 3, 2: 6, 3: 9, 4: 12}
        df_ratios['date'] = pd.to_datetime(
            df_ratios.apply(
                lambda row: f"{int(row['year'])}-{quarter_months.get(int(row['quarter']), 12)}-01",
                axis=1
            ),
            errors='coerce'
        )

    # Organize by tech and telecommunication
    if 'symbol' in df_ratios.columns:
        for symbol in sorted(df_ratios['symbol'].unique()):
            tech_tele_ratios_data[symbol] = df_ratios[df_ratios['symbol'] == symbol].copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_TechTele] Warning: failed to fetch tech and telecommunication ratios: {e}")

# %%
print(tech_tele_symbols)

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Current Ratio', 'code': 'ryq3', 'type': 'absolute', 'rule': 'value ≥ 1', 'threshold': 1, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Cash & Equivalents/Revenue', 'type': 'calculated_peer', 'formula': 'bsa2 / rev', 'codes': ['bsa2', 'rev'], 'rule': 'pct ≥ 0.70', 'cut': 0.70, 'direction': 'higher', 'must_have': False},
            {'id': 3, 'name': 'Days Sales Outstanding (DSO)', 'code': 'ryq16', 'type': 'absolute', 'rule': 'value ≤ 100', 'threshold': 100, 'direction': 'lower', 'must_have': True},
            {'id': 4, 'name': 'Quick Ratio', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value ≥ 0.8', 'threshold': 0.8, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Net Cash / Total Assets', 'type': 'calculated_peer', 'formula': '(bsa2 - bsl1) / bsa1', 'codes': ['bsa2', 'bsl1', 'bsa1'], 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]

    def get_metric_series(self, tech_tele_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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

    def get_metric_period_dict(self, tech_tele_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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

    def calculate_ratio_series(self, tech_tele_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(tech_tele_df, numerator_code)
        denominator_map = self.get_metric_period_dict(tech_tele_df, denominator_code)

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
    
    def evaluate_criterion(self, tech_tele_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(tech_tele_df, criterion['code'], periods=12)
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
                series = self.calculate_ratio_series(tech_tele_df, numerator, denominator, periods=12)
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
                series = self.calculate_ratio_series(tech_tele_df, numerator, denominator, periods=12)
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
    
    def score_tech_tele(self, symbol, tech_tele_df, all_companies_data):
        """Calculate liquidity score for a tech_tele company (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
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

for idx, (symbol, tech_tele_df) in enumerate(tech_tele_ratios_data.items()):
    score_result = scorer.score_tech_tele(symbol, tech_tele_df, tech_tele_ratios_data)
    liq_scores[symbol] = score_result

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Positive Net Profit', 'code': 'ryq29', 'type': 'absolute', 'rule': 'value > 0', 'threshold': 0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'ROE > 10%', 'code': 'ryq12', 'type': 'absolute', 'rule': 'value ≥ 0.10', 'threshold': 0.10, 'direction': 'higher', 'must_have': False},
            {'id': 3, 'name': 'Gross Margin > 15%', 'code': 'ryq25', 'type': 'absolute', 'rule': 'value ≥ 0.15', 'threshold': 0.15, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Revenue Growth > 5%', 'code': 'ryq34', 'type': 'absolute', 'rule': 'value ≥ 0.05', 'threshold': 0.05, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Asset Turnover > 1.0', 'code': 'ryq28', 'type': 'absolute', 'rule': 'value ≥ 1.0', 'threshold': 1.0, 'direction': 'higher', 'must_have': False},
        ]
    
    def get_metric_series(self, tech_tele_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, tech_tele_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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
    
    def calculate_ratio_series(self, tech_tele_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(tech_tele_df, numerator_code)
        denominator_map = self.get_metric_period_dict(tech_tele_df, denominator_code)

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
    
    def calculate_sum_ratio_series(self, tech_tele_df, numerator_codes, denominator_code, periods=12):
        """Calculate ratio series with summed numerator: (code1 + code2 + ...) / denominator"""
        numerator_maps = [self.get_metric_period_dict(tech_tele_df, code) for code in numerator_codes]
        denominator_map = self.get_metric_period_dict(tech_tele_df, denominator_code)

        # Find common periods across all numerators and denominator
        common_periods = set(numerator_maps[0].keys()) if numerator_maps else set()
        for nm in numerator_maps[1:]:
            common_periods = common_periods.intersection(set(nm.keys()))
        common_periods = common_periods.intersection(set(denominator_map.keys()))
        
        common_periods = list(common_periods)
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
            if denom is None or denom == 0:
                continue
            num_sum = sum(nm.get(p, 0) for nm in numerator_maps if nm.get(p) is not None)
            if num_sum is None or num_sum == 0:
                continue
            series.append((p, num_sum / denom))

        return series
    
    def evaluate_criterion(self, tech_tele_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(tech_tele_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(tech_tele_df, criterion['code'], periods=12)
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
            
            if crit_type == 'calculated_peer':
                # Handle R&D Intensity Proxy: (bsa43 + bsa175) / rev
                if criterion['id'] == 6:  # R&D Intensity
                    numerator_codes = ['bsa43', 'bsa175']
                    denominator_code = 'rev'
                    series = self.calculate_sum_ratio_series(tech_tele_df, numerator_codes, denominator_code, periods=12)
                else:
                    # Generic calculated_peer with single numerator/denominator
                    numerator, denominator = criterion['codes'][0], criterion['codes'][1]
                    series = self.calculate_ratio_series(tech_tele_df, numerator, denominator, periods=12)
                
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                
                if criterion['id'] == 6:  # R&D Intensity
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_sum_ratio_series(df, ['bsa43', 'bsa175'], 'rev', periods=None)
                        }
                        for df in all_companies_data.values()
                    ]
                else:
                    numerator, denominator = criterion['codes'][0], criterion['codes'][1]
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
    
    def score_tech_tele(self, symbol, tech_tele_df, all_companies_data):
        """Calculate profitability score for a tech_tele company (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
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

for idx, (symbol, tech_tele_df) in enumerate(tech_tele_ratios_data.items()):
    score_result = prof_scorer.score_tech_tele(symbol, tech_tele_df, tech_tele_ratios_data)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Debt/Equity', 'code': 'ryq10', 'type': 'absolute', 'rule': 'value ≤ 1.0', 'threshold': 1.0, 'direction': 'lower', 'must_have': True},
            {'id': 2, 'name': 'Interest Coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 6.0', 'threshold': 6.0, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Equity/Total Assets', 'type': 'calculated_ratio', 'formula': 'bsa78 / bsa53', 'codes': ['bsa78', 'bsa53'], 'rule': 'value ≥ 0.45', 'threshold': 0.45, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Intangible Assets/Total Assets', 'type': 'calculated_peer', 'formula': 'bsa209 / bsa53', 'codes': ['bsa209', 'bsa53'], 'rule': 'pct ≤ median+0.1', 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_series(self, tech_tele_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, tech_tele_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = tech_tele_df[tech_tele_df['code'] == code].copy()
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
    
    def calculate_ratio_series(self, tech_tele_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(tech_tele_df, numerator_code)
        denominator_map = self.get_metric_period_dict(tech_tele_df, denominator_code)

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
    
    def evaluate_criterion(self, tech_tele_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(tech_tele_df, criterion['code'], periods=12)
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
                series = self.calculate_ratio_series(tech_tele_df, numerator, denominator, periods=12)
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
                series = self.calculate_ratio_series(tech_tele_df, numerator, denominator, periods=12)
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

                    # For Intangible Assets: compare against peer_median + 0.1 (or use percentile approach)
                    if criterion['id'] == 4:  # Intangible Assets/Total Assets
                        peer_median = pd.Series(peer_values).median()
                        threshold = peer_median + 0.1
                        if criterion['direction'] == 'lower':
                            pass_check = value <= threshold
                        else:
                            pass_check = value >= threshold
                    else:
                        # Standard percentile approach
                        if criterion['direction'] == 'lower':
                            pct = (value <= pd.Series(peer_values)).sum() / len(peer_values)
                            pass_check = pct <= criterion['cut']
                        else:
                            pct = (value >= pd.Series(peer_values)).sum() / len(peer_values)
                            pass_check = pct >= criterion['cut']

                    period_passes.append(pass_check)
                    percentiles.append(peer_median if criterion['id'] == 4 else (pct if 'cut' in criterion else None))
                
                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5
                
                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': list(period_values.values()),
                    'percentiles': percentiles,
                    'periods_used': len(period_values)
                }

            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_tech_tele(self, symbol, tech_tele_df, all_companies_data):
        """Calculate solvency score for a tech_tele company (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(tech_tele_df, crit, all_companies_data)
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

for idx, (symbol, tech_tele_df) in enumerate(tech_tele_ratios_data.items()):
    score_result = solv_scorer.score_tech_tele(symbol, tech_tele_df, tech_tele_ratios_data)
    solv_scores[symbol] = score_result

# %%
import numpy as np
class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'PEG Ratio', 'type': 'calculated_absolute', 'formula': 'pe / (growth * 100)', 'components': {'pe': 'ryd21', 'growth': 'ryq34'}, 'rule': 'value <= 1.0', 'threshold': 1.0, 'direction': 'lower', 'must_have': False},
            {'id': 2, 'name': 'EV/EBITDA', 'type': 'peer_percentile', 'code': 'ryd30', 'rule': 'pct <= 0.40', 'cut': 0.40, 'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'Market Cap / Gross Profit', 'type': 'calculated_peer','formula': 'mkt_cap / (gross_margin * rev)', 'components': {'mkt_cap': 'ryd11', 'gross_margin': 'ryq25', 'rev': 'rev'},'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'Rule of 40', 'type': 'calculated_absolute','formula': '(growth + margin) * 100', 'components': {'growth': 'ryq34', 'margin': 'ryq29'},'rule': 'value >= 40', 'threshold': 40, 'direction': 'higher', 'must_have': False}
        ]

    def get_latest_value(self, df, code):
        """Lấy giá trị quý gần nhất"""
        if df is None or df.empty: return None
        if 'date' in df.columns:
            series = df[df['code'] == code].sort_values('date')['value']
        else:
            series = df[df['code'] == code]['value']
            
        return series.iloc[-1] if not series.empty else None

    def evaluate_criterion(self, symbol, tech_tele_df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'peer_percentile':
                val = self.get_latest_value(tech_tele_df, criterion['code'])
                if val is None: return {'pass': False, 'reason': 'Missing data'}

                peer_vals = []
                for df in all_companies_data.values():
                    v = self.get_latest_value(df, criterion['code'])
                    if v is not None: peer_vals.append(v)
                
                if not peer_vals: return {'pass': False}
                
                pct = (val >= pd.Series(peer_vals)).mean() # Mặc định là higher rank
                
                
                if criterion['direction'] == 'lower':
                    pct = (val <= pd.Series(peer_vals)).mean() 
                    pass_check = pct <= criterion['cut']
                else:
                    pass_check = pct >= criterion['cut']

                return {'pass': pass_check, 'value': val, 'percentile': pct}

            if crit_type == 'calculated_absolute':
                comps = {}
                for key, code in criterion['components'].items():
                    v = self.get_latest_value(tech_tele_df, code)
                    if v is None: return {'pass': False, 'reason': f'Missing {key}'}
                    comps[key] = v
                
                val = None
                if criterion['id'] == 1: # PEG
                    pe = comps['pe']
                    growth = comps['growth']
                    if growth <= 0.05:
                        return {'pass': False, 'reason': 'Growth <= 5% or Neg', 'value': growth}
                    val = pe / (growth * 100)
                
                elif criterion['id'] == 4: # Rule of 40
                    val = (comps['growth'] + comps['margin']) * 100
                
                if val is None: return {'pass': False}
                
                if criterion['direction'] == 'lower':
                    pass_check = val <= criterion['threshold']
                else:
                    pass_check = val >= criterion['threshold']
                    
                return {'pass': pass_check, 'value': val}

            if crit_type == 'calculated_peer':
                def calc_formula(df_input):
                    c = {}
                    for key, code in criterion['components'].items():
                        v = self.get_latest_value(df_input, code)
                        if v is None: return None
                        c[key] = v
                    if c['gross_margin'] == 0: return None
                    return c['mkt_cap'] / (c['gross_margin'] * c['rev'])

                my_val = calc_formula(tech_tele_df)
                if my_val is None: return {'pass': False, 'reason': 'Calc Error'}
                
                peer_vals = []
                for df in all_companies_data.values():
                    pv = calc_formula(df)
                    if pv is not None: peer_vals.append(pv)
                
                if not peer_vals: return {'pass': False}
                
                pct = (my_val <= pd.Series(peer_vals)).mean()
                pass_check = pct <= criterion['cut']
                
                return {'pass': pass_check, 'value': my_val, 'percentile': pct}

            return {'pass': False}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score(self, symbol, tech_tele_df, all_companies_data):
        results = {}
        points = 0
        weight = 5.0 / 4 

        for crit in self.criteria:
            res = self.evaluate_criterion(symbol, tech_tele_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = res
            if res['pass']:
                points += weight
        
        return {'symbol': symbol, 'score': round(points, 2), 'details': results}
    
    def score_tech_tele(self, symbol, tech_tele_df, all_companies_data):
        return self.score(symbol, tech_tele_df, all_companies_data)

# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, tech_tele_df in tech_tele_ratios_data.items():
        val_scores[symbol] = val_scorer.score_tech_tele(symbol, tech_tele_df, tech_tele_ratios_data)    
    
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

    print(f"Tech & Tele Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_TECH_TELE] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()
# %%
