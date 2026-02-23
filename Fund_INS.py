# %%
import requests
import pandas as pd
import os
import TechAna_DRAFT as TechAna

all_stocks = []
insurance_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
insurance_ratios_data = {}
as_of_date = os.getenv("Fund_AsOf_Date")

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all insurance symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    insurance_symbols = [s['symbol'] for s in all_stocks if s.get('industry_lv2') == 'Insurance']
    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(insurance_symbols),
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
    
    if not as_of_date:
        try:
            end_dt = pd.to_datetime(getattr(TechAna, 'END_DATE', None), errors = "coerce")
            if pd.notna(end_dt):
                prev_q_end = (end_dt.to_period('Q') - 1).end_time
                as_of_date = prev_q_end.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Organize by insurance
    if 'symbol' in df_ratios.columns:
        for symbol in sorted(df_ratios['symbol'].unique()):
            insurance_ratios_data[symbol] = df_ratios[df_ratios['symbol'] == symbol].copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_INS] Warning: failed to fetch insurance ratios: {e}")

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Total Investment Ratio', 'type': 'calculated_absolute', 'formula': '(bsa2 + bsa5 + bsa43) / bsa53', 'codes': ['bsa2', 'bsa5', 'bsa43', 'bsa53'], 'rule': 'value >= 0.50', 'threshold': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 2, 'name': 'Immediate Liquidity Rank', 'type': 'calculated_peer', 'formula': '(bsa2 + bsa5) / bsa53', 'codes': ['bsa2', 'bsa5', 'bsa53'], 'rule': 'pct >= 0.40', 'cut': 0.40, 'direction': 'higher', 'must_have': False},
            {'id': 3, 'name': 'Receivables Control Rank', 'type': 'calculated_peer', 'formula': 'bsa8 / bsa53', 'codes': ['bsa8', 'bsa53'], 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'Capital Buffer Rank', 'type': 'calculated_peer', 'formula': 'bsa78 / bsa53', 'codes': ['bsa78', 'bsa53'], 'rule': 'pct >= 0.40', 'cut': 0.40, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Quick Ratio Safe Zone', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value >= 0.8', 'threshold': 0.8, 'direction': 'higher', 'must_have': False},
        ]

    def get_metric_series(self, insurance_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def get_metric_period_dict(self, insurance_df, code):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def calculate_ratio_series(self, insurance_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(insurance_df, numerator_code)
        denominator_map = self.get_metric_period_dict(insurance_df, denominator_code)

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
    
    def calculate_complex_formula_series(self, insurance_df, codes, formula, periods=12):
        """Calculate formula-based series for complex expressions like ryq18+ryq16-ryq20"""
        all_maps = {code: self.get_metric_period_dict(insurance_df, code) for code in codes}
        
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
    
    def evaluate_criterion(self, insurance_df, criterion, all_companies_data):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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
                series = self.calculate_complex_formula_series(insurance_df, criterion['codes'], criterion['formula'], periods=12)
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
            
            if crit_type in {'peer_percentile', 'calculated_peer', 'calculated_peer_percentile'}:
                if crit_type == 'calculated_peer':
                    numerator, denominator = criterion['codes']
                    series = self.calculate_ratio_series(insurance_df, numerator, denominator, periods=12)
                elif crit_type == 'calculated_peer_percentile':
                    series = self.calculate_complex_formula_series(insurance_df, criterion['codes'], criterion['formula'], periods=12)
                else:
                    series = self.get_metric_series(insurance_df, criterion['code'], periods=12)

                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}

                if crit_type == 'calculated_peer':
                    numerator, denominator = criterion['codes']
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_ratio_series(df, numerator, denominator, periods=None)
                        }
                        for df in all_companies_data.values()
                    ]
                elif crit_type == 'calculated_peer_percentile':
                    peer_maps = [
                        {
                            p: v for p, v in self.calculate_complex_formula_series(df, criterion['codes'], criterion['formula'], periods=None)
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
    
    def score_insurance(self, symbol, insurance_df, all_companies_data):
        """Calculate liquidity score for a insurance (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),  # Cap at 5, force float
            'details': results
        }

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Positive Net Profit', 'code': 'ryq29', 'type': 'absolute', 'rule': 'value > 0', 'threshold': 0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'ROE >= 10%', 'code': 'ryq12', 'type': 'absolute', 'rule': 'value >= 0.10', 'threshold': 0.10, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'ROA Rank', 'code': 'ryq14', 'type': 'peer_percentile', 'rule': 'pct >= 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Net Margin Rank', 'code': 'ryq29', 'type': 'peer_percentile', 'rule': 'pct >= 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 5, 'name': 'Profit Growth Rank', 'code': 'ryq39', 'type': 'peer_percentile', 'rule': 'pct >= 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]

    def get_metric_series(self, insurance_df, code, periods=12):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def get_metric_period_dict(self, insurance_df, code):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def evaluate_criterion(self, insurance_df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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

    def score_insurance(self, symbol, insurance_df, all_companies_data):
        results = {}
        points = 0
        
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),
            'details': results
        }

# %%
scorer = LiquidityScorer()
liq_scores = {}

for idx, (symbol, insurance_df) in enumerate(insurance_ratios_data.items()):
    score_result = scorer.score_insurance(symbol, insurance_df, insurance_ratios_data)
    liq_scores[symbol] = score_result

# %%
prof_scorer = ProfitabilityScorer()
prof_scores = {}

for idx, (symbol, insurance_df) in enumerate(insurance_ratios_data.items()):
    score_result = prof_scorer.score_insurance(symbol, insurance_df, insurance_ratios_data)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Financial Debt/Equity <= 0.5', 'code': 'ryq6', 'type': 'absolute', 'rule': 'value <= 0.5', 'threshold': 0.5, 'direction': 'lower', 'must_have': True},
            {'id': 2, 'name': 'Financial Leverage Rank', 'code': 'ryq71', 'type': 'peer_percentile', 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'Equity/Assets Ratio', 'type': 'calculated_peer', 'formula': 'bsa78 / bsa53', 'codes': ['bsa78', 'bsa53'], 'rule': 'pct >= 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'Debt/Equity Rank', 'code': 'ryq10', 'type': 'peer_percentile', 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'Positive Retained Earnings', 'code': 'bsa90', 'type': 'absolute', 'rule': 'value > 0', 'threshold': 0, 'direction': 'higher', 'must_have': False}
        ]

    def get_metric_series(self, insurance_df, code, periods=12):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def get_metric_period_dict(self, insurance_df, code):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def calculate_ratio_series(self, insurance_df, numerator_code, denominator_code, periods=12):
        numerator_map = self.get_metric_period_dict(insurance_df, numerator_code)
        denominator_map = self.get_metric_period_dict(insurance_df, denominator_code)

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

    def evaluate_criterion(self, insurance_df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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

            if crit_type == 'calculated_peer':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(insurance_df, numerator, denominator, periods=12)
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

            if crit_type == 'peer_percentile':
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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

    def score_insurance(self, symbol, insurance_df, all_companies_data):
        results = {}
        points = 0
        
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),
            'details': results
        }

# %%
solv_scorer = SolvencyScorer()
solv_scores = {}

for idx, (symbol, insurance_df) in enumerate(insurance_ratios_data.items()):
    score_result = solv_scorer.score_insurance(symbol, insurance_df, insurance_ratios_data)
    solv_scores[symbol] = score_result

# %%
import numpy as np

class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'P/B Rank', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 2, 'name': 'P/B <= 2.5', 'code': 'ryd25', 'type': 'absolute', 'rule': 'value <= 2.5', 'threshold': 2.5, 'direction': 'lower', 'must_have': True},
            {'id': 3, 'name': 'Earnings Yield >= 6%', 'type': 'calculated_absolute', 'formula': 'ryq12 / ryd25', 'codes': ['ryq12', 'ryd25'], 'rule': 'value >= 0.06', 'threshold': 0.06, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'P/B vs ROE', 'type': 'ratio_vs_peer', 'numerator': 'ryd25', 'denominator': 'ryq12', 'rule': 'ratio <= peer_median', 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'Market Cap / Equity Rank', 'type': 'calculated_peer', 'formula': 'ryd11 / bsa78', 'codes': ['ryd11', 'bsa78'], 'rule': 'pct <= 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False}
        ]

    def get_metric_series(self, insurance_df, code, periods=12):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def get_metric_period_dict(self, insurance_df, code):
        metric_data = insurance_df[insurance_df['code'] == code].copy()
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

    def calculate_ratio_series(self, insurance_df, numerator_code, denominator_code, periods=12):
        numerator_map = self.get_metric_period_dict(insurance_df, numerator_code)
        denominator_map = self.get_metric_period_dict(insurance_df, denominator_code)

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

    def calculate_complex_formula_series(self, insurance_df, codes, formula, periods=12):
        all_maps = {code: self.get_metric_period_dict(insurance_df, code) for code in codes}
        
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
                local_context = {code: all_maps[code][p] for code in codes}
                result = eval(formula, {"__builtins__": {}}, local_context)
                if result is not None:
                    series.append((p, result))
            except:
                continue
        
        return series

    def evaluate_criterion(self, insurance_df, criterion, all_companies_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(insurance_df, criterion['code'], periods=12)
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

            if crit_type == 'calculated_absolute':
                series = self.calculate_complex_formula_series(insurance_df, criterion['codes'], criterion['formula'], periods=12)
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

            if crit_type == 'ratio_vs_peer':
                num_map = self.get_metric_period_dict(insurance_df, criterion['numerator'])
                denom_map = self.get_metric_period_dict(insurance_df, criterion['denominator'])

                period_values = {}
                for p in num_map.keys():
                    if p in denom_map and denom_map[p] != 0:
                        period_values[p] = num_map[p] / denom_map[p]

                if not period_values:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                peer_ratios_by_period = {}
                for df in all_companies_data.values():
                    p_num_map = self.get_metric_period_dict(df, criterion['numerator'])
                    p_denom_map = self.get_metric_period_dict(df, criterion['denominator'])
                    for p in p_num_map.keys():
                        if p in p_denom_map and p_denom_map[p] != 0:
                            if p not in peer_ratios_by_period:
                                peer_ratios_by_period[p] = []
                            peer_ratios_by_period[p].append(p_num_map[p] / p_denom_map[p])

                period_passes = []
                for p, value in period_values.items():
                    if p in peer_ratios_by_period:
                        peer_values = peer_ratios_by_period[p]
                        peer_median = pd.Series(peer_values).median()
                        if criterion['direction'] == 'lower':
                            period_passes.append(value <= peer_median)
                        else:
                            period_passes.append(value >= peer_median)
                    else:
                        period_passes.append(False)

                pass_rate = sum(period_passes) / len(period_passes) if period_passes else 0
                is_passed = pass_rate >= 0.5

                return {
                    'pass': is_passed,
                    'pass_rate': pass_rate,
                    'values': list(period_values.values()),
                    'periods_used': len(period_values)
                }

            if crit_type == 'calculated_peer':
                numerator, denominator = criterion['codes']
                series = self.calculate_ratio_series(insurance_df, numerator, denominator, periods=12)
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

    def score(self, symbol, insurance_df, all_companies_data):
        results = {}
        points = 0
        
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(insurance_df, crit, all_companies_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': float(min(points, 5)),
            'details': results
        }

# %%
try:
    val_scorer = RelativeValuationScorer()
    val_scores = {}

    for symbol, insurance_df in insurance_ratios_data.items():
        val_scores[symbol] = val_scorer.score(symbol, insurance_df, insurance_ratios_data)    
    
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
    print(f"[Fund_INSURANCE] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()