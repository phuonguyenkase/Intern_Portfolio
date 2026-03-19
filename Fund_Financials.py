# %%
import requests
import pandas as pd
import os
import TechAna_DRAFT as TechAna
from Filtering_Stock import filter_stocks_by_price_and_liquidity

START_DATE = TechAna.START_DATE
END_DATE = TechAna.END_DATE

all_stocks = []
financials_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
financials_ratios_data = {}
as_of_date = os.getenv("Fund_AsOf_Date")
as_of_ts = None

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all financials symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()

    filtered_stocks = [
        s for s in all_stocks
        if s.get('industry_lv1') == 'Financials' and s.get('industry_lv2') == 'Financial Services'
    ]
    filtered_stocks = filter_stocks_by_price_and_liquidity(
        filtered_stocks,
        min_price=10000,
        min_volume_threshold=20000,
        min_pass_rate=0.50,
        start_date=getattr(TechAna, 'START_DATE', None),
        end_date=getattr(TechAna, 'END_DATE', None),
    )
    financials_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(financials_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }
    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

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

    if 'symbol' in df_ratios.columns:
        df_filtered = df_ratios[df_ratios['symbol'].isin(financials_symbols)]
        for symbol, symbol_data in df_filtered.groupby('symbol'):
            financials_ratios_data[symbol] = symbol_data.copy()
except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_Financials] Warning: failed to fetch financials ratios: {e}")
    
# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Current ratio safety', 'code': 'ryq3', 'type': 'absolute', 'direction': 'higher', 'threshold': 1.0, 'must_have': True},
            {'id': 2, 'name': 'Quick ratio safety', 'code': 'ryq2', 'type': 'absolute', 'direction': 'higher', 'threshold': 0.8, 'must_have': True},
            {'id': 3, 'name': 'Cash ratio buffer', 'code': 'ryq1', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.50, 'must_have': False},
            {'id': 4, 'name': 'DSO control', 'code': 'ryq16', 'type': 'peer_percentile', 'direction': 'lower', 'cut': 0.60, 'must_have': False},
            {'id': 5, 'name': 'Leverage discipline', 'code': 'ryq71', 'type': 'peer_percentile', 'direction': 'lower', 'cut': 0.50, 'must_have': False},
            {'id': 6, 'name': 'Working-cap efficiency', 'code': 'ryq20', 'type': 'peer_percentile', 'direction': 'lower', 'cut': 0.60, 'must_have': False},
        ]
    
    def get_metric_series(self, financial_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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

    def get_metric_period_dict(self, financial_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, financial_df, criterion, all_financials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_financials_data.values()]

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
    
    def score_financial(self, symbol, financial_df, all_financials_data, peer_data_cache=None):
        """Calculate liquidity score for a financial (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results}
    
# %%
scorer = LiquidityScorer()

liq_peer_data_cache = {}
liq_metric_codes = ['ryq1', 'ryq16', 'ryq71', 'ryq20']
for code in liq_metric_codes:
    liq_peer_data_cache[code] = [scorer.get_metric_period_dict(df, code) for df in financials_ratios_data.values()]

liq_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = scorer.score_financial(symbol, financials_df, financials_ratios_data, liq_peer_data_cache)
    liq_scores[symbol] = score_result

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE minimum', 'code': 'ryq12', 'type': 'absolute', 'direction': 'higher', 'threshold': 0.10, 'must_have': True},
            {'id': 2, 'name': 'Net margin minimum', 'code': 'ryq29', 'type': 'absolute', 'direction': 'higher', 'threshold': 0.10, 'must_have': True},
            {'id': 3, 'name': 'ROIC quality', 'code': 'ryq76', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.60, 'must_have': False},
            {'id': 4, 'name': 'EBIT margin', 'code': 'ryq27', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.60, 'must_have': False},
            {'id': 5, 'name': 'Profit growth YoY', 'code': 'ryq39', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.50, 'must_have': False},
            {'id': 6, 'name': 'Revenue growth YoY', 'code': 'ryq34', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.50, 'must_have': False},
        ]
    
    def get_metric_series(self, financial_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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

    def get_metric_period_dict(self, financial_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, financial_df, criterion, all_financials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_financials_data.values()]

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
    
    def score_financial(self, symbol, financial_df, all_financials_data, peer_data_cache=None):
        """Calculate profitability score for a financial (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results}

# %%
prof_scorer = ProfitabilityScorer()

prof_peer_data_cache = {}
prof_metric_codes = ['ryq76', 'ryq27', 'ryq39', 'ryq34']
for code in prof_metric_codes:
    prof_peer_data_cache[code] = [prof_scorer.get_metric_period_dict(df, code) for df in financials_ratios_data.values()]

prof_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = prof_scorer.score_financial(symbol, financials_df, financials_ratios_data, prof_peer_data_cache)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Debt/Equity cap', 'code': 'ryq10', 'type': 'absolute', 'direction': 'lower', 'threshold': 2.5, 'must_have': True},
            {'id': 2, 'name': 'Interest coverage safety', 'code': 'ryq77', 'type': 'absolute', 'direction': 'higher', 'threshold': 2.0, 'must_have': True},
            {'id': 3, 'name': 'Borrowings/Equity', 'code': 'ryq6', 'type': 'peer_percentile', 'direction': 'lower', 'cut': 0.50, 'must_have': False},
            {'id': 4, 'name': 'Financial leverage', 'code': 'ryq71', 'type': 'peer_percentile', 'direction': 'lower', 'cut': 0.50, 'must_have': False},
            {'id': 5, 'name': 'ROA floor', 'code': 'ryq14', 'type': 'absolute', 'direction': 'higher', 'threshold': 0.02, 'must_have': False},
            {'id': 6, 'name': 'Asset turnover', 'code': 'ryq31', 'type': 'peer_percentile', 'direction': 'higher', 'cut': 0.50, 'must_have': False},
        ]

    def get_metric_series(self, financial_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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

    def get_metric_period_dict(self, financial_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = financial_df[financial_df['code'] == code].copy()
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

    def evaluate_criterion(self, financial_df, criterion, all_financials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
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

            elif crit_type == 'peer_percentile':
                series = self.get_metric_series(financial_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_financials_data.values()]

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

    def score_financial(self, symbol, financial_df, all_financials_data, peer_data_cache=None):
        results = {}
        points = 0

        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result

        if all(must_have_results):
            points += 2

        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)

        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results
        }

# %%
solv_scorer = SolvencyScorer()

solv_peer_data_cache = {}
solv_metric_codes = ['ryq6', 'ryq71', 'ryq31']
for code in solv_metric_codes:
    solv_peer_data_cache[code] = [solv_scorer.get_metric_period_dict(df, code) for df in financials_ratios_data.values()]

solv_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = solv_scorer.score_financial(symbol, financials_df, financials_ratios_data, solv_peer_data_cache)
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

    def get_metric_series(self, financials_df, code):
        metric = financials_df[financials_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def get_metric_value(self, financials_df, code):
        s = self.get_metric_series(financials_df, code)
        return None if s.empty else s.iloc[-1]

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, financials_df, criterion, all_financials_data, peer_data_cache=None):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(financials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(financials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_financials_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(financials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(financials_df, code)
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
                pb = self.get_metric_value(financials_df, criterion['pb_code'])
                roe = self.get_metric_value(financials_df, criterion['roe_code'])

                if pb is None or roe is None or roe == 0:
                    return {'pass': False, 'reason': 'Missing PB or ROE or ROE=0', 'pb': pb, 'roe': roe}

                if roe < 0.10:
                    return {'pass': False, 'reason': 'ROE below gate (<10%)', 'pb': pb, 'roe': roe, 'ratio': pb/roe}

                ratio = pb / roe

                peer_ratios = []
                for df in all_financials_data.values():
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

    def score_financials(self, symbol, financials_df, all_financials_data, peer_data_cache=None):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(financials_df, crit, all_financials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financials_df, crit, all_financials_data, peer_data_cache)
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

    for symbol, financials_df in financials_ratios_data.items():
        val_scores[symbol] = val_scorer.score_financials(symbol, financials_df, financials_ratios_data, None)    
    
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
    print(f"[Fund_FINANCIALS] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()
# %%
