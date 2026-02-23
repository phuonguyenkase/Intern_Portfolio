# %%
import requests
import pandas as pd
import numpy as np
import os
import TechAna_DRAFT as TechAna

all_stocks = []
materials_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
materials_ratios_data = {}
as_of_date = os.getenv("Fund_AsOfDate")

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    filtered_stocks = [
        s for s in all_stocks
        if s.get('industry_lv1') == 'Basic Materials']
    materials_symbols = [s['symbol'] for s in filtered_stocks]

    min_price = 10000
    price_date = getattr(TechAna, 'END_DATE', None)
    if price_date and materials_symbols:
        price_params = {
            "symbols": ",".join(materials_symbols),
            "start_date": price_date,
            "end_date": price_date
        }
        r = requests.get(
            "http://192.168.8.190:8000/MKD/stock_daily",
            params=price_params,
            headers={"accept": "application/json"},
            timeout=30
        )
        r.raise_for_status()
        price_payload = r.json()
        price_items = []
        if isinstance(price_payload, dict):
            for sym, rows in price_payload.items():
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            row = {**row, "symbol": row.get("symbol", sym)}
                            price_items.append(row)
                elif isinstance(rows, dict):
                    row = {**rows, "symbol": rows.get("symbol", sym)}
                    price_items.append(row)
        elif isinstance(price_payload, list):
            price_items = price_payload

        price_map = {}
        for row in price_items:
            if not isinstance(row, dict):
                continue
            sym = row.get('symbol')
            if not sym:
                continue
            price = row.get('adj_close')
            if price is None:
                price = row.get('close')
            if price is None:
                continue
            try:
                price_map[sym] = float(price)
            except (TypeError, ValueError):
                continue

        if price_map:
            min_price_symbols = {s for s, v in price_map.items() if v >= min_price}
            filtered_stocks = [s for s in filtered_stocks if s.get('symbol') in min_price_symbols]
            materials_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(materials_symbols),
        "period": "quarterly",
        "num_periods": 12,
        "language": "en"
    }
    r = requests.get(url, params=params, headers={"accept": "application/json"})
    r.raise_for_status()
    all_ratios_data.extend(r.json())

    url_daily = "http://192.168.8.190:8000/MKD/stock-ratios-daily"
    daily_params = {
        "symbols": ",".join(materials_symbols),
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
        materials_symbols,
        key=lambda s: float(matching_volume_12_2025.get(s, float('-inf'))),
        reverse=True
    )
    top_75_stocks = filtered_stocks[:75]
    materials_symbols = top_75_stocks

    df_ratios = pd.DataFrame(all_ratios_data)

    if not as_of_date:
        try:
            end_dt = pd.to_datetime(getattr(TechAna, 'END_DATE', None), errors = "coerce")
            if pd.notna(end_dt):
                prev_q_end = (end_dt.to_period('Q') - 1).end_time
                as_of_date = prev_q_end.strftime("%Y-%m-%d")
        except Exception:
            pass

    if 'symbol' in df_ratios.columns:
        df_filtered = df_ratios[df_ratios['symbol'].isin(materials_symbols)]
        for symbol, symbol_data in df_filtered.groupby('symbol'):
            materials_ratios_data[symbol] = symbol_data.copy()
except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_Materials] Warning: failed to fetch materials ratios: {e}")

# %%
class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Current ratio', 'code': 'ryq3', 'type': 'absolute', 'rule': 'value ≥ 1.0', 'threshold': 1.0, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'Quick ratio', 'code': 'ryq2', 'type': 'absolute', 'rule': 'value ≥ 0.8', 'threshold': 0.8, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Cash ratio', 'code': 'ryq1', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.40', 'cut': 0.40, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'DSO control', 'code': 'ryq16', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'Inventory control (DIO)', 'code': 'ryq18', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.60', 'cut': 0.60, 'direction': 'lower', 'must_have': False},
            {'id': 6, 'name': 'Payable efficiency (DPO)', 'code': 'ryq20', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.40', 'cut': 0.40, 'direction': 'higher', 'must_have': False},
            {'id': 7, 'name': 'Cash Conversion Cycle (CCC)', 'type': 'calculated_peer_percentile', 'formula': 'DSO + DIO - DPO', 'codes': ['ryq16', 'ryq18', 'ryq20'], 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
        ]
    
    def get_metric_series(self, materials_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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

    def get_metric_period_dict(self, materials_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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

    def calculate_ratio_series(self, materials_df, numerator_code, denominator_code, periods=12):
        """Calculate ratio series for last N periods: numerator / denominator"""
        numerator_map = self.get_metric_period_dict(materials_df, numerator_code)
        denominator_map = self.get_metric_period_dict(materials_df, denominator_code)

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
    
    def evaluate_criterion(self, materials_df, criterion, all_materials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(materials_df, criterion['code'], periods=12)
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

            if crit_type in {'peer_percentile', 'calculated_peer_percentile'}:
                if crit_type == 'calculated_peer_percentile':
                    # For CCC: calculate DSO, DIO, DPO and combine
                    dso_series = self.get_metric_series(materials_df, 'ryq16', periods=12)
                    dio_series = self.get_metric_series(materials_df, 'ryq18', periods=12)
                    dpo_series = self.get_metric_series(materials_df, 'ryq20', periods=12)
                    
                    if not dso_series or not dio_series or not dpo_series:
                        return {'pass': False, 'reason': 'Missing DSO/DIO/DPO data', 'values': []}
                    
                    dso_dict = {p: v for p, v in dso_series}
                    dio_dict = {p: v for p, v in dio_series}
                    dpo_dict = {p: v for p, v in dpo_series}
                    
                    common_periods = set(dso_dict.keys()) & set(dio_dict.keys()) & set(dpo_dict.keys())
                    if not common_periods:
                        return {'pass': False, 'reason': 'No common periods', 'values': []}
                    
                    period_values = {p: dso_dict[p] + dio_dict[p] - dpo_dict[p] for p in common_periods}
                else:
                    series = self.get_metric_series(materials_df, criterion['code'], periods=12)
                    if not series:
                        return {'pass': False, 'reason': 'No period values available', 'values': []}
                    period_values = {p: v for p, v in series}

                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_materials_data.values()]

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
    
    def score_materials(self, symbol, materials_df, all_materials_data, peer_data_cache=None):
        """Calculate liquidity score for a materials (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
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

peer_data_cache = {}
metric_codes = ['ryq1', 'ryq16', 'ryq18', 'ryq20']
for code in metric_codes:
    peer_data_cache[code] = [scorer.get_metric_period_dict(df, code) for df in materials_ratios_data.values()]

liq_scores = {}

for idx, (symbol, materials_df) in enumerate(materials_ratios_data.items()):
    score_result = scorer.score_materials(symbol, materials_df, materials_ratios_data, peer_data_cache)
    liq_scores[symbol] = score_result

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Gross margin', 'code': 'ryq25', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 2, 'name': 'EBIT margin floor', 'code': 'ryq27', 'type': 'absolute', 'rule': 'value ≥ 0.00', 'threshold': 0.00, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Net margin', 'code': 'ryq29', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'ROIC quality', 'code': 'ryq76', 'type': 'absolute', 'rule': 'value ≥ 0.05', 'threshold': 0.05, 'direction': 'higher', 'must_have': True},
            {'id': 5, 'name': 'ROE', 'code': 'ryq12', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.40', 'cut': 0.40, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'Revenue growth', 'code': 'ryq34', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]
    
    def get_metric_series(self, materials_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, materials_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, materials_df, criterion, all_materials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                series = self.get_metric_series(materials_df, criterion['code'], periods=12)
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
                series = self.get_metric_series(materials_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_materials_data.values()]

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
    
    def score_materials(self, symbol, materials_df, all_materials_data, peer_data_cache=None):
        """Calculate profitability score for a materials (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': min(points, 5),  # Cap at 5
            'details': results
        }
    
# %%
prof_scorer = ProfitabilityScorer()

prof_peer_data_cache = {}
prof_metric_codes = ['ryq25', 'ryq29', 'ryq12', 'ryq34']
for code in prof_metric_codes:
    prof_peer_data_cache[code] = [prof_scorer.get_metric_period_dict(df, code) for df in materials_ratios_data.values()]

prof_scores = {}

for idx, (symbol, materials_df) in enumerate(materials_ratios_data.items()):
    score_result = prof_scorer.score_materials(symbol, materials_df, materials_ratios_data, prof_peer_data_cache)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Debt/Equity cap', 'code': 'ryq10', 'type': 'absolute', 'rule': 'value ≤ 2.5', 'threshold': 2.5, 'direction': 'lower', 'must_have': True},
            {'id': 2, 'name': 'Borrowings/Equity cap', 'code': 'ryq6', 'type': 'absolute', 'rule': 'value ≤ 2.5', 'threshold': 2.5, 'direction': 'lower', 'must_have': True},
            {'id': 3, 'name': 'Interest coverage', 'code': 'ryq77', 'type': 'absolute', 'rule': 'value ≥ 2.0', 'threshold': 2.0, 'direction': 'higher', 'must_have': True},
            {'id': 4, 'name': 'Financial leverage', 'code': 'ryq71', 'type': 'peer_percentile', 'rule': 'pct ≤ 0.50', 'cut': 0.50, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'ROA stability', 'code': 'ryq14', 'type': 'trend_vol', 'rule': 'std_12q ≤ peer_median_std', 'window': 12, 'direction': 'lower', 'must_have': False},
            {'id': 6, 'name': 'Fixed asset turnover', 'code': 'ryq91', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]
    
    def get_metric_series(self, materials_df, code, periods=12):
        """Extract last N period values for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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
    
    def get_metric_period_dict(self, materials_df, code):
        """Map period_key -> value for a metric code"""
        metric_data = materials_df[materials_df['code'] == code].copy()
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
    
    def evaluate_criterion(self, materials_df, criterion, all_materials_data, peer_data_cache=None):
        """Evaluate a single criterion over the last 12 quarters (with pass rate ≥ 50%)"""
        try:
            crit_type = criterion['type']
            
            # Absolute criterion
            if crit_type == 'absolute':
                series = self.get_metric_series(materials_df, criterion['code'], periods=12)
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
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                series = self.get_metric_series(materials_df, criterion['code'], periods=12)
                if not series:
                    return {'pass': False, 'reason': 'No period values available', 'values': []}

                period_values = {p: v for p, v in series}
                code = criterion['code']
                if peer_data_cache and code in peer_data_cache:
                    peer_maps = peer_data_cache[code]
                else:
                    peer_maps = [self.get_metric_period_dict(df, code) for df in all_materials_data.values()]

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
            
            # Trend volatility criterion (ROA stability)
            elif crit_type == 'trend_vol':
                code = criterion['code']
                series_list = self.get_metric_series(materials_df, code, periods=criterion['window'])
                
                if len(series_list) < criterion['window']:
                    return {'pass': False, 'reason': f'Insufficient data for {code}', 'value': None}
                
                values = [v for _, v in series_list]
                std_val = pd.Series(values).std()
                
                # Compare to peer median std
                peer_stds = []
                for df in all_materials_data.values():
                    peer_series_list = self.get_metric_series(df, code, periods=criterion['window'])
                    if len(peer_series_list) >= criterion['window']:
                        peer_values = [v for _, v in peer_series_list]
                        peer_std = pd.Series(peer_values).std()
                        peer_stds.append(peer_std)
                
                if len(peer_stds) > 0:
                    peer_median_std = pd.Series(peer_stds).median()
                    pass_check = std_val <= peer_median_std
                    return {'pass': pass_check, 'std': std_val, 'peer_median_std': peer_median_std}
                else:
                    return {'pass': False, 'reason': 'No peer data for volatility comparison'}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_materials(self, symbol, materials_df, all_materials_data, peer_data_cache=None):
        """Calculate solvency score for a materials company (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(materials_df, crit, all_materials_data, peer_data_cache)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)  # Distribute 3 points among optional
        
        return {
            'symbol': symbol,
            'score': min(points, 5),  # Cap at 5
            'details': results
        }

# %%
solv_scorer = SolvencyScorer()

solv_peer_data_cache = {}
solv_metric_codes = ['ryq71', 'ryq91']
for code in solv_metric_codes:
    solv_peer_data_cache[code] = [solv_scorer.get_metric_period_dict(df, code) for df in materials_ratios_data.values()]

solv_scores = {}

for idx, (symbol, materials_df) in enumerate(materials_ratios_data.items()):
    score_result = solv_scorer.score_materials(symbol, materials_df, materials_ratios_data, solv_peer_data_cache)
    solv_scores[symbol] = score_result

# %%
class RelativeValuationScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE quality gate', 'code': 'ryq12', 'type': 'absolute', 'rule': 'ROE ≥ 8%', 'threshold': 0.08, 'direction': 'higher', 'must_have': True},
            {'id': 2, 'name': 'P/B cheap vs peer', 'code': 'ryd25', 'type': 'peer_percentile', 'rule': 'pct <= 0.40', 'cut': 0.40,'direction': 'lower', 'must_have': False},
            {'id': 3, 'name': 'EV/EBITDA cheap vs peer', 'code': 'ryd30', 'type': 'peer_percentile', 'rule': 'pct <= 0.30', 'cut': 0.30, 'direction': 'lower', 'must_have': False},
            {'id': 4, 'name': 'P/B below own history (avg_5y excl latest)', 'code': 'ryd25', 'type': 'history_vs_avg_exlatest', 'rule': 'value ≤ avg_5y(excl latest)', 'window_years': 5, 'direction': 'lower', 'must_have': False},
            {'id': 5, 'name': 'P/S below own history (avg_5y excl latest)', 'code': 'ryd26', 'type': 'history_vs_avg_exlatest', 'rule': 'value ≤ avg_5y(excl latest)', 'window_years': 5, 'direction': 'lower', 'must_have': False},
            {'id': 6, 'name': 'Justified PB proxy: (P/B)/ROE ≤ peer_median', 'pb_code': 'ryd25', 'roe_code': 'ryq12', 'type': 'ratio_threshold', 'rule': '(P/B)/ROE ≤ peer_median', 'fallback_abs': 20.0, 'direction': 'lower', 'must_have': False},
        ]

    def get_metric_series(self, materials_df, code):
        metric = materials_df[materials_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def get_metric_value(self, materials_df, code):
        s = self.get_metric_series(materials_df, code)
        return None if s.empty else s.iloc[-1]

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, materials_df, criterion, all_materials_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(materials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(materials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_materials_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(materials_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(materials_df, code)
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
                pb = self.get_metric_value(materials_df, criterion['pb_code'])
                roe = self.get_metric_value(materials_df, criterion['roe_code'])

                if pb is None or roe is None or roe == 0:
                    return {'pass': False, 'reason': 'Missing PB or ROE or ROE=0', 'pb': pb, 'roe': roe}

                if roe < 0.10:
                    return {'pass': False, 'reason': 'ROE below gate (<10%)', 'pb': pb, 'roe': roe, 'ratio': pb/roe}

                ratio = pb / roe

                peer_ratios = []
                for df in all_materials_data.values():
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

    def score_materials(self, symbol, materials_df, all_materials_data):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(materials_df, crit, all_materials_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(materials_df, crit, all_materials_data)
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

    for symbol, materials_df in materials_ratios_data.items():
        val_scores[symbol] = val_scorer.score_materials(symbol, materials_df, materials_ratios_data)    
    
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

except Exception as e:
    print(f"[Fund_MATERIALS] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()