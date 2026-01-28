# %%
import requests
import pandas as pd

all_stocks = []
financials_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
financials_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all financials symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    financials_symbols = [s['symbol'] for s in all_stocks if s.get('industry_lv1') == 'Financials' and s.get('industry_lv2')=='Financial Services']
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

    # Organize by financials (sorted by date/year-quarter)
    if 'symbol' in df_ratios.columns:
        for symbol in sorted(df_ratios['symbol'].unique()):
            df_sym = df_ratios[df_ratios['symbol'] == symbol].copy()
            if 'date' in df_sym.columns:
                df_sym = df_sym.sort_values('date')
            elif 'year' in df_sym.columns and 'quarter' in df_sym.columns:
                df_sym = df_sym.sort_values(['year', 'quarter'])
            financials_ratios_data[symbol] = df_sym

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_FINANCIALS] Warning: failed to fetch financials ratios: {e}")

# %%
print(financials_symbols)

# %%
CODE_MAPPING = {
    'ryq3': 'Current_Ratio',
    'ryq2': 'Quick_Ratio',
    'ryq1': 'Cash_Ratio',
    'ryq16': 'DSO',
    'ryq71': 'Leverage',
    'ryq20': 'Working_Capital_Efficiency'
}

class LiquidityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Current ratio safety', 'code': 'ryq3', 'type': 'absolute', 'direction': 'higher_better', 'threshold': 1.0, 'must_have': True},
            {'id': 2, 'name': 'Quick ratio safety', 'code': 'ryq2', 'type': 'absolute', 'direction': 'higher_better', 'threshold': 0.8, 'must_have': True},
            {'id': 3, 'name': 'Cash ratio buffer', 'code': 'ryq1', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.50, 'must_have': False},
            {'id': 4, 'name': 'DSO control', 'code': 'ryq16', 'type': 'peer_percentile', 'direction': 'lower_better', 'threshold': 0.60, 'must_have': False},
            {'id': 5, 'name': 'Leverage discipline', 'code': 'ryq71', 'type': 'peer_percentile', 'direction': 'lower_better', 'threshold': 0.50, 'must_have': False},
            {'id': 6, 'name': 'Working-cap efficiency', 'code': 'ryq20', 'type': 'peer_percentile', 'direction': 'lower_better', 'threshold': 0.60, 'must_have': False},
        ]
    
    def get_metric_series(self, financial_df, code):
        """Extract series of values for a metric code (last 12 quarters)"""
        metric_data = financial_df[financial_df['code'] == code]
        if len(metric_data) > 0:
            values = pd.to_numeric(metric_data['value'], errors='coerce').dropna()
            if len(values) > 0:
                return values.tail(12)
        return pd.Series(dtype=float)
    
    def get_metric_value(self, financial_df, code):
        """Extract the latest value for a metric code"""
        series = self.get_metric_series(financial_df, code)
        return series.iloc[-1] if len(series) > 0 else None
    
    def evaluate_criterion(self, financial_df, criterion, all_financials_data):
        """Evaluate a single criterion for a financial"""
        try:
            code = criterion['code']
            crit_type = criterion['type']
            direction = criterion['direction']
            
            latest_value = self.get_metric_value(financial_df, code)
            
            if latest_value is None:
                return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
            
            # Absolute criterion
            if crit_type == 'absolute':
                if direction == 'higher_better':
                    pass_check = latest_value >= criterion['threshold']
                else:
                    pass_check = latest_value <= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion['threshold']}
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                peer_values = [self.get_metric_value(df, code) for df in all_financials_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if direction == 'lower_better':
                        pct = sum(1 for v in peer_values if v >= latest_value) / len(peer_values)
                        pass_check = pct <= criterion['threshold']
                    else:
                        pct = sum(1 for v in peer_values if v <= latest_value) / len(peer_values)
                        pass_check = pct >= criterion['threshold']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'threshold': criterion['threshold']}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_financial(self, symbol, financial_df, all_financials_data):
        """Calculate liquidity score for a financial (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results}
    
# %%
scorer = LiquidityScorer()
liq_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = scorer.score_financial(symbol, financials_df, financials_ratios_data)
    liq_scores[symbol] = score_result

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'ROE minimum', 'code': 'ryq12', 'type': 'absolute', 'direction': 'higher_better', 'threshold': 0.10, 'must_have': True},
            {'id': 2, 'name': 'Net margin minimum', 'code': 'ryq29', 'type': 'absolute', 'direction': 'higher_better', 'threshold': 0.10, 'must_have': True},
            {'id': 3, 'name': 'ROIC quality', 'code': 'ryq76', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.60, 'must_have': False},
            {'id': 4, 'name': 'EBIT margin', 'code': 'ryq27', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.60, 'must_have': False},
            {'id': 5, 'name': 'Profit growth YoY', 'code': 'ryq39', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.50, 'must_have': False},
            {'id': 6, 'name': 'Revenue growth YoY', 'code': 'ryq34', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.50, 'must_have': False},
        ]
    
    def get_metric_series(self, financial_df, code):
        """Extract series of values for a metric code (last 12 quarters)"""
        metric_data = financial_df[financial_df['code'] == code]
        if len(metric_data) > 0:
            values = metric_data['value'].dropna()
            if len(values) > 0:
                return values.tail(12)
        return pd.Series(dtype=float)
    
    def get_metric_value(self, financial_df, code):
        """Extract the latest value for a metric code"""
        series = self.get_metric_series(financial_df, code)
        return series.iloc[-1] if len(series) > 0 else None
    
    def evaluate_criterion(self, financial_df, criterion, all_financials_data):
        """Evaluate a single criterion for a financial"""
        try:
            code = criterion['code']
            crit_type = criterion['type']
            direction = criterion['direction']
            
            latest_value = self.get_metric_value(financial_df, code)
            
            if latest_value is None:
                return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
            
            # Absolute criterion
            if crit_type == 'absolute':
                if direction == 'higher_better':
                    pass_check = latest_value >= criterion['threshold']
                else:
                    pass_check = latest_value <= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion['threshold']}
            
            # Peer percentile criterion
            elif crit_type == 'peer_percentile':
                peer_values = [self.get_metric_value(df, code) for df in all_financials_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if direction == 'lower_better':
                        pct = sum(1 for v in peer_values if v >= latest_value) / len(peer_values)
                        pass_check = pct <= criterion['threshold']
                    else:
                        pct = sum(1 for v in peer_values if v <= latest_value) / len(peer_values)
                        pass_check = pct >= criterion['threshold']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'threshold': criterion['threshold']}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_financial(self, symbol, financial_df, all_financials_data):
        """Calculate profitability score for a financial (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data)
            results[f"C{crit['id']}"] = result
            if result['pass']:
                points += 3 / len(optional_criteria)
        
        return {
            'symbol': symbol,
            'score': min(points, 5),
            'details': results}

# %%
prof_scorer = ProfitabilityScorer()
prof_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = prof_scorer.score_financial(symbol, financials_df, financials_ratios_data)
    prof_scores[symbol] = score_result

# %%
class SolvencyScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Debt/Equity cap', 'code': 'ryq10', 'type': 'absolute', 'direction': 'lower_better', 'threshold': 2.5, 'must_have': True},
            {'id': 2, 'name': 'Interest coverage safety', 'code': 'ryq77', 'type': 'absolute', 'direction': 'higher_better', 'threshold': 2.0, 'must_have': True},
            {'id': 3, 'name': 'Borrowings/Equity', 'code': 'ryq6', 'type': 'peer_percentile', 'direction': 'lower_better', 'threshold': 0.50, 'must_have': False},
            {'id': 4, 'name': 'Financial leverage', 'code': 'ryq71', 'type': 'peer_percentile', 'direction': 'lower_better', 'threshold': 0.50, 'must_have': False},
            {'id': 5, 'name': 'ROA floor', 'code': 'ryq14', 'type': 'peer_or_abs', 'direction': 'higher_better', 'abs_threshold': 0.02, 'must_have': False},
            {'id': 6, 'name': 'Asset turnover', 'code': 'ryq31', 'type': 'peer_percentile', 'direction': 'higher_better', 'threshold': 0.50, 'must_have': False},
        ]

    def get_metric_series(self, financial_df, code):
        metric_data = financial_df[financial_df['code'] == code]
        if len(metric_data) > 0:
            values = metric_data['value'].dropna()
            if len(values) > 0:
                return values.tail(12)
        return pd.Series(dtype=float)

    def get_metric_value(self, financial_df, code):
        series = self.get_metric_series(financial_df, code)
        return series.iloc[-1] if len(series) > 0 else None

    def evaluate_criterion(self, financial_df, criterion, all_financials_data):
        try:
            code = criterion['code']
            crit_type = criterion['type']
            direction = criterion['direction']

            latest_value = self.get_metric_value(financial_df, code)

            if latest_value is None:
                return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

            if crit_type == 'absolute':
                if direction == 'higher_better':
                    pass_check = latest_value >= criterion['threshold']
                else:
                    pass_check = latest_value <= criterion['threshold']
                return {'pass': pass_check, 'value': latest_value, 'threshold': criterion.get('threshold')}

            elif crit_type == 'peer_percentile':
                peer_values = [self.get_metric_value(df, code) for df in all_financials_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if direction == 'lower_better':
                        pct = sum(1 for v in peer_values if v >= latest_value) / len(peer_values)
                        pass_check = pct <= criterion['threshold']
                    else:
                        pct = sum(1 for v in peer_values if v <= latest_value) / len(peer_values)
                        pass_check = pct >= criterion['threshold']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'threshold': criterion['threshold']}

            elif crit_type == 'peer_or_abs':
                peer_values = [self.get_metric_value(df, code) for df in all_financials_data.values()]
                peer_values = [v for v in peer_values if v is not None]

                abs_pass = latest_value >= criterion.get('abs_threshold', 0)
                peer_pass = False
                peer_median = None
                if len(peer_values) > 0:
                    sorted_peers = sorted(peer_values)
                    peer_median = sorted_peers[len(sorted_peers) // 2]
                    peer_pass = latest_value >= peer_median if direction == 'higher_better' else latest_value <= peer_median

                pass_check = abs_pass or peer_pass
                return {
                    'pass': pass_check,
                    'value': latest_value,
                    'abs_threshold': criterion.get('abs_threshold'),
                    'peer_median': peer_median,
                    'abs_pass': abs_pass,
                    'peer_pass': peer_pass,
                }

            return {'pass': False, 'reason': 'Unknown criterion type'}

        except Exception as e:
            return {'pass': False, 'reason': str(e)}

    def score_financial(self, symbol, financial_df, all_financials_data):
        results = {}
        points = 0

        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(financial_df, crit, all_financials_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result

        if all(must_have_results):
            points += 2

        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financial_df, crit, all_financials_data)
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
solv_scores = {}

for idx, (symbol, financials_df) in enumerate(financials_ratios_data.items()):
    score_result = solv_scorer.score_financial(symbol, financials_df, financials_ratios_data)
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

    def evaluate_criterion(self, financials_df, criterion, all_financials_data):
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

    def score_financials(self, symbol, financials_df, all_financials_data):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(financials_df, crit, all_financials_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(financials_df, crit, all_financials_data)
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
        val_scores[symbol] = val_scorer.score_financials(symbol, financials_df, financials_ratios_data)    
    
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

    print(f"Financials Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_FINANCIALS] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()