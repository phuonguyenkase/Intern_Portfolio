# %%
import requests
import pandas as pd

consumer = ['Consumer Goods', 'Consumer Services']
all_stocks = []
consumers_symbols = []
all_ratios_data = []
df_ratios = pd.DataFrame()
consumers_ratios_data = {}

# Make exports safe even if fetch or scoring fails
combined_scores_draft = pd.DataFrame()

try:
    # Get all symbols
    r = requests.get("http://192.168.8.190:8000/MKD/stock_info")
    r.raise_for_status()
    all_stocks = r.json()
    filtered_stocks = [
        s for s in all_stocks
        if s.get('industry_lv1') in consumer]
    consumers_symbols = [s['symbol'] for s in filtered_stocks]

    url = "http://192.168.8.190:8000/MKD/stock-ratios"
    params = {
        "symbols": ",".join(consumers_symbols),
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
    top_100_stocks = filtered_stocks[:100]
    consumers_symbols = [s['symbol'] for s in top_100_stocks]

    # Convert to DataFrame
    df_ratios = pd.DataFrame(all_ratios_data)

    if 'symbol' in df_ratios.columns:
        for symbol in consumers_symbols:
            symbol_data = df_ratios[df_ratios['symbol'] == symbol]
            if not symbol_data.empty:
                consumers_ratios_data[symbol] = symbol_data.copy()

except Exception as e:
    # Keep importable even if API is down
    print(f"[Fund_Consumers] Warning: failed to fetch consumers ratios: {e}")

# %%
print(consumers_symbols)

# %%
CODE_MAPPING = {
    'ryq3': 'Current Ratio',
    'ryq2': 'Quick Ratio',
    'ryq1': 'Cash Ratio',
    'ryq16': 'DSO Control',
    'ryq18': 'Inventory Control (DIO)',
    'ryq20': 'Payable efficiency (DPO)',
}

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
    
    def get_metric_value(self, consumers_df, code):
        """Extract the latest value for a metric code"""
        metric_data = consumers_df[consumers_df['code'] == code].copy()
        if len(metric_data) > 0:
            # Ensure latest means most recent quarter; sort then take last
            if 'date' in metric_data.columns:
                metric_data = metric_data.sort_values('date')
            elif {'year', 'quarter'}.issubset(metric_data.columns):
                metric_data = metric_data.sort_values(['year', 'quarter'])
            series = metric_data['value'].dropna()
            return series.iloc[-1] if len(series) > 0 else None
        return None
    
    def calculate_ccc(self, consumers_df):
        """Calculate Cash Conversion Cycle: DSO + DIO - DPO"""
        dso = self.get_metric_value(consumers_df, 'ryq16')
        dio = self.get_metric_value(consumers_df, 'ryq18')
        dpo = self.get_metric_value(consumers_df, 'ryq20')
        
        if dso is None or dio is None or dpo is None:
            return None
        
        return dso + dio - dpo
    
    def evaluate_criterion(self, consumers_df, criterion, all_companies_data):
        """Evaluate a single criterion for a consumers"""
        try:
            crit_type = criterion['type']
            
            # Calculated peer percentile (CCC)
            if crit_type == 'calculated_peer_percentile':
                latest_value = self.calculate_ccc(consumers_df)
                
                if latest_value is None:
                    return {'pass': False, 'reason': 'CCC calculation failed (missing DSO/DIO/DPO)', 'value': None}
                
                # Calculate CCC for all peers
                peer_values = []
                for df in all_companies_data.values():
                    peer_ccc = self.calculate_ccc(df)
                    if peer_ccc is not None:
                        peer_values.append(peer_ccc)
                
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = sum(1 for v in peer_values if latest_value <= v) / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = sum(1 for v in peer_values if latest_value >= v) / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
                else:
                    return {'pass': False, 'reason': 'No peer CCC values available', 'value': latest_value}
            
            code = criterion['code']
            latest_value = self.get_metric_value(consumers_df, code)
            
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
                peer_values = [self.get_metric_value(df, code) for df in all_companies_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = (latest_value <= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = (latest_value >= pd.Series(peer_values)).sum() / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_consumers(self, symbol, consumers_df, all_companies_data):
        """Calculate liquidity score for a consumers (0-5 points)"""
        results = {}
        points = 0
        max_points = 5
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
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

for idx, (symbol, consumers_df) in enumerate(consumers_ratios_data.items()):
    score_result = scorer.score_consumers(symbol, consumers_df, consumers_ratios_data)
    liq_scores[symbol] = score_result

# %%
class ProfitabilityScorer:
    def __init__(self):
        self.criteria = [
            {'id': 1, 'name': 'Gross margin', 'code': 'ryq25', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 2, 'name': 'EBIT margin floor', 'code': 'ryq27', 'type': 'absolute', 'rule': 'value ≥ 0.05', 'threshold': 0.05, 'direction': 'higher', 'must_have': True},
            {'id': 3, 'name': 'Net margin', 'code': 'ryq29', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 4, 'name': 'ROIC quality', 'code': 'ryq76', 'type': 'absolute', 'rule': 'value ≥ 0.08', 'threshold': 0.08, 'direction': 'higher', 'must_have': True},
            {'id': 5, 'name': 'ROE', 'code': 'ryq12', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
            {'id': 6, 'name': 'Revenue growth', 'code': 'ryq34', 'type': 'peer_percentile', 'rule': 'pct ≥ 0.50', 'cut': 0.50, 'direction': 'higher', 'must_have': False},
        ]
    
    def get_metric_value(self, consumers_df, code):
        """Extract the latest value for a metric code"""
        metric_data = consumers_df[consumers_df['code'] == code].copy()
        if len(metric_data) > 0:
            if 'date' in metric_data.columns:
                metric_data = metric_data.sort_values('date')
            elif {'year', 'quarter'}.issubset(metric_data.columns):
                metric_data = metric_data.sort_values(['year', 'quarter'])
            series = metric_data['value'].dropna()
            return series.iloc[-1] if len(series) > 0 else None
        return None
    
    def evaluate_criterion(self, consumers_df, criterion, all_companies_data):
        """Evaluate a single criterion for a consumers"""
        try:
            code = criterion['code']
            crit_type = criterion['type']
            
            latest_value = self.get_metric_value(consumers_df, code)
            
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
                peer_values = [self.get_metric_value(df, code) for df in all_companies_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = sum(1 for v in peer_values if latest_value <= v) / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = sum(1 for v in peer_values if latest_value >= v) / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_consumers(self, symbol, consumers_df, all_companies_data):
        """Calculate profitability score for a consumers (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
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
prof_scores = {}

for idx, (symbol, consumers_df) in enumerate(consumers_ratios_data.items()):
    score_result = prof_scorer.score_consumers(symbol, consumers_df, consumers_ratios_data)
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
    
    def get_metric_value(self, consumers_df, code):
        """Extract the latest value for a metric code"""
        metric_data = consumers_df[consumers_df['code'] == code].copy()
        if len(metric_data) > 0:
            if 'date' in metric_data.columns:
                metric_data = metric_data.sort_values('date')
            elif {'year', 'quarter'}.issubset(metric_data.columns):
                metric_data = metric_data.sort_values(['year', 'quarter'])
            series = metric_data['value'].dropna()
            return series.iloc[-1] if len(series) > 0 else None
        return None
    
    def get_metric_series(self, consumers_df, code):
        """Extract last 12 quarters for a metric code as a series"""
        metric_data = consumers_df[consumers_df['code'] == code].copy()
        if len(metric_data) > 0:
            if 'date' in metric_data.columns:
                metric_data = metric_data.sort_values('date')
            elif {'year', 'quarter'}.issubset(metric_data.columns):
                metric_data = metric_data.sort_values(['year', 'quarter'])
            series = metric_data['value'].dropna()
            return series.tail(12)
        return pd.Series()
    
    def evaluate_criterion(self, consumers_df, criterion, all_companies_data):
        """Evaluate a single criterion for a consumers company"""
        try:
            crit_type = criterion['type']
            
            # Absolute criterion
            if crit_type == 'absolute':
                code = criterion['code']
                latest_value = self.get_metric_value(consumers_df, code)
                
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
                latest_value = self.get_metric_value(consumers_df, code)
                
                if latest_value is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}
                
                peer_values = [self.get_metric_value(df, code) for df in all_companies_data.values()]
                peer_values = [v for v in peer_values if v is not None]
                if len(peer_values) > 0:
                    if criterion['direction'] == 'lower':
                        pct = sum(1 for v in peer_values if latest_value <= v) / len(peer_values)
                        pass_check = pct <= criterion['cut']
                    else:
                        pct = sum(1 for v in peer_values if latest_value >= v) / len(peer_values)
                        pass_check = pct >= criterion['cut']
                    return {'pass': pass_check, 'value': latest_value, 'percentile': pct, 'cut': criterion['cut']}
            
            # Trend volatility criterion (ROA stability)
            elif crit_type == 'trend_vol':
                code = criterion['code']
                metric_series = self.get_metric_series(consumers_df, code)
                
                if len(metric_series) < criterion['window']:
                    return {'pass': False, 'reason': f'Insufficient data for {code}', 'value': None}
                
                # Get recent window
                recent_data = metric_series.tail(criterion['window'])
                std_val = recent_data.std()
                
                # Compare to peer median std
                peer_stds = []
                for df in all_companies_data.values():
                    peer_series = self.get_metric_series(df, code)
                    if len(peer_series) >= criterion['window']:
                        peer_std = peer_series.tail(criterion['window']).std()
                        peer_stds.append(peer_std)
                
                if len(peer_stds) > 0:
                    sorted_stds = sorted(peer_stds)
                    peer_median_std = sorted_stds[len(sorted_stds) // 2]
                    pass_check = std_val <= peer_median_std
                    return {'pass': pass_check, 'std': std_val, 'peer_median_std': peer_median_std}
                else:
                    return {'pass': False, 'reason': 'No peer data for volatility comparison'}
            
            return {'pass': False, 'reason': 'Unknown criterion type'}
        
        except Exception as e:
            return {'pass': False, 'reason': str(e)}
    
    def score_consumers(self, symbol, consumers_df, all_companies_data):
        """Calculate solvency score for a consumer company (0-5 points)"""
        results = {}
        points = 0
        
        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit['must_have']:
                result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c['must_have']]
        for crit in optional_criteria:
            result = self.evaluate_criterion(consumers_df, crit, all_companies_data)
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
solv_scores = {}

for idx, (symbol, consumers_df) in enumerate(consumers_ratios_data.items()):
    score_result = solv_scorer.score_consumers(symbol, consumers_df, consumers_ratios_data)
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

    def get_metric_series(self, consumers_df, code):
        metric = consumers_df[consumers_df['code'] == code]
        if len(metric) == 0:
            return pd.Series(dtype=float)
        return metric['value'].dropna()

    def get_metric_value(self, consumers_df, code):
        s = self.get_metric_series(consumers_df, code)
        return None if s.empty else s.iloc[-1]

    def _ecdf_percentile(self, latest_value, peer_values):
        peer = pd.Series([v for v in peer_values if v is not None]).dropna()
        if peer.empty:
            return None
        return float((peer <= latest_value).mean())

    def evaluate_criterion(self, consumers_df, criterion, all_consumers_data):
        try:
            crit_type = criterion['type']

            if crit_type == 'absolute':
                code = criterion['code']
                latest = self.get_metric_value(consumers_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                if criterion['direction'] == 'lower':
                    pass_check = latest <= criterion['threshold']
                else:
                    pass_check = latest >= criterion['threshold']

                return {'pass': pass_check, 'value': latest, 'threshold': criterion['threshold']}

            if crit_type == 'peer_percentile':
                code = criterion['code']
                latest = self.get_metric_value(consumers_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                peer_values = [self.get_metric_value(df, code) for df in all_consumers_data.values()]
                pct = self._ecdf_percentile(latest, peer_values)
                if pct is None:
                    return {'pass': False, 'reason': 'No peer values', 'value': latest}

                pass_check = pct <= criterion['cut'] if criterion['direction'] == 'lower' else pct >= criterion['cut']

                return {'pass': pass_check, 'value': latest, 'percentile': pct, 'cut': criterion['cut']}

            if crit_type == 'history_vs_avg_exlatest':
                code = criterion['code']
                latest = self.get_metric_value(consumers_df, code)
                if latest is None:
                    return {'pass': False, 'reason': f'Code {code} not found', 'value': None}

                s = self.get_metric_series(consumers_df, code)
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
                pb = self.get_metric_value(consumers_df, criterion['pb_code'])
                roe = self.get_metric_value(consumers_df, criterion['roe_code'])

                if pb is None or roe is None or roe == 0:
                    return {'pass': False, 'reason': 'Missing PB or ROE or ROE=0', 'pb': pb, 'roe': roe}

                if roe < 0.10:
                    return {'pass': False, 'reason': 'ROE below gate (<10%)', 'pb': pb, 'roe': roe, 'ratio': pb/roe}

                ratio = pb / roe

                peer_ratios = []
                for df in all_consumers_data.values():
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

    def score_consumers(self, symbol, consumers_df, all_consumers_data):
        results = {}
        points = 0.0

        # Must-have criteria (2 points if all pass, 0 if any fail)
        must_have_results = []
        for crit in self.criteria:
            if crit.get('must_have', False):
                result = self.evaluate_criterion(consumers_df, crit, all_consumers_data)
                must_have_results.append(result['pass'])
                results[f"C{crit['id']}"] = result
        
        if all(must_have_results):
            points += 2  # 2 points for passing all must-have criteria
        
        # Optional criteria (distribute remaining 3 points)
        optional_criteria = [c for c in self.criteria if not c.get('must_have', False)]
        for crit in optional_criteria:
            result = self.evaluate_criterion(consumers_df, crit, all_consumers_data)
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

    for symbol, consumers_df in consumers_ratios_data.items():
        val_scores[symbol] = val_scorer.score_consumers(symbol, consumers_df, consumers_ratios_data)    
    
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

    print(f"Consumers Comprehensive Scores: \n {combined_scores_draft}")
except Exception as e:
    print(f"[Fund_CONSUMERS] ERROR: failed to build combined_scores_draft: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    combined_scores_draft = pd.DataFrame()