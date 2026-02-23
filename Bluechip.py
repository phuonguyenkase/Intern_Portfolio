# %%
"""
Dynamic bluechip classification based on market capitalization per industry.
Top 4 by market cap = Mega Caps
Rank 5-10 by market cap = Large Caps
"""

import requests
import pandas as pd

# Industry mappings
INDUSTRY_MAPPING = {
    'Real Estate': {'industry_lv1': 'Financials', 'industry_lv2': 'Real Estate'},
    'Bank': {'industry_lv1': 'Financials', 'industry_lv2': 'Bank'},
    'Financials': {'industry_lv1': 'Financials', 'industry_lv2': None},
    'Insurance': {'industry_lv1': 'Financials', 'industry_lv2': 'Insurance'},
    'Industrials_CM': {'industry_lv1': 'Industrials', 'industry_lv2': 'Construction & Materials'},
    'Industrials_GS': {'industry_lv1': 'Industrials', 'industry_lv2': 'Goods & Services'},
    'Consumer': {'industry_lv1': 'Consumer', 'industry_lv2': None},
    'Energy_Utilities': {'industry_lv1': ['Energy', 'Utilities'], 'industry_lv2': None},
    'Healthcare': {'industry_lv1': 'Healthcare', 'industry_lv2': None},
    'Materials': {'industry_lv1': 'Materials', 'industry_lv2': None},
    'Technology': {'industry_lv1': ['Information Technology', 'Telecommunication'], 'industry_lv2': None},
}

def get_market_cap_rankings(symbols_list, year=2025, quarter=3, top_n=10):
    if not symbols_list:
        return {'ranked_symbols': [], 'mega_caps': [], 'large_caps': []}
    
    try:
        url = "http://192.168.8.190:8000/MKD/stock-ratios"
        params = {
            "symbols": ",".join(symbols_list),
            "period": "quarterly",
            "num_periods": 4,  # Get last 4 quarters
            "language": "en"
        }
        
        r = requests.get(url, params=params, headers={"accept": "application/json"}, timeout=30)
        r.raise_for_status()
        all_ratios_data = r.json()
        
        market_caps = {}
        for record in all_ratios_data:
            if (record.get('code') == 'ryd11' and 
                record.get('year') == year and 
                record.get('quarter') == quarter):
                symbol = record.get('symbol')
                value = record.get('value')
                if symbol and value is not None:
                    market_caps[symbol] = float(value)
        
        if not market_caps:
            market_cap_by_symbol = {}
            for record in all_ratios_data:
                if record.get('code') == 'ryd11':
                    symbol = record.get('symbol')
                    value = record.get('value')
                    year_q = record.get('year')
                    quarter_q = record.get('quarter')
                    if symbol and value is not None and year_q and quarter_q:
                        period_key = (year_q, quarter_q)
                        if symbol not in market_cap_by_symbol:
                            market_cap_by_symbol[symbol] = {}
                        market_cap_by_symbol[symbol][period_key] = float(value)
            
            # Get latest period for each symbol
            for symbol, periods in market_cap_by_symbol.items():
                if periods:
                    latest_period = max(periods.keys())
                    market_caps[symbol] = periods[latest_period]
        
        if not market_caps:
            print(f"[config_bluechip] No market cap data found")
            return {'ranked_symbols': [], 'mega_caps': [], 'large_caps': []}
        
        # Sort by market cap descending
        ranked = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:top_n]  # Take top N
        
        mega_caps = [sym for sym, _ in ranked[:4]]
        large_caps = [sym for sym, _ in ranked[4:10]]
        
        return {
            'ranked_symbols': ranked,
            'mega_caps': mega_caps,
            'large_caps': large_caps
        }
        
    except Exception as e:
        print(f"[config_bluechip] Error fetching market caps: {e}")
        return {'ranked_symbols': [], 'mega_caps': [], 'large_caps': []}

def get_bluechips_for_industry(industry_name, symbols_list, year=2025, quarter=3):
    result = get_market_cap_rankings(symbols_list, year, quarter, top_n=10)
    
    return {
        'industry': industry_name,
        'mega_caps': result['mega_caps'],
        'large_caps': result['large_caps'],
        'all_bluechips': result['mega_caps'] + result['large_caps'],
        'rankings': result['ranked_symbols']
    }

# Cache for bluechips by industry (to avoid repeated API calls)
_bluechip_cache = {}

def get_cached_bluechips(industry_name, symbols_list, year=2025, quarter=3, force_refresh=False):
    cache_key = f"{industry_name}_{year}_Q{quarter}"
    
    if not force_refresh and cache_key in _bluechip_cache:
        return _bluechip_cache[cache_key]
    
    result = get_bluechips_for_industry(industry_name, symbols_list, year, quarter)
    _bluechip_cache[cache_key] = result
    
    return result