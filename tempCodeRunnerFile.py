# %%
import TechAna_DRAFT as TechAna
import pandas as pd
import requests
import numpy as np
import importlib
import warnings
warnings.filterwarnings('ignore')

# %%
INDUSTRY_MAP = {
    1: {'name': 'Banks','module': 'TotalScore_Bank'},
    2: {'name': 'Consumer','module': 'TotalScore_Consumer'},
    3: {'name': 'Financials','module': 'TotalScore_Financials'},
    4: {'name': 'Construction_and_materials','module': 'TotalScore_CM'},
    5: {'name': 'Goods_and_services','module': 'TotalScore_GS'},
    6: {'name': 'HealthCare','module': 'TotalScore_HealthCare'},
    7: {'name': 'Insurance','module': 'TotalScore_Insurance'},
    8: {'name': 'Materials','module': 'TotalScore_Materials'},
    9: {'name': 'RealEstate','module': 'TotalScore_RealEstate'},
    10: {'name': 'Utilities_and_Energy','module': 'TotalScore_UtiEne'},
    11: {'name': 'TechTele','module': 'TotalScore_TechTele'},
}

# %%
def _parse_stock_payload(payload):
    records = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for sym, rows in payload.items():
            if isinstance(rows, list):
                for r in rows:
                    if 'symbol' not in r:
                        r = {**r, 'symbol': sym}
                    records.append(r)
            elif isinstance(rows, dict):
                if 'symbol' not in rows:
                    rows = {**rows, 'symbol': sym}
                records.append(rows)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    for col in ['date', 'Date', 'trading_date', 'TradingDate']:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col], errors='coerce')
            break

    # Use Adj Close for all OHLC if available
    if 'adj_close' in df.columns:
        adj = pd.to_numeric(df['adj_close'], errors='coerce')
        df['open'] = adj
        df['high'] = adj
        df['low'] = adj
        df['close'] = adj

    else:
        for col in ['open', 'Open']:
            if col in df.columns:
                df['open'] = pd.to_numeric(df[col], errors='coerce')
                break
        for col in ['high', 'High']:
            if col in df.columns:
                df['high'] = pd.to_numeric(df[col], errors='coerce')
                break
        for col in ['low', 'Low']:
            if col in df.columns:
                df['low'] = pd.to_numeric(df[col], errors='coerce')
                break
        for col in ['close', 'Close']:
            if col in df.columns:
                df['close'] = pd.to_numeric(df[col], errors='coerce')
                break

    return df.dropna(subset=['symbol', 'date'])


def _compute_max_drawdown(series):
    if series is None or len(series) == 0:
        return 0.0
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min()) if len(drawdown) > 0 else 0.0


def _fetch_prices(symbols, start_date, end_date):
    if not symbols:
        return pd.DataFrame()
    params = {
        "symbols": ",".join(symbols),
        "start_date": start_date,
        "end_date": end_date
    }
    resp = requests.get(
        "http://192.168.8.190:8000/MKD/stock_daily",
        params=params,
        headers={"accept": "application/json"},
        timeout=30
    )
    resp.raise_for_status()
    payload = resp.json()
    return _parse_stock_payload(payload)


def _run_quarter_trades(symbols, start_date, end_date, df_prices, entry_override=None, original_entry_override=None):
    entry_override = entry_override or {}
    original_entry_override = original_entry_override or {}

    trades = []
    for sym in symbols:
        df_sym = df_prices[df_prices['symbol'] == sym].sort_values('date').reset_index(drop=True)
        if df_sym.empty:
            continue

        entry_row = df_sym.iloc[0]
        entry_date = entry_row['date']
        entry_price = entry_override.get(sym, entry_row.get('open', np.nan))
        if pd.isna(entry_price):
            continue

        original_entry_price = original_entry_override.get(sym, entry_price)

        sl_price = entry_price * 0.85
        tp_price = entry_price * 1.25

        exit_date = df_sym.iloc[-1]['date']
        exit_price = df_sym.iloc[-1]['close'] if 'close' in df_sym.columns else entry_price
        exit_reason = 'Keep Position'

        start_idx = 3 if len(df_sym) > 3 else len(df_sym)
        for i in range(start_idx, len(df_sym)):
            row = df_sym.iloc[i]
            day_open = row.get('open', np.nan)
            day_low = row.get('low', np.nan)
            day_high = row.get('high', np.nan)

            # Stoploss check (-15%)
            if pd.notna(day_low) and day_low <= sl_price:
                if pd.notna(day_open) and day_open <= sl_price:
                    exit_price = day_open
                else:
                    exit_price = sl_price
                exit_date = row['date']
                exit_reason = 'Stop Loss'
                break

            # Take profit check (+25%)
            if pd.notna(day_high) and day_high >= tp_price:
                exit_price = tp_price
                exit_date = row['date']
                exit_reason = 'Take Profit'
                break

        ret_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
        cum_ret_pct = (exit_price - original_entry_price) / original_entry_price if original_entry_price else 0.0

        max_dd = 0.0
        max_ret = 0.0
        min_ret = 0.0
        if 'close' in df_sym.columns:
            hold_df = df_sym[(df_sym['date'] >= entry_date) & (df_sym['date'] <= exit_date)]
            close_series = hold_df['close'].dropna()
            price_path = pd.concat([pd.Series([entry_price]), close_series], ignore_index=True)
            max_dd = _compute_max_drawdown(price_path)

            if not close_series.empty and entry_price:
                max_ret = (close_series.max() - entry_price) / entry_price
                min_ret = (close_series.min() - entry_price) / entry_price

        trades.append({
            'Symbol': sym,
            'Entry_Date': entry_date,
            'Entry_Price': entry_price,
            'Original_Entry_Price': original_entry_price,
            'Exit_Date': exit_date,
            'Exit_Price': exit_price,
            'Exit_Reason': exit_reason,
            'Return_Pct': ret_pct,
            'Max_Return_Pct': max_ret,
            'Min_Return_Pct': min_ret,
            'Cum_Return_Pct': cum_ret_pct,
            'Max_Drawdown': max_dd
        })

    return pd.DataFrame(trades)

def print_summary(df_trades, label, start_date, end_date, industry_name):
    if df_trades.empty:
        print(f'No trades generated for {label}.')
        return

    win_rate = (df_trades['Return_Pct'] > 0).mean()
    avg_returns = df_trades['Return_Pct'].mean()
    max_drawdown = df_trades['Max_Drawdown'].min()
    max_return = df_trades['Return_Pct'].max()
    min_return = df_trades['Return_Pct'].min()

    summary_df = pd.DataFrame([{
        'Industry': industry_name,
        'Label': label,
        'Period_Start': start_date,
        'Period_End': end_date,
        'Win_Rate': win_rate,
        'Average_Return': avg_returns,
        'Max_Return': max_return,
        'Min_Return': min_return,
        'Max_Drawdown': max_drawdown,
        'Deals': len(df_trades)
    }])

    print(f'\nTrade Results ({label}):')
    print(df_trades.to_string(index=False))
    print('\nSummary:')
    print(summary_df.to_string(index=False))

def run_backtest(industry_id):
    if industry_id not in INDUSTRY_MAP:
        print(f"Error: Industry ID {industry_id} not found in configuration.")
        return

    config = INDUSTRY_MAP[industry_id]
    industry_name = config['name']
    module_name = config['module']
    
    print(f"STARTING BACKTEST FOR: {industry_id} - {industry_name} (Module: {module_name})")
    try:
        TotalScore_Module = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import module '{module_name}'. Check if file exists.")
        return

    TECH_START_DATE = TechAna.START_DATE
    TECH_END_DATE = TechAna.END_DATE

    tech_start = pd.to_datetime(TECH_START_DATE)
    tech_end = pd.to_datetime(TECH_END_DATE)
    tech_q = tech_end.to_period('Q')

    curr_q = tech_q + 1
    next_q = curr_q + 1

    PREV_START_DATE = tech_q.start_time.strftime('%Y-%m-%d')
    PREV_END_DATE = tech_q.end_time.strftime('%Y-%m-%d')
    START_DATE = curr_q.start_time.strftime('%Y-%m-%d')
    END_DATE = curr_q.end_time.strftime('%Y-%m-%d')
    NEXT_START_DATE = next_q.start_time.strftime('%Y-%m-%d')
    NEXT_END_DATE = next_q.end_time.strftime('%Y-%m-%d')

# Ranking for quarter t-1 (entry list for backtest quarter t)
    df_total_prev = TotalScore_Module.get_total_score(PREV_START_DATE, PREV_END_DATE, industry=industry_name)
    df_rank_prev = df_total_prev[['Symbol', 'Final_Score']].copy()
    df_rank_prev = df_rank_prev.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    TOP10_PREV = df_rank_prev.head(10)['Symbol'].tolist()

# Ranking for quarter t (used for rollover filter and next-quarter new entries)
    df_total_curr = TotalScore_Module.get_total_score(START_DATE, END_DATE, industry=industry_name)
    df_rank_curr = df_total_curr[['Symbol', 'Final_Score']].copy()
    df_rank_curr = df_rank_curr.sort_values('Final_Score', ascending=False).reset_index(drop=True)
    TOP10_CURR = df_rank_curr.head(10)['Symbol'].tolist()
    TOP13_CURR = df_rank_curr.head(13)['Symbol'].tolist()

    df_prices_curr = _fetch_prices(TOP10_PREV, START_DATE, END_DATE)
    df_trades_curr = _run_quarter_trades(TOP10_PREV, START_DATE, END_DATE, df_prices_curr)

    rollover_symbols = []
    entry_override_next = {}
    original_entry_override_next = {}

    if not df_trades_curr.empty:
        pending_df = df_trades_curr[df_trades_curr['Exit_Reason'] == 'Keep Position']
        for _, row in pending_df.iterrows():
            sym = row['Symbol']
            if sym in TOP13_CURR:
                rollover_symbols.append(sym)
                original_entry_price = row.get('Original_Entry_Price', row['Entry_Price'])
                if row.get('Return_Pct', 0) > 0:
                    reference_price = row['Exit_Price']
                else:
                    reference_price = original_entry_price

                entry_override_next[sym] = reference_price
                original_entry_override_next[sym] = original_entry_price
                df_trades_curr.loc[df_trades_curr['Symbol'] == sym, 'Exit_Reason'] = 'Rollover'

    print_summary(df_trades_curr, f"{curr_q.year}Q{curr_q.quarter}", START_DATE, END_DATE, industry_name)

    symbols_next = TOP10_CURR + [s for s in rollover_symbols if s not in TOP10_CURR]
    df_prices_next = _fetch_prices(symbols_next, NEXT_START_DATE, NEXT_END_DATE)
    df_trades_next = _run_quarter_trades(
        symbols_next,
        NEXT_START_DATE,
        NEXT_END_DATE,
        df_prices_next,
        entry_override=entry_override_next,
        original_entry_override=original_entry_override_next
    )

    print_summary(df_trades_next, f"{next_q.year}Q{next_q.quarter}", NEXT_START_DATE, NEXT_END_DATE, industry_name)

# %%
if __name__ == "__main__":
    try:
        selection = int(input("Enter Industry ID to Backtest: "))
        run_backtest(selection)
    except ValueError:
        print("Invalid input. Please enter a number.")

#%%