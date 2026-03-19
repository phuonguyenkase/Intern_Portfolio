import requests
import pandas as pd
import TechAna_DRAFT as TechAna

STOCK_DAILY_URL = "http://192.168.8.190:8000/MKD/stock_daily"


def _normalize_stock_daily_rows(payload):
    rows = []

    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        for symbol, value in payload.items():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        rows.append({**row, "symbol": row.get("symbol", symbol)})
            elif isinstance(value, dict):
                rows.append({**value, "symbol": value.get("symbol", symbol)})

    return rows


def _fetch_avg_price_map(symbols, start_date, end_date):
    """Return average adj_close (fallback to close) per symbol over the date range."""
    if not symbols or not start_date or not end_date:
        return {}

    params = {
        "symbols": ",".join(symbols),
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(
        STOCK_DAILY_URL,
        params=params,
        headers={"accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    rows = _normalize_stock_daily_rows(response.json())
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns:
        return {}

    # Use adj_close; fall back to close when adj_close is missing.
    if "adj_close" in df.columns:
        df["_price"] = pd.to_numeric(df["adj_close"], errors="coerce")
        if "close" in df.columns:
            df["_price"] = df["_price"].fillna(pd.to_numeric(df["close"], errors="coerce"))
    elif "close" in df.columns:
        df["_price"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        return {}

    avg = df.dropna(subset=["_price"]).groupby("symbol")["_price"].mean()
    return avg.to_dict()


def _fetch_volume_frame(symbols, start_date, end_date):
    if not symbols or not start_date or not end_date:
        return pd.DataFrame()

    params = {
        "symbols": ",".join(symbols),
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(
        STOCK_DAILY_URL,
        params=params,
        headers={"accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()

    rows = _normalize_stock_daily_rows(response.json())
    if not rows:
        return pd.DataFrame()

    df_vol = pd.DataFrame(rows)
    if "symbol" not in df_vol.columns or "total_volume" not in df_vol.columns:
        return pd.DataFrame()

    df_vol["vol"] = pd.to_numeric(df_vol["total_volume"], errors="coerce")
    return df_vol.dropna(subset=["vol"])


def filter_stocks_by_price_and_liquidity(
    stocks,
    min_price=10000,
    min_volume_threshold=20000,
    min_pass_rate=0.50,
    start_date=None,
    end_date=None,
):
    """Filter stocks by minimum price and trading liquidity pass-rate."""
    if not stocks:
        return []

    start_date = start_date or getattr(TechAna, "START_DATE", None)
    end_date = end_date or getattr(TechAna, "END_DATE", None)

    filtered_stocks = [s for s in stocks if isinstance(s, dict) and s.get("symbol")]
    symbols = [s["symbol"] for s in filtered_stocks]

    price_map = _fetch_avg_price_map(symbols, start_date, end_date)
    if price_map:
        min_price_symbols = {symbol for symbol, price in price_map.items() if price >= min_price}
        filtered_stocks = [s for s in filtered_stocks if s.get("symbol") in min_price_symbols]
        symbols = [s["symbol"] for s in filtered_stocks]

    df_vol = _fetch_volume_frame(symbols, start_date, end_date)
    if not df_vol.empty:
        pass_rate_per_sym = (
            df_vol.groupby("symbol")["vol"]
            .apply(lambda x: (x >= min_volume_threshold).mean())
        )
        liquid_symbols = set(pass_rate_per_sym[pass_rate_per_sym > min_pass_rate].index)
        filtered_stocks = [s for s in filtered_stocks if s.get("symbol") in liquid_symbols]

    return filtered_stocks
