#%%
from pypfopt import EfficientFrontier, objective_functions, risk_models, HRPOpt, expected_returns
import sys
if 'Bluechip' in sys.modules:
    import Bluechip
    Bluechip._bluechip_cache.clear()
import TechAna_DRAFT as TechAna

START_DATE = TechAna.START_DATE
END_DATE = TechAna.END_DATE

import numpy as np
import pandas as pd
import requests
import scipy.cluster.hierarchy as sch
from scipy.fft import rfft, irfft, rfftfreq
from scipy.spatial.distance import squareform

#%%
print(quarterly := pd.Timestamp(END_DATE).quarter)
print(yearly := pd.Timestamp(END_DATE).year)

#%%
import TotalScore_Bank
import TotalScore_HealthCare
import TotalScore_Consumer
import TotalScore_GS
import TotalScore_RealEstate
import TotalScore_CM

# %%
import TotalScore_TechTele
import TotalScore_Financials
import TotalScore_UtiEne
import TotalScore_Insurance
import TotalScore_Materials

# %%
score_bank        = TotalScore_Bank.get_total_score(START_DATE, END_DATE)
score_healthcare  = TotalScore_HealthCare.get_total_score(START_DATE, END_DATE)
score_consumer    = TotalScore_Consumer.get_total_score(START_DATE, END_DATE)
score_gs          = TotalScore_GS.get_total_score(START_DATE, END_DATE)
score_real_estate = TotalScore_RealEstate.get_total_score(START_DATE, END_DATE)
score_cm          = TotalScore_CM.get_total_score(START_DATE, END_DATE)

#%%
score_financials  = TotalScore_Financials.get_total_score(START_DATE, END_DATE)
score_uti_ene     = TotalScore_UtiEne.get_total_score(START_DATE, END_DATE)
score_insurance   = TotalScore_Insurance.get_total_score(START_DATE, END_DATE)
score_materials   = TotalScore_Materials.get_total_score(START_DATE, END_DATE)
score_tech_tele   = TotalScore_TechTele.get_total_score(START_DATE, END_DATE)

#%%
df_bank        = score_bank.head(7)['Symbol'].tolist()
df_healthcare  = score_healthcare.head(2)['Symbol'].tolist()
df_consumer    = score_consumer.head(17)['Symbol'].tolist()
df_gs          = score_gs.head(9)['Symbol'].tolist()
df_real_estate = score_real_estate.head(17)['Symbol'].tolist()
df_cm          = score_cm.head(13)['Symbol'].tolist()
df_financials  = score_financials.head(7)['Symbol'].tolist()
df_uti_ene     = score_uti_ene.head(8)['Symbol'].tolist()
df_insurance   = score_insurance.head(3)['Symbol'].tolist()
df_materials   = score_materials.head(13)['Symbol'].tolist()
df_tech_tele   = score_tech_tele.head(4)['Symbol'].tolist()

# %%
industry_symbols = {
    "Bank": df_bank,
    "Goods_and_Services": df_gs,
    "Consumer": df_consumer,
    "HealthCare": df_healthcare,
    "RealEstate": df_real_estate,
    "CM": df_cm,
    "Financials": df_financials,
    "Utilities_and_Energy": df_uti_ene,
    "Insurance": df_insurance,
    "Materials": df_materials,
    "Tech_and_Telecom": df_tech_tele
}

symbols_df = pd.DataFrame(
    [
        {"symbol": sym, "industry": ind}
        for ind, syms in industry_symbols.items()
        for sym in syms
    ]
).drop_duplicates(subset=["symbol"]).reset_index(drop=True)

symbols = symbols_df["symbol"].tolist()

def _parse_stock_payload(payload):
    records = []
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for sym, rows in payload.items():
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        if "symbol" not in row:
                            row = {**row, "symbol": sym}
                        records.append(row)
            elif isinstance(rows, dict):
                row = rows
                if "symbol" not in row:
                    row = {**row, "symbol": sym}
                records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    for col in ["date", "Date", "trading_date", "TradingDate"]:
        if col in df.columns:
            df["date"] = pd.to_datetime(df[col], errors="coerce")
            break

    if "adj_close" in df.columns:
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    elif "close" in df.columns:
        df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")

    return df.dropna(subset=["symbol", "date", "adj_close"])


def _fetch_prices(symbol_list, start_date, end_date):
    if not symbol_list:
        return pd.DataFrame()
    params = {
        "symbols": ",".join(symbol_list),
        "start_date": start_date,
        "end_date": end_date
    }
    resp = requests.get(
        "http://192.168.8.190:8000/MKD/stock_daily",
        params=params,
        headers={"accept": "application/json"},
        timeout=60
    )
    resp.raise_for_status()
    payload = resp.json()
    return _parse_stock_payload(payload)

# %%
def _build_price_matrix(df_prices, filter_symbols=None):
    if df_prices.empty:
        return pd.DataFrame()
    price_matrix = df_prices.pivot_table(
        index="date",
        columns="symbol",
        values="adj_close",
        aggfunc="last"
    ).sort_index()
    
    if filter_symbols is not None:
        price_matrix = price_matrix[[col for col in price_matrix.columns if col in filter_symbols]]
    
    return price_matrix

def _compute_log_returns(price_matrix):
    return np.log(price_matrix / price_matrix.shift(1)).dropna(how="all")

def _compute_sortino(returns, annualize=252):
    downside = returns.where(returns < 0)
    downside_std = downside.std(ddof=0)
    mean_ret = returns.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = mean_ret / downside_std
    sortino = sortino.replace([np.inf, -np.inf], np.nan)
    if annualize:
        sortino = sortino * np.sqrt(annualize)
    return sortino

def _compute_cvar(returns, num_simulations=10000, time_horizon=23, confidence_level=0.05):
    cvar_dict = {}
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    for sym in returns.columns:
        m = mu[sym]
        s = sigma[sym]
        
        # Bo qua neu du lieu loi
        if pd.isna(m) or pd.isna(s) or s == 0:
            cvar_dict[sym] = np.nan
            continue
            
        # Mo phong Monte Carlo cho Geometric Brownian Motion (GBM)
        simulated_paths = np.random.normal(m, s, (time_horizon, num_simulations))
        
        # Tinh luy ke return sau time_horizon
        cumulative_returns = np.exp(simulated_paths.sum(axis=0)) - 1
        var_cut = np.percentile(cumulative_returns, confidence_level * 100)
        tail_losses = cumulative_returns[cumulative_returns <= var_cut]
        if tail_losses.size == 0:
            cvar = -var_cut
        else:
            cvar = -float(np.mean(tail_losses))
        cvar_dict[sym] = cvar
        
    return pd.Series(cvar_dict)

def _apply_risk_cutoff(price_matrix, cvar_threshold=0.5, min_assets=10):
    if price_matrix.empty:
        return price_matrix, pd.Series(dtype=float), pd.Index([])

    returns = _compute_log_returns(price_matrix)
    cvar_series = _compute_cvar(returns)
    cvar_series = cvar_series.reindex(price_matrix.columns)

    keep_symbols = cvar_series[cvar_series <= cvar_threshold].index
    if len(keep_symbols) < min_assets:
        keep_symbols = cvar_series.sort_values(ascending=True).head(min_assets).index

    dropped_symbols = pd.Index([c for c in price_matrix.columns if c not in set(keep_symbols)])
    filtered = price_matrix.reindex(columns=keep_symbols).dropna(how="all", axis=1)
    return filtered, cvar_series, dropped_symbols

def _compute_sortino(returns, annualize=252):
    downside = returns.where(returns < 0)
    downside_std = downside.std(ddof=0)
    mean_ret = returns.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        sortino = mean_ret / downside_std
    sortino = sortino.replace([np.inf, -np.inf], np.nan)
    if annualize:
        sortino = sortino * np.sqrt(annualize)
    return sortino

def _get_imported_tech_score_series(symbol_index):
    score_frames = [
        score_bank,
        score_healthcare,
        score_consumer,
        score_gs,
        score_real_estate,
        score_cm,
        score_financials,
        score_uti_ene,
        score_insurance,
        score_materials,
        score_tech_tele,
    ]

    tech_rows = []
    for df_score in score_frames:
        if isinstance(df_score, pd.DataFrame) and not df_score.empty:
            cols = set(df_score.columns)
            if "Symbol" in cols and "Tech_Score" in cols:
                tech_rows.append(df_score[["Symbol", "Tech_Score"]].copy())

    if not tech_rows:
        return pd.Series(0.0, index=symbol_index, dtype=float)

    tech_df = pd.concat(tech_rows, ignore_index=True)
    tech_df["Tech_Score"] = pd.to_numeric(tech_df["Tech_Score"], errors="coerce")
    tech_df = tech_df.dropna(subset=["Symbol", "Tech_Score"])
    tech_df = tech_df.drop_duplicates(subset=["Symbol"], keep="last")

    tech_series = pd.Series(tech_df["Tech_Score"].values, index=tech_df["Symbol"].values, dtype=float)
    tech_series = tech_series.reindex(symbol_index).fillna(0.0)
    return tech_series.clip(lower=0.0, upper=1.0)

def _compute_dpt_score(price_matrix, start_date=None, end_date=None, min_periods=45, min_cycle=12, max_cycle=90):
    if price_matrix.empty:
        return pd.Series(dtype=float)

    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE

    prices = price_matrix.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)].copy()
    if prices.empty:
        return pd.Series(0.0, index=price_matrix.columns)

    dpt_dict = {}
    for sym in prices.columns:
        s = pd.to_numeric(prices[sym], errors="coerce").dropna()
        if len(s) < min_periods:
            dpt_dict[sym] = 0.0
            continue

        log_price = np.log(s)
        x = (log_price - log_price.mean()).to_numpy()
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0)
        fft_vals = np.fft.rfft(x)

        cycle_mask = (freqs >= (1.0 / max_cycle)) & (freqs <= (1.0 / min_cycle))
        cycle_mask[0] = False
        filtered_fft = np.where(cycle_mask, fft_vals, 0)
        cyc = np.fft.irfft(filtered_fft, n=n)

        if np.allclose(cyc.std(), 0):
            dpt_dict[sym] = 0.0
            continue

        cyc_series = pd.Series(cyc, index=s.index)
        cmin = cyc_series.min()
        cmax = cyc_series.max()
        if np.isclose(cmax, cmin):
            trough_score = 0.5
        else:
            trough_score = float((cmax - cyc_series.iloc[-1]) / (cmax - cmin))

        lookback_slope = min(5, len(cyc_series) - 1)
        slope = cyc_series.iloc[-1] - cyc_series.iloc[-1 - lookback_slope]
        slope_scale = float(cyc_series.std())
        if slope_scale == 0:
            slope_score = 0.5
        else:
            slope_score = float(1 / (1 + np.exp(-(slope / slope_scale))))

        amp_strength = float(cyc_series.std() / (pd.Series(x).std() + 1e-9))
        amp_score = float(np.clip(amp_strength, 0.0, 1.0))

        dpt_score = (0.6 * np.clip(trough_score, 0.0, 1.0)) + (0.3 * slope_score) + (0.1 * amp_score)
        dpt_dict[sym] = float(np.clip(dpt_score, 0.0, 1.0))

    dpt_series = pd.Series(dpt_dict).reindex(price_matrix.columns).fillna(0.0)
    return dpt_series

def _compute_tech_score(price_matrix, sortino_series):
    if price_matrix.empty:
        return pd.Series(dtype=float)

    imported_tech = _get_imported_tech_score_series(price_matrix.columns).fillna(0.0)
    sortino_rank = sortino_series.reindex(price_matrix.columns).fillna(0).rank(pct=True)
    tech_score = (0.2 * sortino_rank) + (0.8 * imported_tech.clip(lower=0.0, upper=1.0))
    return tech_score.fillna(0)

def _cluster_and_select(corr_matrix, sortino_series, dpt_score, price_matrix, min_clusters=10, top_n=40, max_per_cluster=3, max_total_picks=27):#, max_per_sector=5): 
    if corr_matrix.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    sortino_series = sortino_series.reindex(price_matrix.columns).fillna(0.0)
    dpt_score = dpt_score.reindex(price_matrix.columns).fillna(0.0)

    # Two-stream funnel
    half_n = max(top_n // 2, 1)
    # Stream 1 – Alpha Xu hướng: top half_n cổ phiếu có tech_score cao nhất
    tech_score_full = _compute_tech_score(price_matrix, sortino_series).reindex(price_matrix.columns).fillna(0.0)
    group_trend = tech_score_full.nlargest(half_n).index
    # Stream 2 – Alpha Hoàn nguyên: top half_n cổ phiếu có dpt_score cao nhất
    group_revert = dpt_score.nlargest(half_n).index
    valid_symbols = group_trend.union(group_revert)
    if len(valid_symbols) < min_clusters:
        valid_symbols = tech_score_full.nlargest(min_clusters).index

    corr_clean = corr_matrix.loc[valid_symbols, valid_symbols].clip(lower=-1, upper=1).fillna(0.0)
    dpt_valid = dpt_score.reindex(valid_symbols).fillna(0.0)
    tech_score_valid = tech_score_full.reindex(valid_symbols).fillna(0.0)
    score_valid = (0.6 * tech_score_valid + 0.4 * dpt_valid).fillna(0.0)

    with np.errstate(invalid="ignore"):
        distance = np.sqrt(0.5 * (1 - corr_clean))
    distance = distance.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    condensed = squareform(distance.values, checks=False)
    linkage = sch.linkage(condensed, method="ward")
    
    distance_threshold = 0.5 * max(linkage[:, 2])
    clusters = sch.fcluster(linkage, t=distance_threshold, criterion="distance")

    sector_map = symbols_df.set_index('symbol')['industry'].to_dict()
    sector_values = [sector_map.get(sym, 'Unknown') for sym in valid_symbols]

    cluster_df = pd.DataFrame({
        "symbol": valid_symbols,
        "cluster": clusters,
        "tech_score": tech_score_valid.values,
        "dpt_score": dpt_valid.values,
        "score": score_valid.values,
        "sector": sector_values
    })

    cluster_df = cluster_df.sort_values(by="score", ascending=False)
    
    picks = []
    cluster_counts = {}
    sector_counts = {}

    for _, row in cluster_df.iterrows():
        c_id = row['cluster']
        sector = row['sector']
        
        if c_id not in cluster_counts:
            cluster_counts[c_id] = 0
        if sector not in sector_counts:
            sector_counts[sector] = 0
            
        if cluster_counts[c_id] < max_per_cluster: #and sector_counts[sector] < max_per_sector:
            picks.append(row)
            cluster_counts[c_id] += 1
            #sector_counts[sector] += 1
            
        if len(picks) >= max_total_picks:
            break
    selected = pd.DataFrame(picks)
    if not selected.empty:
        selected = selected.sort_values(["sector", "cluster", "score"], ascending=[True, True, False]).reset_index(drop=True)
    return linkage, cluster_df, selected

# %%
df_prices = _fetch_prices(symbols, START_DATE, END_DATE)
price_matrix = _build_price_matrix(df_prices, filter_symbols=symbols)
price_matrix, cvar_universe, dropped_by_risk = _apply_risk_cutoff(price_matrix, cvar_threshold=0.3)
if len(dropped_by_risk) > 0:
    print(f"Risk cutoff dropped {len(dropped_by_risk)} symbols")

log_returns = _compute_log_returns(price_matrix)
corr_matrix = log_returns.corr(method='spearman')
sortino_ratio = _compute_sortino(log_returns)
dpt_score = _compute_dpt_score(price_matrix, start_date=START_DATE, end_date=END_DATE)

linkage, cluster_members, selected_symbols = _cluster_and_select(corr_matrix, sortino_ratio, dpt_score, price_matrix)

#%%
print(selected_symbols)
# %%
symbols_q1 = {}
symbols_q2 = {}
symbols_q3 = {}
POSITION_SYMBOLS = set().union(symbols_q1, symbols_q2, symbols_q3)

# %%
def _get_lookback_dates(end_date=None, lookback_days=189):
    end_dt = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date).normalize()
    start_dt = end_dt - pd.Timedelta(days=lookback_days)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
#%% Use when the market sideways or bear
def _optimize_portfolio(selected_df, target_end_date, lookback_days=189, risk_free_rate=0.03):
    position = POSITION_SYMBOLS
    
    selected_symbols_list = set(selected_df["symbol"].dropna().unique().tolist())
    all_tickers = list(selected_symbols_list.union(position))
    
    start_dt, end_dt = _get_lookback_dates(end_date=target_end_date, lookback_days=lookback_days)
    df_prices_1y = _fetch_prices(all_tickers, start_dt, end_dt)
    price_matrix_1y = _build_price_matrix(df_prices_1y, filter_symbols=all_tickers).dropna(how="all")

    threshold = int(price_matrix_1y.shape[0] * 0.6) 
    price_matrix_1y = price_matrix_1y.dropna(axis=1, thresh=threshold).ffill().bfill()
    
    returns_1y = price_matrix_1y.pct_change().iloc[1:] 
    
    nan_counts = returns_1y.isna().sum()
    valid_tickers = nan_counts[nan_counts == 0].index.tolist()
    
    price_matrix_1y = price_matrix_1y[valid_tickers]
    returns_1y = returns_1y[valid_tickers]

    dpt_scores = _compute_dpt_score(price_matrix_1y, start_date=start_dt, end_date=end_dt)
    dpt_scores = dpt_scores.reindex(valid_tickers).fillna(0.5).clip(0, 1)

    if dpt_scores.max() > dpt_scores.min():
        custom_mu = 0.05 + dpt_scores * (0.3 - 0.05)
        
    cov_matrix = risk_models.CovarianceShrinkage(price_matrix_1y).ledoit_wolf()
    
    # MVO and HRP optimization
    ef = EfficientFrontier(custom_mu, cov_matrix, weight_bounds=(0.0, 0.20))
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    try:
        ef.max_quadratic_utility(risk_aversion=1.5)
    except:
        ef.min_volatility()
    
    w_mvo = pd.Series(ef.clean_weights(cutoff=0.0))
    w_mvo = w_mvo.reindex(valid_tickers).fillna(0)

    hrp = HRPOpt(returns=returns_1y)
    w_hrp_raw = pd.Series(hrp.optimize())
    w_hrp_raw = w_hrp_raw.reindex(valid_tickers).fillna(0)

    w_blend = (0.6 * w_mvo) + (0.4 * w_hrp_raw)

    w_final = w_blend.copy()
    
    privileged_symbols = set(position)

    while True:
        w_final = w_final[w_final > 0]
        if w_final.empty:
            break
            
        w_final = w_final / w_final.sum()

        min_thresholds = pd.Series(0.04, index=w_final.index)
        
        is_privileged = w_final.index.isin(privileged_symbols)
        min_thresholds[is_privileged] = 0.02

        violations = w_final[w_final < min_thresholds]
        
        if violations.empty:
            break

        w_final = w_final.drop(violations.idxmin())

    weights = w_final.round(4)
    
    diff = 1.0 - weights.sum()
    if diff != 0 and not weights.empty:
        weights.iloc[weights.argmax()] += diff
        weights = weights.round(4)

    ef_perf = EfficientFrontier(custom_mu, cov_matrix)
    ef_perf.weights = weights.to_dict()
    try:
        perf = ef_perf.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    except:
        perf = (0, 0, 0)
        
    return weights, perf
#%% Use when the market is bullish
def _compute_momentum(price_matrix):
    if price_matrix.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    current_prices = price_matrix.iloc[-1]
    sma_20 = price_matrix.iloc[-20:].mean()
    sma_50 = price_matrix.iloc[-50:].mean()
    momentum = (0.6 * (current_prices / sma_20)) + (0.4 * (current_prices / sma_50))
    return momentum

def _compute_score_series(price_matrix, sortino_series, tech_scores):
    if price_matrix.empty:
        return pd.Series(dtype=float)

    momentum = _compute_momentum(price_matrix)
    momentum_rank = momentum.rank(pct=True)
    sortino_rank = sortino_series.rank(pct=True)
    tech_series = pd.Series(
        {sym: tech_scores.get(sym, np.nan) for sym in price_matrix.columns}
    )
    tech_rank = tech_series.rank(pct=True)
    tech_score = momentum_rank * 0.6 + tech_rank * 0.4
    composite_score = (0.4 * sortino_rank) + (0.6 * tech_score)
    return composite_score.fillna(0)

def _optimize_portfolio(selected_df, target_end_date, lookback_days=189, risk_free_rate=0.03):
    position = POSITION_SYMBOLS
    
    selected_symbols_list = set(selected_df["symbol"].dropna().unique().tolist())
    all_tickers = list(selected_symbols_list.union(position))
    
    start_dt, end_dt = _get_lookback_dates(end_date=target_end_date, lookback_days=lookback_days)
    df_prices_1y = _fetch_prices(all_tickers, start_dt, end_dt)
    price_matrix_1y = _build_price_matrix(df_prices_1y, filter_symbols=all_tickers).dropna(how="all")

    threshold = int(price_matrix_1y.shape[0] * 0.6) 
    price_matrix_1y = price_matrix_1y.dropna(axis=1, thresh=threshold)
    
    price_matrix_1y = price_matrix_1y.ffill().bfill()
    
    returns_1y = price_matrix_1y.pct_change()
    returns_1y = returns_1y.iloc[1:] 
    
    nan_counts = returns_1y.isna().sum()
    valid_tickers = nan_counts[nan_counts == 0].index.tolist()
    
    price_matrix_1y = price_matrix_1y[valid_tickers]
    returns_1y = returns_1y[valid_tickers]
    tech_scores = _compute_tech_score(price_matrix_1y, _compute_sortino(returns_1y))
    sortino_1y = _compute_sortino(returns_1y)
    scores = _compute_score_series(price_matrix_1y, sortino_1y, tech_scores=tech_scores)

    # Tính Expected Return
    if scores.max() > scores.min():
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
        power_scores = norm_scores ** 2
        custom_mu = 0.05 + power_scores * (0.3 - 0.05)
    else:
        custom_mu = pd.Series(0.10, index=valid_tickers)
        power_scores = pd.Series(1.0, index=valid_tickers)
        
    cov_matrix = risk_models.CovarianceShrinkage(price_matrix_1y).ledoit_wolf()
    
    # MVO and HRP optimization
    ef = EfficientFrontier(custom_mu, cov_matrix, weight_bounds=(0.0, 0.20))
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    try:
        ef.max_quadratic_utility(risk_aversion=1.5)
    except:
        ef.min_volatility()
    
    w_mvo = pd.Series(ef.clean_weights(cutoff=0.0))
    w_mvo = w_mvo.reindex(valid_tickers).fillna(0)

    hrp = HRPOpt(returns=returns_1y)
    w_hrp_raw = pd.Series(hrp.optimize())
    w_hrp_raw = w_hrp_raw.reindex(valid_tickers).fillna(0)

    w_hrp_tilted = w_hrp_raw * (1 + power_scores)
    w_hrp = w_hrp_tilted / w_hrp_tilted.sum()
    w_blend = (0.65 * w_mvo) + (0.35 * w_hrp)

    w_final = w_blend.copy()
    
    privileged_symbols = set(position)

    while True:
        w_final = w_final[w_final > 0]
        if w_final.empty:
            break
            
        w_final = w_final / w_final.sum()

        min_thresholds = pd.Series(0.04, index=w_final.index)
        
        is_privileged = w_final.index.isin(privileged_symbols)
        min_thresholds[is_privileged] = 0.02

        violations = w_final[w_final < min_thresholds]
        
        if violations.empty:
            break

        w_final = w_final.drop(violations.idxmin())

    weights = w_final.round(4)
    
    diff = 1.0 - weights.sum()
    if diff != 0 and not weights.empty:
        weights.iloc[weights.argmax()] += diff
        weights = weights.round(4)

    ef_perf = EfficientFrontier(custom_mu, cov_matrix)
    ef_perf.weights = weights.to_dict()
    try:
        perf = ef_perf.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
    except:
        perf = (0, 0, 0)
        
    return weights, perf

# %%
weights, performance = _optimize_portfolio(selected_symbols, target_end_date=END_DATE)

if weights is not None and not weights.empty:
    active_weights = weights[weights > 0].sort_values(ascending=False)
    active_symbols = active_weights.index.tolist()
    
    print("Optimized weights:")
    print(active_weights)

    waitlist_symbols = [sym for sym in selected_symbols['symbol'].tolist() if sym not in active_symbols]
    waitlist_df = selected_symbols[selected_symbols['symbol'].isin(waitlist_symbols)]
    waitlist_symbols = waitlist_df.sort_values(by='score', ascending=False)['symbol'].tolist()
    
    print(f"Active Portfolio ({len(active_symbols)} stocks)")

#%%
def _compute_max_drawdown(series):
    if series is None or len(series) == 0:
        return 0.0
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min()) if len(drawdown) > 0 else 0.0

def _fetch_beta_map(symbol_list, target_date=None):
    beta_map = {}
    try:
        params = {}
        if target_date:
            params['date'] = target_date

        resp = requests.get(
            "http://192.168.8.190:8000/MKD/beta_calculation", 
            params=params, 
            timeout=60
        )
        
        if resp.status_code == 200:
            beta_json = resp.json()
            df_beta = pd.DataFrame(beta_json if isinstance(beta_json, list) else [beta_json])

            sym_col = next((c for c in ['symbol', 'Symbol', 'ticker', 'Ticker'] if c in df_beta.columns), 'symbol')
            if sym_col not in df_beta.columns:
                df_beta['symbol'] = None

            for sym in symbol_list:
                row = df_beta[df_beta[sym_col] == sym]
                if not row.empty:
                    beta_map[sym] = row.iloc[-1].to_dict()
    except Exception as e:
        print(f"Risk beta fetch error: {e}")
    
    return beta_map
    
def _simulate_dynamic_nav(active_weights, symbol_to_cluster, waitlist_by_cluster, initial_capital, start_date, end_date, df_prices, beta_map, carried_positions_config=None):
    if carried_positions_config is None:
        carried_positions_config = {}
        
    actual_dates = pd.DatetimeIndex(sorted(df_prices['date'].unique()))
    portfolio_daily_nav = pd.Series(0.0, index=actual_dates)
    trades_log = []
    
    max_allowed_capital = initial_capital * active_weights.min()

    for initial_sym, weight in active_weights.items():
        bucket_weight = weight
        bucket_capital = initial_capital * weight
        current_sym = initial_sym
        current_start_date = pd.to_datetime(start_date)
        bucket_nav = pd.Series(index=actual_dates, dtype=float)
        bucket_cash = 0.0

        while current_start_date <= pd.to_datetime(end_date):
            df_sym = df_prices[(df_prices['symbol'] == current_sym) & (df_prices['date'] >= current_start_date)].sort_values('date')
            
            if df_sym.empty or pd.isna(df_sym.iloc[0].get('adj_close', np.nan)):
                bucket_nav.loc[current_start_date:] = bucket_capital + bucket_cash
                break

            current_market_price = df_sym.iloc[0]['adj_close']
            shares = bucket_capital / current_market_price 
            
            if current_sym in carried_positions_config:
                config_data = carried_positions_config[current_sym]
                original_entry_price = config_data.get('entry_price', current_market_price)
                highest_price = max(config_data.get('highest_price', original_entry_price), current_market_price)
                trade_entry_date = config_data.get('entry_date', df_sym.iloc[0]['date'])
            else:
                original_entry_price = current_market_price
                highest_price = current_market_price
                trade_entry_date = df_sym.iloc[0]['date']
            
            beta_data = beta_map.get(current_sym, {})
            beta_val = beta_data.get('beta_3m', None)
            try:
                beta_magnitude = abs(float(beta_val))
            except (TypeError, ValueError):
                beta_magnitude = 1.0
            
            # Tính Stop Loss dựa trên original_entry_price (Giá gốc) thay vì entry_price
            base_sl_pct = 0.85  
            sl_price = original_entry_price * base_sl_pct
            
            if beta_magnitude > 1.2:
                take_profit_pct = 0.3
            elif beta_magnitude < 0.9:
                take_profit_pct = 0.2
            else:
                take_profit_pct = 0.25
            
            # Tính Take Profit dựa trên original_entry_price
            tp_price = original_entry_price * (1 + take_profit_pct)
            
            exit_date = df_sym.iloc[-1]['date']
            exit_price = df_sym.iloc[-1]['adj_close']
            exit_reason = 'Keep Position'
            
            trade_dates, trade_values = [], []
            days_held = 0

            for _, row in df_sym.iterrows():
                day_date = row['date']
                day_price = row.get('adj_close', np.nan)

                if pd.isna(day_price):
                    continue

                if day_price > highest_price:
                    highest_price = day_price

                if day_price >= tp_price:
                    exit_date = day_date
                    exit_price = day_price
                    exit_reason = 'Take Profit'
                    trade_dates.append(day_date)
                    trade_values.append(shares * exit_price + bucket_cash)
                    break

                if day_price <= sl_price:
                    exit_date = day_date
                    exit_price = day_price
                    exit_reason = 'Stop Loss'
                    trade_dates.append(day_date)
                    trade_values.append(shares * exit_price + bucket_cash)
                    break 

                trade_dates.append(day_date)
                trade_values.append(shares * day_price + bucket_cash)
                days_held += 1

            bucket_nav.update(pd.Series(trade_values, index=trade_dates))
            final_trade_value = shares * exit_price
            
            # Ghi log lợi nhuận % dựa trên Giá gốc và Ngày mua gốc
            trades_log.append({
                'Symbol': current_sym,
                'Weight': bucket_weight,
                'Beta': round(beta_magnitude, 2),
                'Entry_Date': trade_entry_date.strftime('%Y-%m-%d') if isinstance(trade_entry_date, pd.Timestamp) else trade_entry_date,
                'Exit_Date': exit_date.strftime('%Y-%m-%d'), 
                'Return_Pct': (exit_price - original_entry_price) / original_entry_price,
                'Highest_Return_Pct': (highest_price - original_entry_price) / original_entry_price,
                'Exit_Reason': exit_reason
            })

            if exit_reason in ['Stop Loss', 'Take Profit'] and exit_date < pd.to_datetime(end_date):
                current_start_date = exit_date + pd.Timedelta(days=1)
                if current_sym in carried_positions_config:
                    del carried_positions_config[current_sym]
                    
                break
            else:
                break

        portfolio_daily_nav = portfolio_daily_nav.add(bucket_nav.ffill().fillna(0), fill_value=0)
    return portfolio_daily_nav, pd.DataFrame(trades_log)

#%%
def _auto_fetch_carried_positions(symbols, prev_start_date, prev_end_date):
    if not symbols:
        return {}
        
    df_prev = _fetch_prices(list(symbols), prev_start_date, prev_end_date)
    
    if df_prev.empty:
        return {}
        
    carried_config = {}
    for sym in symbols:
        df_sym = df_prev[df_prev['symbol'] == sym].sort_values('date')
        
        if not df_sym.empty:
            entry_date = df_sym.iloc[0]['date']
            entry_price = df_sym.iloc[0]['adj_close']
            highest_price = df_sym['adj_close'].max()
            
            carried_config[sym] = {
                'entry_date': entry_date,
                'entry_price': entry_price,
                'highest_price': highest_price
            }
            
    return carried_config

def run_forward_test(weights, cluster_members, initial_capital= 1_000_000_000, entry_delay_days=0, carried_config=None):
    if carried_config is None:
        carried_config = {}
        
    curr_q = pd.to_datetime(TechAna.END_DATE).to_period('Q') + 1
    
    # Calculate base start and end dates
    base_start_date = curr_q.start_time
    NEXT_END_DATE = curr_q.end_time.strftime('%Y-%m-%d')
    
    # Apply entry delay
    delayed_start_date = base_start_date + pd.Timedelta(days=entry_delay_days)
    NEXT_START_DATE = delayed_start_date.strftime('%Y-%m-%d')
    
    delay_info = f" (Entry delayed by {entry_delay_days} days)" if entry_delay_days > 0 else ""
    print(f"\n{'='*50}\n DYNAMIC FORWARD TEST PORTFOLIO ({curr_q}){delay_info}\n{'='*50}")
    print(f"Entry Date: {NEXT_START_DATE} | Exit Date: {NEXT_END_DATE}")
    
    if weights is None or weights.empty:
        print("Error: Input weights are empty.")
        return {}

    active_weights = weights[weights > 0].copy()
    active_symbols = active_weights.index.tolist()

    all_target_symbols = active_symbols 
    
    df_prices = _fetch_prices(all_target_symbols, NEXT_START_DATE, NEXT_END_DATE)
    
    if df_prices.empty:
        print("Error: Could not fetch prices for the target quarter.")
        return {}

    beta_map = _fetch_beta_map(all_target_symbols, target_date=NEXT_START_DATE)

    portfolio_nav, trades_log = _simulate_dynamic_nav(
        active_weights=active_weights, 
        symbol_to_cluster={},   
        waitlist_by_cluster={},
        initial_capital=initial_capital, 
        start_date=NEXT_START_DATE, 
        end_date=NEXT_END_DATE, 
        df_prices=df_prices, 
        beta_map=beta_map,
        carried_positions_config=carried_config
    )

    # Build VNINDEX market benchmark in the same test window.
    benchmark_ret_annual = 0.0
    benchmark_vol_annual = 0.0
    benchmark_sharpe = 0.0
    benchmark_ret_quarter = 0.0
    rf_rate = 0.03
    try:
        vni_dict = TechAna.APIFetcher.fetch_batch(
            "http://192.168.8.190:8000/MKD/stock_index_daily",
            ["VNINDEX"],
            NEXT_START_DATE,
            NEXT_END_DATE
        )
        vni_df = vni_dict.get("VNINDEX")
        if isinstance(vni_df, pd.DataFrame) and not vni_df.empty:
            price_col = "close" if "close" in vni_df.columns else ("adj_close" if "adj_close" in vni_df.columns else None)
            if price_col is not None:
                vni_prices = pd.to_numeric(vni_df[price_col], errors="coerce").dropna().sort_index()
                vni_daily = vni_prices.pct_change().dropna()
                if not vni_daily.empty:
                    benchmark_ret_annual = float(vni_daily.mean() * 252)
                    benchmark_vol_annual = float(vni_daily.std(ddof=0) * np.sqrt(252))
                    benchmark_ret_quarter = float((1.0 + vni_daily).prod() - 1.0)
                    if benchmark_vol_annual > 0:
                        benchmark_sharpe = (benchmark_ret_annual - rf_rate) / benchmark_vol_annual
    except Exception as e:
        print(f"VNINDEX benchmark error: {e}")

    profit = portfolio_nav.iloc[-1] - initial_capital
    daily_ret = portfolio_nav.pct_change().dropna()
    port_ret_annual = float(daily_ret.mean() * 252) if not daily_ret.empty else 0.0
    port_vol_annual = float(daily_ret.std(ddof=0) * np.sqrt(252)) if not daily_ret.empty else 0.0
    sharpe = (port_ret_annual - rf_rate) / port_vol_annual if port_vol_annual > 0 else 0.0

    beta_values = []
    beta_weights = []
    for sym, w in active_weights.items():
        beta_raw = beta_map.get(sym, {}).get('beta_3m', np.nan)
        try:
            beta_val = float(beta_raw)
        except (TypeError, ValueError):
            beta_val = np.nan
        if not pd.isna(beta_val):
            beta_values.append(beta_val)
            beta_weights.append(float(w))

    portfolio_beta = float(np.average(beta_values, weights=beta_weights)) if beta_values else 1.0

    final_nav = portfolio_nav.iloc[-1]
    total_return = profit / initial_capital
    n_days = max(len(daily_ret), 1)
    rf_quarter = float((1.0 + rf_rate) ** (n_days / 252.0) - 1.0)
    alpha_quarter = total_return - (rf_quarter + portfolio_beta * (benchmark_ret_quarter - rf_quarter))
    max_drawdown = _compute_max_drawdown(portfolio_nav)
    max_loss = ((portfolio_nav - initial_capital) / initial_capital).min()
    win_rate = (trades_log['Return_Pct'] > 0).sum() / len(trades_log) if len(trades_log) > 0 else 0

    print(f"Final NAV:      {final_nav:,.0f} VND\n"
          f"Profit/Loss:    {profit:,.0f} VND\n"
          f"Return:         {total_return:.2%}\n"
          f"Max Drawdown:   {max_drawdown:.2%}\n"
          f"Max Loss:       {max_loss:.2%}\n"
            f"Win Rate:       {win_rate:.2%}\n"
            f"Sharpe:         {sharpe:.3f}\n"
            f"Alpha (Quarter):{alpha_quarter:.2%}\n")
    print("=" * 50)
    print(trades_log.to_string(index=False))

    try:
        filename = f'Portfolio_Report_{curr_q}.xlsx'
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            summary_stats = pd.DataFrame({
                'Metric': ['Quarter', 'Initial Capital', 'Final NAV', 'Profit/Loss', 'Return (%)', 
                          'Max Drawdown (%)', 'Max Loss (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Alpha Q (%)',
                          'Portfolio Beta', 'VNINDEX Return Q (%)', 'VNINDEX Sharpe'],
                'Value': [str(curr_q), initial_capital, final_nav, profit, 
                         total_return * 100, max_drawdown * 100, max_loss * 100, 
                         win_rate * 100, sharpe, alpha_quarter * 100,
                         portfolio_beta, benchmark_ret_quarter * 100, benchmark_sharpe]
            })
            summary_stats.to_excel(writer, sheet_name='Summary', index=False)
            
            daily_portfolio = pd.DataFrame({
                'Date': portfolio_nav.index,
                'NAV': portfolio_nav.values,
            })
            daily_portfolio.to_excel(writer, sheet_name='Daily_Returns', index=False)
            
            trades_log.to_excel(writer, sheet_name='Trades_Log', index=False)
        
        print(f"Exported all data to: {filename}")
        
    except Exception as e:
        print(f"Error exporting Excel file: {e}")
    return beta_map

# %%
config_q1 = _auto_fetch_carried_positions(symbols_q1, "2025-01-01", "2025-03-31")
config_q2 = _auto_fetch_carried_positions(symbols_q2, "2025-04-01", "2025-06-30")
config_q3 = _auto_fetch_carried_positions(symbols_q3, "2025-07-01", "2025-09-30")
master_carried_config = {**config_q1, **config_q2, **config_q3}
beta_map = run_forward_test(weights=weights, cluster_members=cluster_members, entry_delay_days=0, carried_config=master_carried_config)

# %%
print(selected_symbols)

#%%
sortino_ratio.sort_values(ascending=False).head(20)