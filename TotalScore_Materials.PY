# %%
import pandas as pd
import TechAna

START_DATE = "2025-09-01"
END_DATE = "2025-12-31"

try:
    from Fund_Materials import combined_scores_draft as df_final
    print("Success")
except ImportError:
    print("ERROR")
    df_final = pd.DataFrame(columns=['Symbol', 'Total_Score'])

if not df_final.empty:
    symbols_to_analyze = df_final['Symbol'].tolist()
else:
    symbols_to_analyze = []
    print("There's no stock")

# %%
import numpy as np
if symbols_to_analyze:
    print(f"Running comprehensive analysis for {len(symbols_to_analyze)} Basic Materials symbols...")
    tech_scores, foreign_scores, risk_scores = TechAna.analyze_symbols(
        symbols_to_analyze, 
        START_DATE, 
        END_DATE,
        industry='Materials',
        year=2025,
        quarter=3)

    stock_data = TechAna.APIFetcher.fetch_batch(
        "http://192.168.8.190:8000/MKD/stock_daily", 
        symbols_to_analyze, START_DATE, END_DATE
    )

    risk_records = TechAna.calculate_risk_metrics_batch(symbols_to_analyze, stock_data)
    risk_map = {r['symbol']: r for r in risk_records}
    
    for sym in symbols_to_analyze:
        df = stock_data.get(sym)
        if df is not None and len(df) > 0:
            close = df['close'].dropna()
            if len(close) >= 20:
                # Kiểm tra nếu có total_value column (chính xác nhất)
                if 'total_value' in df.columns:
                    lookback_20d = min(20, len(df))
                    total_value_20d = df['total_value'].iloc[-lookback_20d:]
                    adv_20 = np.mean(total_value_20d)
                    if sym in risk_map:
                        risk_map[sym]['adv_20_abs'] = adv_20
                else:
                    volume_col = None
                    for col in ['total_volume', 'volume', 'Volume', 'vol']:
                        if col in df.columns:
                            volume_col = col
                            break
                    
                    if volume_col is not None:
                        lookback_20d = min(20, len(df))
                        prices_20d = close.iloc[-lookback_20d:]
                        volume_20d = df[volume_col].iloc[-lookback_20d:]
                        daily_value = prices_20d.values * volume_20d.values
                        adv_20 = np.mean(daily_value)
                        if sym in risk_map:
                            risk_map[sym]['adv_20_abs'] = adv_20
    
    ADV_20_MIN_MATERIALS = 5e9

    fund_map = dict(zip(df_final['Symbol'], df_final['Total_Score']))

    final_rows = []
    for sym in symbols_to_analyze:
        f_score_raw = fund_map.get(sym, 0)                  
        t_score_raw = tech_scores.get(sym, {}).get('score', 0)    
        fo_score_raw = foreign_scores.get(sym, {}).get('score', 0) 
        r_score_raw = risk_scores.get(sym, {}).get('score', 0)
        
        if sym in risk_map:
            adv_20 = risk_map[sym].get('adv_20_abs', 0)
            if adv_20 < ADV_20_MIN_MATERIALS:
                r_score_raw = 0
        else:
            r_score_raw = 0    
        
        final_rows.append({
            'Symbol': sym,
            'Fund_Score': f_score_raw/20.0, 
            'Tech_Score': t_score_raw/5.0,      
            'Foreign_Score': fo_score_raw/5.0,  
            'Risk_Score': r_score_raw/5.0       
        })

    df_total = pd.DataFrame(final_rows)
    
    W_FUND = 0.25
    W_TECH = 0.35
    W_FOREIGN = 0.25
    W_RISK = 0.15

    df_total['Final_Score'] = (
        W_FUND * (df_total['Fund_Score']) +
        W_TECH * (df_total['Tech_Score']) +
        W_FOREIGN * (df_total['Foreign_Score']) +
        W_RISK * (df_total['Risk_Score'])
    )

    # Sắp xếp và xếp hạng
    df_total = df_total.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
    df_total['Rank'] = df_total.index + 1

    # Làm tròn số hiển thị
    cols_to_round = ['Final_Score']
    df_total[cols_to_round] = df_total[cols_to_round].round(2)

    # Đổi tên cột hiển thị cho đẹp
    display_df = df_total[[
        'Rank', 'Symbol', 
        'Fund_Score',
        'Tech_Score',
        'Foreign_Score', 
        'Risk_Score', 
        'Final_Score'
    ]]

    print("Summary Score for Basic Materials:")
    
else:
    print("Không có dữ liệu để tổng hợp.")

# %%
print(display_df.to_string(index=False))