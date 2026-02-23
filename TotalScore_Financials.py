# %%
import pandas as pd
import TechAna_DRAFT as TechAna
import RiskControl_Fin as RiskControl
from TechAna_DRAFT import START_DATE, END_DATE

try:
    from Fund_Financials import combined_scores_draft as df_final
    print("Success")
except ImportError:
    print("ERROR")
    df_final = pd.DataFrame(columns=['Symbol', 'Total_Score'])

if not df_final.empty:
    symbols_to_analyze = df_final['Symbol'].tolist()
else:
    symbols_to_analyze = []
    print("There's no stock")


def _derive_year_quarter(end_date):
    end_dt = pd.to_datetime(end_date, errors='coerce')
    if pd.isna(end_dt):
        return None, None
    period = end_dt.to_period('Q')
    return int(period.year), int(period.quarter)

# %%
def get_total_score(start_date, end_date, industry='Financials', year=None, quarter=None):
    if year is None or quarter is None:
        year, quarter = _derive_year_quarter(end_date)

    if not symbols_to_analyze:
        return pd.DataFrame(columns=[
            'Rank', 'Symbol', 'Fund_Score', 'Tech_Score',
            'Foreign_Score', 'Risk_Score', 'Final_Score'
        ])

    tech_scores, foreign_scores, risk_scores = TechAna.analyze_symbols(
        symbols_to_analyze,
        start_date,
        end_date,
        industry=industry,
        year=year,
        quarter=quarter
    )

    fund_map = dict(zip(df_final['Symbol'], df_final['Total_Score']))
    
    # Use risk_scores from RiskControl_Fin
    stock_data = TechAna.APIFetcher.fetch_batch(
        "http://192.168.8.190:8000/MKD/stock_daily", 
        symbols_to_analyze, START_DATE, END_DATE
    )
    risk_records = RiskControl.calculate_risk_metrics_batch(symbols_to_analyze, stock_data)
    risk_scorer = RiskControl.RiskControlScorer()
    risk_scores = {}
    for rec in risk_records:
        risk_scores[rec['symbol']] = risk_scorer.score_symbol(rec['symbol'], rec, risk_records)
    
    final_rows = []
    for sym in symbols_to_analyze:
        f_score = fund_map.get(sym, 0)                  
        t_score = tech_scores.get(sym, {}).get('score', 0)    
        fo_score = foreign_scores.get(sym, {}).get('score', 0) 
        r_score = risk_scores.get(sym, {}).get('score', 0)    
        
        final_rows.append({
            'Symbol': sym,
            'Fund_Score': f_score/20.0, 
            'Tech_Score': t_score/5.0,      
            'Foreign_Score': fo_score/5.0,  
            'Risk_Score': r_score/5.0       
        })

    df_total = pd.DataFrame(final_rows)
    
    W_FUND = 0.3
    W_TECH = 0.4
    W_FOREIGN = 0.15
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
    return display_df

# %%
if __name__ == "__main__":
    if symbols_to_analyze:
        display_df = get_total_score(START_DATE, END_DATE, industry='Financials')
        print(display_df.to_string(index=False))
    else:
        print("No data validated for analysis.")