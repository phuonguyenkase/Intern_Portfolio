#%%
import TechAna
if not TechAna.tech_scores or len(TechAna.tech_scores) == 0:
    TechAna.run_analysis()

from Fund_Bank import combined_scores_draft
from TechAna import tech_scores, foreign_scores, risk_scores
import pandas as pd

#%%
# Build symbol universe across all sources
fund_map = dict(zip(combined_scores_draft['Symbol'], combined_scores_draft['Total_Score']))
symbols = sorted(set(fund_map.keys()) | set(tech_scores.keys()) | set(foreign_scores.keys()) | set(risk_scores.keys()))

# %%
#BANK
print("FINAL SUMMARY TABLE: BANK's TOTAL SCORE")
print("="*50)

rows = []
for sym in symbols:
    tech_val = tech_scores.get(sym, {}).get('score', 0)
    foreign_val = foreign_scores.get(sym, {}).get('score', 0)
    risk_val = risk_scores.get(sym, {}).get('score', 0)
    fund_val = fund_map.get(sym, 0)
    rows.append({
        'Symbol': sym,
        'Fundamental_Score': fund_val/20.0,
        'Tech_Score': tech_val/5.0,
        'Foreign_Score': foreign_val/5.0,
        'Risk_Score': risk_val/5.0
    })

total_df = pd.DataFrame(rows)

# Weighted final score (adjust weights as needed)
total_df['Final_Score'] = (
    0.25 * total_df['Fundamental_Score'] +
    0.35 * total_df['Tech_Score'] +
    0.2 * total_df['Foreign_Score'] +
    0.15 * total_df['Risk_Score']
)

# Rank by Composite Score
total_df['Rank'] = total_df['Final_Score'].rank(method='min', ascending=False).astype(int)

# Sort and round
total_df = total_df.sort_values(by='Final_Score', ascending=False).reset_index(drop=True)
for col in ['Fundamental_Score', 'Tech_Score', 'Foreign_Score', 'Risk_Score',  'Final_Score']:
    total_df[col] = total_df[col].round(2)

# Reorder columns
total_df = total_df[['Rank', 'Symbol', 'Fundamental_Score', 'Tech_Score', 'Foreign_Score', 'Risk_Score', 'Final_Score']]
print(total_df.to_string(index=False))
# %%
