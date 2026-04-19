import pandas as pd
import json

df = pd.read_csv('Data/Virtue Foundation Ghana v0.3 - Sheet1.csv')

relevant_columns = ['name', 'description', 'address_line1', 'address_city', 'address_country', 'specialties', 'procedure', 'equipment']

print(f"=== FIRST 8 ROWS (CSV indices 0-7) ===\n")
for idx, row in df.head(8).iterrows():
    print(f"--- Row {idx} ---")
    for col in relevant_columns:
        val = row.get(col)
        if pd.notna(val):
            print(f"  {col}: {str(val)[:120]}")
    print()
