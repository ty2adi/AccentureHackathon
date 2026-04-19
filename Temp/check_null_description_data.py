import pandas as pd
df = pd.read_csv('Data/Virtue Foundation Ghana v0.3 - Sheet1.csv')

no_desc = df[df['description'].isna()].copy()
relevant = ['missionStatement', 'organizationDescription', 'procedure', 'equipment', 'capability', 'specialties']

no_desc['other_data_count'] = no_desc[relevant].notna().sum(axis=1)

print(f"Rows WITHOUT description: {len(no_desc)}")
print(f"Of those, rows with ZERO other relevant data: {(no_desc['other_data_count'] == 0).sum()}")
print(f"Of those, rows with at least 1 other field:   {(no_desc['other_data_count'] >= 1).sum()}")
print(f"Of those, rows with 2+ other fields:          {(no_desc['other_data_count'] >= 2).sum()}")
print()
print("Breakdown by other_data_count:")
print(no_desc['other_data_count'].value_counts().sort_index())
