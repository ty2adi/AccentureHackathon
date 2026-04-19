import json
import pandas as pd

with open('Data/full_extraction.json', encoding='utf-8') as f:
    data = json.load(f)

df = pd.read_csv('Data/Virtue Foundation Ghana v0.3 - Sheet1.csv')
expected_indices = set(df.index)
found_indices = {r['original_csv_index'] for r in data}

missing = expected_indices - found_indices
all_indices = [r['original_csv_index'] for r in data]
dup_set = {x for x in all_indices if all_indices.count(x) > 1}

print(f"Total entries in JSON : {len(data)}")
print(f"Total rows in CSV     : {len(df)}")
print(f"Unique indices in JSON: {len(found_indices)}")
print(f"Missing indices       : {sorted(missing) if missing else 'None — all present!'}")
print(f"Duplicate indices     : {sorted(dup_set) if dup_set else 'None'}")
