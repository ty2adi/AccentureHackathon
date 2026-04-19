import lancedb
import pandas as pd

def inspect_vectors():
    # 1. Connect to the DB
    try:
        db = lancedb.connect("Data/medical_intelligence_db")
        tbl = db.open_table("hospitals")
    except Exception as e:
        print(f"[ERROR] Could not open database: {e}")
        return

    print(f"[OK] Successfully opened table 'hospitals'")
    print(f"Row count: {len(tbl)}")

    # 2. Extract a few rows as a Pandas DataFrame to 'see' the columns
    # We use to_pandas() to inspect the actual data
    df = tbl.to_pandas()
    
    print("\n--- SAMPLE DATA (First 3 rows) ---")
    # We'll print the name and a snippet of the vector
    for idx, row in df.head(3).iterrows():
        vec = row['vector']
        print(f"\n[Row {idx}] Name: {row['name']}")
        print(f"Vector (Type: {type(vec)}, Length: {len(vec)})")
        # Print just the first 5 numbers of the 3072 dimensions
        print(f"Vector Preview (first 5 dims): {vec[:5]}")

if __name__ == "__main__":
    inspect_vectors()
