import json
import os
from dotenv import load_dotenv

# We won't run this to build the DB yet, just to TEST if we can load and see the data
def test_data_prep():
    with open('Data/full_extraction.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total records to index: {len(data)}")
    
    # Example of how we'll build the 'searchable_text'
    sample = data[0]
    search_text = f"Name: {sample.get('name')}. City: {sample.get('address_city')}. Specialties: {sample.get('specialties')}. Equipment: {sample.get('equipment')}"
    print(f"\nSample Search Text for Embedding:\n{search_text}")
    print(f"\nDimensions expected: 1536 (text-embedding-3-small)")

if __name__ == "__main__":
    test_data_prep()
