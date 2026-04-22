# =============================================================================
# Databricks IDP Agent — PHASE 2: LANCEDB INDEXER
# =============================================================================
# This script converts the extracted JSON data into a high-performance
# Vector Database (LanceDB) using OpenAI's 'text-embedding-3-large' model.
# =============================================================================

import os
import json
import lancedb
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Pathing
INPUT_JSON = "Data/full_extraction.json"
DB_PATH = "Data/medical_intelligence_db"
TABLE_NAME = "hospitals"

# Configure OpenAI Client
client = OpenAI(api_key=os.environ.get("open_ai_api_key"))
EMBEDDING_MODEL = "text-embedding-3-large"  # 3072 dimensions

def build_rich_summary(record):
    """
    Creates a structured string specifically designed for the embedding model.
    Includes address, contact, and medical fields so the vector captures
    all searchable dimensions of a facility.
    """
    name = record.get('name', 'Unknown')
    
    # Full address chain (street → city → region → country)
    addr_parts = [
        record.get('address_line1', ''),
        record.get('address_line2', ''),
        record.get('address_city', ''),
        record.get('address_stateOrRegion', ''),
        record.get('address_country', '')
    ]
    full_address = ", ".join([p for p in addr_parts if p and p != 'None'])
    
    # Contact info
    phone = record.get('officialPhone', '') or ''
    email = record.get('email', '') or ''
    
    # Medical fields
    specialties = record.get('specialties', [])
    procedures = record.get('procedure', [])
    equipment = record.get('equipment', [])
    
    spec_str = ", ".join(specialties) if isinstance(specialties, list) else str(specialties)
    proc_str = ", ".join(procedures) if isinstance(procedures, list) else str(procedures)
    equip_str = ", ".join(equipment) if isinstance(equipment, list) else str(equipment)
    
    summary = (
        f"FACILITY: {name} | "
        f"ADDRESS: {full_address} | "
        f"CONTACT: {phone} {email} | "
        f"MEDICAL SPECIALTIES: {spec_str} | "
        f"PROCEDURES: {proc_str} | "
        f"EQUIPMENT: {equip_str} | "
        f"DESCRIPTION: {record.get('description', '')}"
    )
    return summary

def generate_knowledge_base():
    if not os.path.exists(INPUT_JSON):
        print(f"[ERR] {INPUT_JSON} not found. Run extractor first.")
        return

    print(f"[DATA] Loading records from {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    db = lancedb.connect(DB_PATH)
    BATCH_SIZE = 50
    final_data_to_index = []
    
    print(f"[AI] Generating embeddings using {EMBEDDING_MODEL}...")
    for i in tqdm(range(0, len(all_records), BATCH_SIZE)):
        batch = all_records[i : i + BATCH_SIZE]
        summaries = [build_rich_summary(r) for r in batch]
        
        try:
            response = client.embeddings.create(input=summaries, model=EMBEDDING_MODEL)
            vectors = [item.embedding for item in response.data]
            
            for j, record in enumerate(batch):
                final_data_to_index.append({
                    "vector": vectors[j],
                    "id": record.get("original_csv_index"),
                    "name": record.get("name"),
                    "search_summary": summaries[j],
                    "full_metadata": json.dumps(record)
                })
        except Exception as e:
            print(f"[ERR] Batch failure at index {i}: {e}")
            continue

    print(f"[DB] Saving {len(final_data_to_index)} records to '{TABLE_NAME}'...")
    tbl = db.create_table(TABLE_NAME, data=final_data_to_index, mode="overwrite")
    
    print("\n[SUCCESS] PHASE 2 COMPLETE!")
    print(f"Path: {DB_PATH}")
    print(f"Items: {len(tbl)}")
    print(f"Dimensions: {len(vectors[0])}")

if __name__ == "__main__":
    generate_knowledge_base()
