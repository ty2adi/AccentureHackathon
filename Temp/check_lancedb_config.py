import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import os
from dotenv import load_dotenv

load_dotenv()

# Define the embedding model
# Note: We use 'text-embedding-3-small' which produces 1536-dimensional vectors.
openai_embed = get_registry().get("openai").create(
    name="text-embedding-3-small",
    api_key=os.environ.get("open_ai_api_key")
)

class HospitalRecord(LanceModel):
    # original_csv_index helps us link back to the source or full JSON
    vector: Vector(openai_embed.ndims()) = openai_embed.VectorField()
    original_csv_index: int
    name: str = openai_embed.SourceField()
    address_city: str
    address_country: str
    # We store the full JSON as a string so the agent can read every detail
    full_metadata: str

def test_lancedb_connection():
    try:
        db = lancedb.connect("Data/vector_db")
        print("Successfully connected to LanceDB at Data/vector_db")
        print(f"Embedding dimensions: {openai_embed.ndims()}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_lancedb_connection()
