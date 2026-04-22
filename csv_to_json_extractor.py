# =============================================================================
# Databricks IDP Agent — CSV TO JSON EXTRACTOR
# =============================================================================
# This script handles the Phase 1 extraction of the IDP architecture.
# It processes ALL 988 rows of the complete CSV dataset through both
# LLM extraction pipelines (Task A: Demographics, Task B: Medical Intelligence).
#
# Estimated cost: ~$0.60 USD for all 988 rows on gpt-4.1-nano.
# Features:
#   - 5 concurrent threads for balanced TPM usage
#   - Wave system: Automatic retry and cooldown for rate limits
#   - Checkpoint/resume system: safe to stop and restart mid-run
#   - Saves final output to Data/full_extraction.json
# =============================================================================

import os
import sys
import json
import time
import threading
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import RateLimitError, APIStatusError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Load the open_ai_api_key from our local .env file securely
load_dotenv()

# Add the Databricks official hackathon folder to our script path
# so we can import their precise Python files without copying/pasting them.
sys.path.append(os.path.join(os.getcwd(), 'Prompts_and_Pydantic_Models'))

# Import from FREE_FORM.PY: Handles unstructured medical equipment, procedures, and capabilities
from prompts_and_pydantic_models.free_form import FacilityFacts, FREE_FORM_SYSTEM_PROMPT

# Import from FACILITY_AND_NGO_FIELDS.PY: Handles strict name extraction and address splitting
from prompts_and_pydantic_models.facility_and_ngo_fields import Facility, ORGANIZATION_INFORMATION_SYSTEM_PROMPT

print("Libraries and schemas loaded successfully!")

# =============================================================================
# 1. Configure the AI Engines (LLMs & Prompts)
# =============================================================================
# gpt-4.1-nano: OpenAI's cheapest model at $0.10 input / $0.40 output per 1M tokens.
# Excellent at structured JSON extraction. Temperature is 0 so it doesn't hallucinate!
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=os.environ.get("open_ai_api_key"), temperature=0)

# TASK A FRAMEWORK: Focuses on demographics, operator types, and correctly parsing addresses out of the name.
prompt_fields = ChatPromptTemplate.from_messages([
    ("system", ORGANIZATION_INFORMATION_SYSTEM_PROMPT),
    ("human", "Raw content block:\n\n{text}")
])
runnable_fields = prompt_fields | llm.with_structured_output(schema=Facility)

# TASK B FRAMEWORK: Focuses strictly on free-form medical capabilities and equipment
prompt_facts = ChatPromptTemplate.from_messages([
    ("system", FREE_FORM_SYSTEM_PROMPT),
    ("human", "Raw content block:\n\n{text}")
])
runnable_facts = prompt_facts | llm.with_structured_output(schema=FacilityFacts)

print("AI Runnables configured and ready.")

# =============================================================================
# 2. Load the COMPLETE CSV Dataset (All 988 Rows)
# =============================================================================
CSV_PATH        = "Data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
OUTPUT_PATH     = os.path.join("Data", "full_extraction.json")
CHECKPOINT_PATH = os.path.join("Data", "full_extraction_checkpoint.json")
CHECKPOINT_EVERY = 25   # Save progress to disk every 25 completed rows

print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Process the ENTIRE dataset
target_df = df.copy()

print(f"Total rows in CSV: {len(target_df)}")
print(f"Estimated cost: ~$0.60 USD at gpt-4.1-nano rates.")

# Load existing checkpoint if it exists
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        all_extracted_results = json.load(f)
    already_processed = {r['original_csv_index'] for r in all_extracted_results}
    remaining = len(target_df) - len(already_processed)
    print(f"Resuming from checkpoint: {len(already_processed)} rows already done, {remaining} remaining.")
else:
    all_extracted_results = []
    already_processed = set()
    print("Starting fresh extraction.")

# =============================================================================
# 3. The Grand Extraction Loop (Concurrent Threading)
# =============================================================================
relevant_columns = [
    'description', 'missionStatement', 'organizationDescription', 'procedure',
    'equipment', 'capability', 'specialties', 'websites', 'address_line1', 'name'
]

results_lock = threading.Lock()

@retry(
    retry=retry_if_exception_type((RateLimitError, APIStatusError)),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    stop=stop_after_attempt(4),
    reraise=True
)
def process_single_hospital(index, raw_organization_context, combined_context):
    """
    Processes a single hospital row through both extraction pipelines.
    """
    print(f"[{index}] {raw_organization_context[:45]}...")
    try:
        fields_result = runnable_fields.invoke({"organization": raw_organization_context, "text": combined_context})
        facts_result  = runnable_facts.invoke({"organization": raw_organization_context, "text": combined_context})

        parsed_fields = json.loads(fields_result.model_dump_json()) if fields_result else {}
        parsed_facts  = json.loads(facts_result.model_dump_json())  if facts_result  else {}

        merged_result = {**parsed_fields, **parsed_facts}
        merged_result['original_csv_index'] = index

        return merged_result
    except Exception as e:
        print(f" -> Rate limit wave-fail [{index}]: {e}")
        return None

rows_to_process = [
    (index, row) for index, row in target_df.iterrows()
    if index not in already_processed
]

RATE_LIMIT_COOLDOWN = 60
wave = 1

while rows_to_process:
    print(f"\nWave {wave}: Launching {len(rows_to_process)} rows across 5 parallel threads...\n")
    failed_indices = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for index, row in rows_to_process:
            raw_organization_context = str(row.get('name', 'Unknown'))
            context_dict     = {col: str(row[col]) if pd.notna(row.get(col)) else None for col in relevant_columns if col in df.columns}
            combined_context = json.dumps(context_dict, indent=2)
            futures[executor.submit(process_single_hospital, index, raw_organization_context, combined_context)] = index

        for future in concurrent.futures.as_completed(futures):
            row_index = futures[future]
            result    = future.result()
            if result is None:
                failed_indices.append(row_index)
            else:
                with results_lock:
                    all_extracted_results.append(result)
                    if len(all_extracted_results) % CHECKPOINT_EVERY == 0:
                        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                            json.dump(all_extracted_results, f, indent=4)
                        print(f"Checkpoint saved: {len(all_extracted_results)}/{len(target_df)}")

    if failed_indices:
        print(f"Wave {wave} complete. {len(failed_indices)} failed — cooling down...")
        time.sleep(RATE_LIMIT_COOLDOWN)
        rows_to_process = [(idx, target_df.loc[idx]) for idx in failed_indices]
        wave += 1
    else:
        rows_to_process = []

all_extracted_results.sort(key=lambda x: x.get('original_csv_index', 0))

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(all_extracted_results, file, indent=4)

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print(f"\nPipeline complete! Saved to {OUTPUT_PATH}")
