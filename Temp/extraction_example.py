# %% [markdown]
# # Databricks IDP Agent Extraction Test
# This script represents Phase 1 of the architecture. We take raw, messy CSV data
# and run it through two different LLM extraction pipelines using the strict 
# Pydantic schema rules defined by the Databricks Hackathon organizers.

# %%
import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load the secret open_ai_api_key from our local .env file securely
load_dotenv()

# Add the Databricks official hackathon folder to our script path
# so we can import their precise Python files without copying/pasting them.
sys.path.append(os.path.join(os.getcwd(), 'Prompts_and_Pydantic_Models'))

# Import from FREE_FORM.PY: Handles unstructured medical equipment, procedures, and capabilities
from prompts_and_pydantic_models.free_form import FacilityFacts, FREE_FORM_SYSTEM_PROMPT

# Import from FACILITY_AND_NGO_FIELDS.PY: Handles strict name extraction and address splitting
from prompts_and_pydantic_models.facility_and_ngo_fields import Facility, ORGANIZATION_INFORMATION_SYSTEM_PROMPT

print("Libraries and schemas loaded successfully!")

# %% [markdown]
# ## 1. Configure the AI Engines (LLMs & Prompts)
# We define TWO distinct tasks here. 
# Databricks explicitly wants structural logic (finding addresses) separated from 
# medical extraction (finding MRI machines) to prevent the LLM from getting confused.

# %%
# gpt-4.1-nano: OpenAI's cheapest model at $0.10 input / $0.40 output per 1M tokens.
# 5x cheaper than gpt-3.5-turbo while being newer and better at structured JSON extraction.
# Temperature is 0 so it doesn't hallucinate!
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

# %% [markdown]
# ## 2. Load the Full CSV Dataset
# Estimated cost for the full 658-row dataset on gpt-4.1-nano: ~$0.39 USD.
# Well within the $4-5 budget. Let's process everything!
# 
# A checkpoint file is used so if the script crashes halfway, it resumes from where it
# left off — no re-spending money on rows already processed.

# %%
csv_path = "Data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
CHECKPOINT_PATH = os.path.join("Data", "full_extraction_checkpoint.json")

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# Drop any hospital that doesn't have at least a basic description (removes ~329 empty rows)
target_df = df.dropna(subset=['description']).copy()

print(f"Total processable rows (have descriptions): {len(target_df)}")
print(f"Estimated cost: ~$0.39 USD at gpt-4.1-nano rates.")

# Load existing checkpoint if it exists (so we can resume a crashed run!)
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        all_extracted_results = json.load(f)
    already_processed = {r['original_csv_index'] for r in all_extracted_results}
    print(f"♻️  Resuming from checkpoint: {len(already_processed)} rows already done, {len(target_df) - len(already_processed)} remaining.")
else:
    all_extracted_results = []
    already_processed = set()
    print("🆕 Starting fresh extraction.")

# %% [markdown]
# ## 3. The Grand Extraction Loop (Concurrent Threading)
# Instead of processing rows sequentially (which takes hours), we use a ThreadPoolExecutor.
# This blasts multiple rows to OpenAI simultaneously!
# OpenAI's rate limits are far more permissive than Gemini's free tier.
# max_workers=10 gives us strong parallelism. Increase to 20+ if you have a high-tier API plan.
# Results are checkpointed to disk every 25 rows so progress is never lost.

# %%
import concurrent.futures

import threading

# The 10 columns we identified as potentially containing medical facts or structural hints
relevant_columns = ['description', 'missionStatement', 'organizationDescription', 'procedure', 
                    'equipment', 'capability', 'specialties', 'websites', 'address_line1', 'name']

# Thread-safe lock for writing to the shared results list and checkpointing
results_lock = threading.Lock()
CHECKPOINT_EVERY = 25  # Save progress to disk every 25 completed rows
completed_count = 0

def process_single_hospital(index, raw_organization_context, combined_context):
    """Function to process a single hospital's dual-extraction, allowing us to run this on multiple threads!"""
    print(f"🚀 [{index}] {raw_organization_context[:45]}...")
    try:
        # EXECUTE TASK A & B simultaneously within this thread
        fields_result = runnable_fields.invoke({"organization": raw_organization_context, "text": combined_context})
        facts_result = runnable_facts.invoke({"organization": raw_organization_context, "text": combined_context})
        
        # Unpack safely
        parsed_fields = json.loads(fields_result.model_dump_json()) if fields_result else {}
        parsed_facts = json.loads(facts_result.model_dump_json()) if facts_result else {}
        
        merged_result = {**parsed_fields, **parsed_facts}
        merged_result['original_csv_index'] = index
        
        print(f" -> ✅ Done [{index}]")
        return merged_result
    except Exception as e:
        print(f" -> ❌ Failed [{index}]: {e}")
        return None

# Filter out rows we've already processed (checkpoint resume)
rows_to_process = [(index, row) for index, row in target_df.iterrows() if index not in already_processed]
print(f"\n🔄 Launching {len(rows_to_process)} rows across 10 parallel threads...\n")

# Launch the Thread Pool!
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = {}
    for index, row in rows_to_process:
        raw_organization_context = str(row.get('name', 'Unknown'))
        
        # Pack up our context dynamically
        context_dict = {col: str(row[col]) if pd.notna(row.get(col)) else None for col in relevant_columns if col in df.columns}
        combined_context = json.dumps(context_dict, indent=2)
        
        # Submit the task to the background threads
        futures[executor.submit(process_single_hospital, index, raw_organization_context, combined_context)] = index
        
    # As the threads finish their API calls, collect the results
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result:
            with results_lock:
                all_extracted_results.append(result)
                completed_count = len(all_extracted_results)
                
                # Checkpoint every N rows so we never lose progress
                if completed_count % CHECKPOINT_EVERY == 0:
                    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                        json.dump(all_extracted_results, f, indent=4)
                    total = len(target_df)
                    print(f"\n💾 Checkpoint saved: {completed_count}/{total} rows complete ({completed_count/total*100:.1f}%)\n")

# %% [markdown]
# ## 4. Compile and Save Final Results
# Save the complete master JSON and clean up the checkpoint file.

# %%
output_path = os.path.join("Data", "full_extraction.json")
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(all_extracted_results, file, indent=4)

# Clean up the checkpoint file now that we're fully done
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print(f"\n✅ Pipeline complete! Processed {len(all_extracted_results)} hospitals.")
print(f"📁 Master JSON saved to: {output_path}")
