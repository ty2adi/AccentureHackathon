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
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the secret GEMINI_API_KEY from our local .env file securely
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
# Configure our Gemini model to use the most recent 2.5 flash API. Temperature is 0 so it doesn't hallucinate!
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.environ.get("GEMINI_API_KEY"), temperature=0)

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
# ## 2. Load the CSV & Perform Intelligent Sampling
# We don't want to process the entire 800KB CSV during testing because it will burn API costs.
# We will grab 5 samples: 3 rows that are incredibly dense with information, and 2 rows that are almost totally blank/null.

# %%
csv_path = "Data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# Drop any hospital that doesn't have at least a basic description
df_with_desc = df.dropna(subset=['description']).copy()

# Score the rows based on how many columns actually have text in them
df_with_desc['non_null_count'] = df_with_desc.notna().sum(axis=1)

# Sort descending to find the 3 rows with the most data
dense_samples = df_with_desc.sort_values(by='non_null_count', ascending=False).head(3)

# Sort ascending to find the 2 rows with the least data (mostly nulls!)
sparse_samples = df_with_desc.sort_values(by='non_null_count', ascending=True).head(2)

# Combine them into a single test dataframe of 5 records
target_df = pd.concat([dense_samples, sparse_samples]).drop_duplicates()

print(f"Sampling queued: {len(dense_samples)} data-heavy controls, {len(sparse_samples)} sparse controls)")

# %% [markdown]
# ## 3. The Grand Extraction Loop
# This sweeps through the 5 selected rows. It collects ALL relevant unstructured text columns, 
# formats them as key-value JSON pairs (ignoring Pandas NaNs), and blasts the payload into our two LLM tasks.

# %%
all_extracted_results = []
# The 10 columns we identified as potentially containing medical facts or structural hints
relevant_columns = ['description', 'missionStatement', 'organizationDescription', 'procedure', 
                    'equipment', 'capability', 'specialties', 'websites', 'address_line1', 'name']

for index, row in target_df.iterrows():
    # We pass the raw name into the instruction so the AI knows its strict extraction bounds
    raw_organization_context = row.get('name', 'Unknown')
    
    # Pack up our context dynamically, explicitly placing JSON nulls if the cell is completely empty.
    context_dict = {}
    for col in relevant_columns:
        if col in df.columns:
            context_dict[col] = str(row[col]) if pd.notna(row.get(col)) else None
    
    # We serialize it to JSON since LLMs are natively brilliant at interpreting JSON payloads
    combined_context = json.dumps(context_dict, indent=2)
    print(f"\nProcessing CSV Index [{index}]: {raw_organization_context[:40]}...")
    
    try:
        # EXECUTE TASK A: Clean up the name, extract city, list the affiliations.
        fields_result = runnable_fields.invoke({
            "organization": raw_organization_context,
            "text": combined_context
        })
        
        # EXECUTE TASK B: Scrape the string for medical equipment, operations, etc.
        facts_result = runnable_facts.invoke({
            "organization": raw_organization_context,
            "text": combined_context
        })
        
        # Unpack the returned Pydantic objects safely back into dictionaries
        parsed_fields = json.loads(fields_result.model_dump_json()) if fields_result else {}
        parsed_facts = json.loads(facts_result.model_dump_json()) if facts_result else {}
        
        # Merge both dictionaries into a massive, final comprehensive JSON file
        merged_result = {**parsed_fields, **parsed_facts}
        merged_result['original_csv_index'] = index
        
        all_extracted_results.append(merged_result)
        print(" -> ✅ LLM Dual-Extraction successful.")
        
    except Exception as e:
        print(f" -> ❌ Failed to extract. Error: {e}")

# %% [markdown]
# ## 4. Compile and Save Results
# This pushes the final list of 5 deeply parsed hospitals back to the hard drive so the LangGraph agent can read it later.

# %%
output_path = os.path.join("Data", "extracted_sample.json")
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(all_extracted_results, file, indent=4)

print(f"\n✅ Pipeline complete! Saved formatted JSON to: {output_path}")
