# =============================================================================
# Databricks IDP Agent — PHASE 3: CLINICAL REASONING AGENT (CLI)
# =============================================================================
# This is the final production reasoning agent for the IDP pipeline.
# Usage: python reasoning_agent.py "Your medical question here"
# =============================================================================

import os
import sys
import json
import argparse
import lancedb
from typing import TypedDict, List
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

# Load secrets
load_dotenv()
client = OpenAI(api_key=os.environ.get("open_ai_api_key"))

DB_PATH = "Data/medical_intelligence_db"
TABLE_NAME = "hospitals"

class AgentState(TypedDict):
    original_query: str
    search_query: str
    retrieved_results: List[dict]
    final_report: str

def analyze_query(state: AgentState):
    print(f" -> [ANALYZER] Identifying intent for: '{state['original_query']}'")
    prompt = f"Convert to dense Keywords (FACILITY | MEDICAL | LOCATION). Question: {state['original_query']}"
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"search_query": response.choices[0].message.content.strip()}

def retrieve_contexts(state: AgentState):
    print(f" -> [RETRIEVER] Searching Knowledge Base for: '{state['search_query']}'")
    response = client.embeddings.create(input=[state['search_query']], model="text-embedding-3-large")
    query_vector = response.data[0].embedding
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(TABLE_NAME)
    results = tbl.search(query_vector).limit(5).to_list()
    contexts = [json.loads(r['full_metadata']) for r in results]
    return {"retrieved_results": contexts}

def generate_report(state: AgentState):
    print(f" -> [REASONER] Reasoning over {len(state['retrieved_results'])} results...")
    context_str = json.dumps(state["retrieved_results"], indent=2)
    prompt = f"Answer the question: {state['original_query']} using this data: {context_str}. Include [CSV Index: N] citations."
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"final_report": response.choices[0].message.content}

def build_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("retrieve", retrieve_contexts)
    workflow.add_node("respond", generate_report)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, nargs='?', help="Your medical question")
    args = parser.parse_args()
    
    user_query = args.query
    if not user_query:
        print("\nVIRTUE FOUNDATION CLINICAL AGENT")
        user_query = input("QUESTION: ").strip()
    
    if not user_query: return

    agent = build_agent()
    final_state = agent.invoke({"original_query": user_query, "search_query": "", "retrieved_results": [], "final_report": ""})
    
    print("\n" + "="*60)
    print(" FINAL CLINICAL REPORT")
    print("="*60 + "\n")
    print(final_state["final_report"])

if __name__ == "__main__":
    main()
