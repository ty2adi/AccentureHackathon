import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

# ==========================================
# 1. Define the "State" of our Agent
# ==========================================
# LangGraph passes this State dictionary between different nodes (functions).
class AgentState(TypedDict):
    question: str
    context: list[str]  # The RAG documents retrieved from the vector DB
    answer: str

# ==========================================
# 2. Define the Nodes (The functions/steps)
# ==========================================
def retrieve_data(state: AgentState):
    """
    Simulates searching a Vector Database for the query using our new data.
    In reality, you would:
    1. Load `Data/extracted_sample.json` (our Phase 1 Master JSON) into LanceDB/FAISS.
    2. Convert `state['question']` to an embedding.
    3. Query the database to retrieve the top matching hospitals.
    """
    print("--- 🔍 STEP 1: RETRIEVING DATA FROM VECTOR DB ---")
    query = state["question"]
    
    # Mocking a search result based on the query that matches our dual-extraction schema
    mock_db_result = [
        "name: WAAF. capabilities: HIV/AIDS initiatives. operatorTypeId: public. city: Takoradi",
        "name: St. Jude. equipment: CT scanner. capacity: 12 beds"
    ]
    
    return {"context": mock_db_result}

def generate_answer(state: AgentState):
    """
    Takes the retrieved context and the user's question, and asks the LLM to answer.
    """
    print("--- 🧠 STEP 2: GENERATING ANSWER ---")
    question = state["question"]
    context = state["context"]
    
    # In reality, you'd call llm.invoke(prompt) here. We mock it for the skeleton:
    prompt = f"Answer the question: '{question}' using ONLY this context: {context}"
    
    # Simulating LLM response
    mock_llm_response = f"Based on the database, 1st Foundation Clinic is in Accra."
    
    return {"answer": mock_llm_response}

# ==========================================
# 3. Build the Graph
# ==========================================
# Initialize the graph
workflow = StateGraph(AgentState)

# Add our nodes (the steps)
workflow.add_node("retrieve", retrieve_data)
workflow.add_node("generate", generate_answer)

# Define the flow (edges)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile it into a runnable app
app = workflow.compile()

# ==========================================
# 4. Run the Agent
# ==========================================
if __name__ == "__main__":
    user_query = "Where is the nearest clinic in Accra?"
    
    print(f"USER: {user_query}\n")
    
    # Run the graph
    final_state = app.invoke({"question": user_query, "context": [], "answer": ""})
    
    print("\n--- FINAL OUTPUT ---")
    print(final_state["answer"])
