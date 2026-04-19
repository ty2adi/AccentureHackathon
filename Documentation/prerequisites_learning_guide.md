# Machine Learning Prerequisites & Tooling Guide

To successfully build the Intelligent Document Parsing (IDP) Agent for this challenge, you don't need to learn every AI concept, but you do need targeted, practical knowledge in a few specific domains. Here is exactly what you should understand and the tools you will use.

## 1. Information Extraction & Structured Outputs
Your first major MVP task is pulling messy unstructured text into strict schemas.

*   **What to understand:**
    *   **Prompt Engineering for Data Extraction**: How to ask an LLM to find specific needle-in-a-haystack items.
    *   **Structured Outputs**: Forcing an LLM to return valid JSON rather than chatty conversational text.
*   **Specific Tools / Libraries to Learn**:
    *   `pydantic`: This is crucial. Pydantic is a Python library used to define data structures (like the *Procedure*, *Equipment*, and *Facility* schemas). Modern LLM wrappers use Pydantic models to strictly enforce the output shape.
    *   `instructor` or LangChain's `with_structured_output`: Libraries specifically designed to map LLM text generation directly into Pydantic models.

## 2. RAG (Retrieval-Augmented Generation)
Once you have structured your data, you need the AI system to be able to search through it dynamically.

*   **What to understand:**
    *   **Embeddings & Embedding Models**: How text is converted into math/vectors so the machine understands semantic meaning (e.g., matching "broken bone doctor" to "orthopedic surgery").
    *   **Chunking Strategy**: How to break down large documents so you don't overwhelm the LLM's context window.
    *   **Semantic Search vs. Hybrid Search**: Searching by meaning (vectors) vs. searching by exact keywords (BM25 algorithms).
*   **Specific Tools / Libraries to Learn**:
    *   **Vector Databases**: 
        *   `LanceDB` or `FAISS`: Local, easy-to-use vector databases to store your structured text and embeddings. 
        *   *Databricks Vector Search* (if utilizing the Databricks ecosystem).
    *   **Embedding Models**: e.g., OpenAI's `text-embedding-3-small` or open-source HuggingFace models like `BGE` or `MiniLM`.

## 3. Agentic Workflows (LangChain / LangGraph)
Agents differ from standard chatbots because they have access to "Tools" and can loop, plan, and make decisions dynamically before returning an answer.

*   **What to understand:**
    *   **Tool Calling / Function Calling**: How an LLM is taught to execute Python functions (like searching a database, performing math, or checking a map).
    *   **Stateful Graphs & Routing**: Understanding that an agent is a state-machine that routes decisions based on the current context.
*   **Specific Tools / Libraries to Learn**:
    *   **LangGraph**: The current industry standard for building robust, multi-agent workflows. It allows you to define cycles (loops) and maintain application "state" across an agent's reasoning process. (Highly recommended over basic standard LangChain for agents).
    *   **CrewAI** (Alternative): An alternative framework focused on giving "roles" to different agents (e.g., an "Extraction Agent" and a "Planning Agent") and letting them talk to each other.

## 4. Working with Structured Data (Text-to-SQL)
Because you have a structured schema of regions, operator types, numbers of doctors, etc., you'll need the agent to perform analytics over this.

*   **What to understand:**
    *   **Text2SQL / Text-to-Pandas**: How to formulate an agent that converts English queries ("How many Level II trauma centers are in Ghana?") into SQL or Pandas operations.
*   **Specific Tools / Libraries to Learn**:
    *   `pandas` / `polars`: The standard Python libraries for data table manipulation.
    *   *Databricks Genie* (optional but mentioned in the challenge): A Databricks feature for natural-language-to-SQL workflows.

## 5. Experiment Tracking (For the "Citations" Stretch Goal)
To trace *why* the agent made a decision, logging its steps is critical.

*   **What to understand:**
    *   **Tracing / Observability**: Keeping track of what tools the agent called, what data it retrieved, and how much time/tokens it consumed.
*   **Specific Tools / Libraries to Learn**:
    *   **MLflow**: Often deeply integrated into Databricks. MLflow Tracing allows you to record the inputs and outputs of every LangGraph node (providing the row-level citation stretch goal).
    *   *LangSmith*: The native LangChain observability tool for debugging agents rapidly.

---

### Suggested Learning Path to Start Today:
1.  **Skip the basic LangChain tutorials** and go straight to **Pydantic + LLM Structured Outputs** tutorials. Learn how to map text to a Python class.
2.  Follow a tutorial purely on **LangGraph** (look for "LangGraph Tool Calling Agent" examples). You can build almost the entire challenge inside LangGraph.
3.  Play around with **FAISS** or **LanceDB** locally by feeding it a few dummy facility strings and searching them.
