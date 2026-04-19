# Databricks Accenture Challenge: IDP Agent Master Documentation

## 1. Introduction
By 2030, the world will face a shortage of over 10 million healthcare workers. The **Virtue Foundation** tracks global medical infrastructure, but much of their data is trapped in messy, unstructured CSV datasets and medical notes. 

The goal of this project is to build an **Intelligent Document Parsing (IDP) Agent** capable of intelligently isolating medical capabilities from this unstructured text, structuring them cleanly, and then utilizing a reasoning engine to help NGO planners find "medical deserts".

---

## 2. Core Architecture: Dual-Task Extraction (Phase 1)
Instead of forcing the LLM to do everything at once (which leads to hallucinations), our extraction script (`extraction_example.py`) uses a Dual-LLM architecture, dynamically splitting the workload using official **Databricks Pydantic Models**:

### Task A: Structural Demographics (Using `Facility` Schema)
The CSV data is exceptionally messy (ex: sticking raw addresses inside the hospital name column). 
*   **The Engine**: We process row data through the `facility_and_ngo_fields.py` pipeline.
*   **The Result**: The LLM rigidly standardizes phone numbers, classifies the `operatorType` (public/private), and safely pulls city references out of the hospital name.

### Task B: Medical Intelligence (Using `FacilityFacts` Schema)
*   **The Engine**: We process the exact same text row through a completely different pipeline: `free_form.py`.
*   **The Result**: The LLM acts as a Chief Medical Officer, purely ignoring addresses and hunting specifically to populate three final arrays: `procedures`, `equipment`, and `capabilities` (e.g. searching for MRI machines, ICU capacity).

Both results are ultimately merged back together into a pristine Master JSON dictionary ready for Vector Database indexing.

---

## 3. Next Steps: Building the Knowledge Base & Agent (Phases 2 & 3)
1.  **Vector Database (RAG)**: The compiled Master JSON will be converted into embeddings and stored in LanceDB or FAISS. 
2.  **LangGraph Agent**: When an NGO planner queries *"Where is the nearest public Level II trauma center?"*, the LangGraph Agent (`basic_agent_langgraph.py`) searches the Vector Database, pulls the matching hospitals, and formulates a precise, geographically aware answer.

---

## 4. Local Codebase Setup & Collaboration Guide

This project is built for standard, secure Python engineering using Git. Follow these exact steps to pull this project and run it on your local machine.

### Core Startup Instructions
1.  **Clone the Repository**: Download this workspace to your machine via GitHub.
2.  **Initialize the Environment**: Do not use your global Python. Create an isolated workspace:
    ```bash
    python -m venv .venv
    ```
3.  **Activate the Environment**:
    *   *Windows (Powershell)*: `.\.venv\Scripts\activate`
    *   *Mac/Linux*: `source .venv/bin/activate`
4.  **Install Requirements**: Download the required math/AI packages.
    ```bash
    pip install -r requirements.txt
    ```

### API Security (CRITICAL)
**Never upload your `.env` file to GitHub!** The repository uses `python-dotenv` specifically to prevent API keys from leaking.
1. Create a local file in your main folder named exactly `.env`
2. Generate an API Key from Google AI Studio.
3. Paste the key inside your `.env` like this:
   `GEMINI_API_KEY="AIzaSy_YOUR_KEY_HERE"`
   
### Starter Files Directory
*   `extraction_example.py`: The Phase 1 processor. Loops our Pandas dataset through Gemini.
*   `basic_agent_langgraph.py`: The Phase 3 Agent skeleton.
*   `Data/`: Holds the raw CSV datasets and Markdown schema definitions.
*   `Prompts_and_Pydantic_Models/`: The official Databricks instructions and schemas.
