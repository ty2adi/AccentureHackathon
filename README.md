# Databricks Accenture Challenge: Intelligent IDP Agent

This repository contains the **Intelligent Document Parsing (IDP) Agent** development for the Virtue Foundation project. The goal is to extract medical intelligence from massive, unstructured datasets to help NGO planners identify healthcare gaps.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+
- OpenAI API Key

### 2. Isolated Setup
1. **Initialize the Environment**: `python -m venv .venv`
2. **Activate it**: `.\.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac)
3. **Install Dependencies**: `pip install -r requirements.txt`

### 3. API Security
Create a `.env` file in the root and add your OpenAI key:
```env
open_ai_api_key="sk-proj-..."
```

---

## 🛠 Project Structure

This repo contains both **Production Pipelines** for the full 988-row dataset and **Skeleton Examples** for learning and rapid testing.

### 🏭 Production Pipeline (Phase 1 & 2)
- **`csv_to_json_extractor.py`**: The heavy-duty extractor for all 988 rows. Features Wave-based retries for rate limits and checkpointing.
- **`json_to_lancedb_indexer.py`**: Converts the extracted JSON into a 3072-dimensional LanceDB vector database.

### 🧠 Reasoning Agent (Phase 3)
- **`reasoning_agent.py`**: The production CLI agent. Run as `python reasoning_agent.py "Your Question"` or just run it to use interactive mode.

### 🔬 Skeleton & Examples (For Teammates)
- **`extraction_example.py`**: A 5-row sample extraction to test OpenAI connectivity.
- **`basic_agent_langgraph.py`**: A simple LangGraph skeleton showing how retrieval works.

---

## 📐 Run Instructions

1. **Full Extraction**: `python csv_to_json_extractor.py`
2. **Index Database**: `python json_to_lancedb_indexer.py`
3. **Run Agent**: `python reasoning_agent.py`

---

## 🏆 Development Roadmap
- [x] Migrate from Gemini to OpenAI (gpt-4.1-nano).
- [x] Implement Wave-based retry logic for TPM limits.
- [x] Full 988-row extraction pipeline.
- [x] **Phase 2: Indexing** — Build a Vector Database (LanceDB) with 3072-dim embeddings.
- [x] **Phase 3: Reasoning** — Fully functional LangGraph reasoning agent.

---

## ⚖️ License
Developed for the Databricks & Accenture IDP Agent Hackathon.
