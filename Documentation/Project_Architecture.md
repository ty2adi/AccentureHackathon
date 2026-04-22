# Project Architecture: IDP Agent for Medical Intelligence

This document details the architecture and data flow of the Intelligent Document Parsing (IDP) Agent built for the Databricks Accenture Challenge.

## Overview

The goal of this project is to transform unstructured healthcare facility data into a structured knowledge base that an AI agent can reason over. The architecture consists of three primary phases.

---

## Phase 1: Dual-Task Extraction (`csv_to_json_extractor.py`)

To minimize "hallucinations" and ensure strictly validated data, we split the extraction into two specialized LLM pipelines.

### 1. Structural Demographics (Task A)
- **Model**: OpenAI `gpt-4.1-nano`
- **Schema**: `Facility` (Pydantic)
- **Focus**: Extracts official names, operator types (public/private), and performs complex address splitting/cleaning.
- **Logic**: Strips location information that is often incorrectly embedded in the "Name" column of source CSVs.

### 2. Medical Intelligence (Task B)
- **Model**: OpenAI `gpt-4.1-nano`
- **Schema**: `FacilityFacts` (Pydantic)
- **Focus**: Parses procedures, medical equipment, and clinical capabilities.
- **Logic**: Acts as a "Chief Medical Officer" to identify specific infrastructure (e.g., MRI machines, HIV/AIDS programs) regardless of where they appear in the text.

### Implementation: The Wave Pipeline
The `csv_to_json_extractor.py` script manages this process with:
- **Parallelism**: 5 concurrent threads.
- **Resilience**: A "Wave" system that retries failed rows with exponential backoff and waits for TPM (Tokens Per Minute) window resets.
- **Checkpointing**: Saves progress every 25 rows to allow for seamless resumption after failure.

---

## Phase 2: Knowledge Base Indexing (`json_to_lancedb_indexer.py`)

The structured JSON output from Phase 1 is ingested into a Vector Database.

- **Storage**: LanceDB (local, high-performance columnar vector DB).
- **Embeddings**: OpenAI `text-embedding-3-large` producing 3072-dimensional vectors.
- **Rich Summary Strategy**: Each record is converted into a structured search string (`FACILITY | LOCATION | SPECIALTIES | EQUIPMENT | DESCRIPTION`) before embedding. This prioritizes medical "signals" over noise.
- **Indexing**: Allows for semantic search based on capabilities (e.g., "Find clinics that can treat TB in Accra").

---

## Phase 3: Reasoning Agent (`reasoning_agent.py`)

A multi-step reasoning system built using LangGraph processes complex user queries.

1. **Analyzer**: Translates human queries into dense medical keywords optimized for vector search.
2. **Retriever**: Embeds the optimized query and searches LanceDB for the top 5 matching facilities.
3. **Reasoner**: Evaluates the retrieved facilities and generates a natural language clinical report with `[CSV Index: N]` citations for data verification.

---

## Project Structure

```text
Accenture Project/
├── Data/                            # Datasets and extraction results
│   ├── Virtue Foundation Ghana...csv  # Source 988-row CSV
│   ├── full_extraction.json           # Phase 1 output (987 clinical records)
│   └── medical_intelligence_db/       # Phase 2 LanceDB vector database
├── Documentation/                   # Architecture and learning guides
├── Prompts_and_Pydantic_Models/     # Official hackathon schemas and prompts
├── Temp/                            # Intermediate test scripts and artifacts
├── csv_to_json_extractor.py         # Phase 1: Full 988-row extraction pipeline
├── json_to_lancedb_indexer.py       # Phase 2: JSON → 3072-dim LanceDB vectors
├── reasoning_agent.py               # Phase 3: Clinical reasoning agent (CLI)
├── requirements.txt                 # Project dependencies
└── README.md                        # Setup and collaboration guide
```
