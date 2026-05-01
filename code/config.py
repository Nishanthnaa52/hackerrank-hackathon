"""
Centralized configuration for the Support Triage Agent.
All paths, model names, and tunables live here.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUPPORT_TICKETS_DIR = BASE_DIR / "support_tickets"

INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"

# ── Local RAG Index (FAISS) ────────────────────────────────────────────
INDEX_DIR = DATA_DIR / "index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3

# ── Local LLM (Ollama) ─────────────────────────────────────────────────
# Make sure you run `ollama run llama3` in your terminal first.
LOCAL_LLM = os.getenv("LOCAL_LLM", "llama3")

# ── LLM ────────────────────────────────────────────────────────────────
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# ── Domains ────────────────────────────────────────────────────────────
DOMAINS = ["hackerrank", "claude", "visa"]
