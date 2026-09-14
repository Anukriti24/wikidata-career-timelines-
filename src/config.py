import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
# Dataset-specific paths (Pantheon's timelines/evaluation/final outputs, CVD's
# equivalents) live as local constants in each dataset's own pipeline/evaluate/finalize
# module (src/pipeline_pantheon.py, src/pipeline_cvd.py, etc.) rather than here - only
# genuinely shared/global paths belong in this file.
DATA_DIR = Path("data")
RAW_PANTHEON_PATH = DATA_DIR / "raw" / "pantheon.csv"

# --- Sampling ------------------------------------------------------------
# Configurable via env var so a small pilot run (e.g. SAMPLE_SIZE=30) can validate the
# whole chain before committing GPU time to the full 10,000-person run.
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 10000))

# --- Languages -------------------------------------------------------------
SUPPORTED_LANGS = ["en", "de"]

# --- Wikidata / Wikipedia -------------------------------------------------
CONTACT_EMAIL = "e12433727@student.tuwien.ac.at"
USER_AGENT = f"TUWien-OccupationTimelineBot/0.1 (student project; contact: {CONTACT_EMAIL})"

# --- Ollama / LLM ----------------------------------------------------------
# Configurable via env var so a pipeline run can reach an Ollama server running
# elsewhere (e.g. a remote GPU host) - defaults to localhost for local/single-machine
# runs.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.1:8b"  # change to "phi3:mini" if your laptop is slow
OLLAMA_TEMPERATURE = 0.2
OLLAMA_MAX_RETRIES = 3
OLLAMA_TIMEOUT_SECONDS = 300  # long bios / many occupations can take well over 120s on an 8B model

# --- Pipeline concurrency ---------------------------------------------------
# Number of people processed concurrently in src.timeline.process_people (each doing a
# Wikipedia fetch + an Ollama call). Configurable via env var so this can be matched to
# OLLAMA_NUM_PARALLEL on whichever Ollama server it's pointed at.
PIPELINE_MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", 8))
