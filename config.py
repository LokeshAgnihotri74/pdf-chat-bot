"""
Configuration and settings for the Smart AI Agent Assistant.
Dual-Component System: Advanced PDF RAG + Tool-Calling Agent.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Supported Groq Models (Free Tier on https://console.groq.com)
SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant"
]
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# RAG Configuration
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
TOP_K_RESULTS = 4

# Strict Hallucination Guardrail Message
STRICT_GUARDRAIL_FALLBACK = "I cannot find the answer in the provided document."

# ChromaDB Distance Metrics and Explanations
DISTANCE_METRICS = {
    "cosine": {
        "name": "Cosine Similarity (hnsw:space = 'cosine')",
        "description": (
            "Measures the angular cosine distance between document and query vectors. "
            "Because vectors are normalized, it is invariant to document length and token count. "
            "Best practice for semantic search and question answering."
        ),
        "formula": "Distance = 1 - (A · B) / (||A|| * ||B||)"
    },
    "l2": {
        "name": "Squared Euclidean Distance (hnsw:space = 'l2')",
        "description": (
            "Measures the straight-line geometric distance between vector coordinates in hyperspace. "
            "Sensitive to vector magnitudes and text length differences. "
            "Useful when magnitude of embedding carries semantic density information."
        ),
        "formula": "Distance = Σ (A_i - B_i)²"
    }
}
DEFAULT_DISTANCE_METRIC = os.getenv("DEFAULT_DISTANCE_METRIC", "cosine")

# Open-Meteo Weather API (100% Free, Zero API Key)
OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Rate Limiting & Retry Configuration
RETRY_MAX_ATTEMPTS = 3
RETRY_MIN_WAIT = 2
RETRY_MAX_WAIT = 10
