"""
config.py

Configuración centralizada del servidor MCP. Todo lo que depende del
entorno (rutas, nombres de colección, umbrales) vive aquí para que
server.py y los tools no tengan valores mágicos dispersos.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Persistencia del índice vectorial (debe coincidir con ingestion/build_index.py) ---
CHROMA_PERSIST_PATH = os.environ.get("GROUNDEDKB_CHROMA_PATH", ".chroma_index")
COLLECTION_NAME = os.environ.get("GROUNDEDKB_COLLECTION", "corporate_kb")

# --- Corpus original en disco (para resumir_documento, que necesita el texto completo) ---
CORPUS_DIR = Path(os.environ.get("GROUNDEDKB_CORPUS_DIR", "corpus"))

# --- Umbral de faithfulness: si la distancia coseno del mejor chunk supera esto,
#     el sistema prefiere decir "no lo sé" antes que inventar una respuesta. ---
MAX_ACCEPTABLE_DISTANCE = float(os.environ.get("GROUNDEDKB_MAX_DISTANCE", "0.45"))

# --- Logging de auditoría estructurado (guardrails.py escribe aquí) ---
AUDIT_LOG_PATH = Path(os.environ.get("GROUNDEDKB_AUDIT_LOG", "logs/mcp_audit.log"))

# --- Rol por defecto cuando el cliente MCP no indica uno explícito ---
DEFAULT_ROLE = os.environ.get("GROUNDEDKB_DEFAULT_ROLE", "employee")

# --- Backend de embeddings (debe coincidir con el usado en build_index.py) ---
# Ver mcp_server/embedder_factory.py — "hashing" u "sentence-transformers".
EMBEDDER_BACKEND = os.environ.get("GROUNDEDKB_EMBEDDER_BACKEND", "hashing")
