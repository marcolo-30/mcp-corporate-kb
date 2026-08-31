"""
agent_client/config.py

Configuración del cliente. Todo lo que depende del entorno vive aquí,
igual que en mcp_server/config.py.
"""

from __future__ import annotations

import os
import sys

# --- Modelo de Ollama para el razonamiento del agente ---
# Modelos confirmados disponibles en este proyecto: qwen3.5,
# qwen2.5-coder:7b-instruct-q4_K_..., qwen2.5:7b.
# qwen3.5 por defecto: es el más capaz de los tres para tool-calling +
# generación de respuestas en lenguaje natural (el "-coder" está afinado
# para código, no para este caso de uso; qwen2.5:7b es la alternativa
# más liviana si qwen3.5 es demasiado lento en tu hardware).
OLLAMA_MODEL = os.environ.get("GROUNDEDKB_OLLAMA_MODEL", "qwen3.5")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Temperatura baja a propósito: este agente debe responder en base a
# fragmentos citados, no "creativamente".
OLLAMA_TEMPERATURE = float(os.environ.get("GROUNDEDKB_OLLAMA_TEMPERATURE", "0.1"))

# --- Cómo lanzar el servidor MCP como subproceso stdio ---
# Por defecto se asume que agent_client/ vive junto a mcp_server/ en la
# raíz del repo y que el cliente se ejecuta desde ahí (python -m
# agent_client.agent). Si necesitas otro intérprete/ruta, sobreescribe
# estas dos variables.
MCP_SERVER_COMMAND = os.environ.get("GROUNDEDKB_MCP_COMMAND", sys.executable)
MCP_SERVER_ARGS = os.environ.get("GROUNDEDKB_MCP_ARGS", "-m mcp_server.server").split()

# --- Rol con el que este cliente llama a las tools del servidor MCP ---
# Ver mcp_server/guardrails.py para la matriz de permisos por rol.
MCP_CLIENT_ROLE = os.environ.get("GROUNDEDKB_CLIENT_ROLE", "employee")
