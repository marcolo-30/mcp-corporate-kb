"""
mcp_server/server.py

Punto de entrada del servidor MCP. Corre por stdio (transporte por
defecto y más estándar del SDK — es lo que espera cualquier cliente
MCP, incluyendo Claude Desktop/Code, sin configuración extra).

Uso:
    python -m mcp_server.server

*** NOTA DE COMPATIBILIDAD ***
El SDK oficial de MCP renombró `FastMCP` (mcp<2) a `MCPServer` (mcp>=2)
en una versión reciente, manteniendo la misma API de decoradores
(`@mcp.tool()`) y `mcp.run()`. Este shim intenta ambas para que el
servidor funcione sin importar cuál tengas instalada — revisa tu
`pyproject.toml` / lockfile para saber cuál es en tu caso.
"""

from __future__ import annotations

try:
    from mcp.server.fastmcp import FastMCP  # mcp < 2.x
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp >= 2.x

from ingestion.vectorstore import VectorStore

from mcp_server import config
from mcp_server.embedder_factory import build_embedder
from mcp_server.tools.buscar_politica import buscar_politica as _buscar_politica
from mcp_server.tools.citar_fuente import citar_fuente as _citar_fuente
from mcp_server.tools.listar_documentos import listar_documentos as _listar_documentos
from mcp_server.tools.resumir_documento import resumir_documento as _resumir_documento

mcp = FastMCP(
    name="groundedkb",
    instructions=(
        "Servidor MCP de base de conocimiento corporativa. Todas las "
        "respuestas de buscar_politica y citar_fuente están ancladas a un "
        "documento y fragmento exacto — si una pregunta no tiene respaldo "
        "en el corpus, las tools lo reportan explícitamente en vez de "
        "inventar contenido. Usa buscar_politica primero; usa citar_fuente "
        "para verificar un chunk_id específico antes de presentarlo como cita."
    ),
)

# Un solo VectorStore compartido por todas las tool calls del proceso
# (Chroma es seguro para lecturas concurrentes desde un mismo cliente).
_vectorstore = VectorStore(
    embedder=build_embedder(),
    persist_path=config.CHROMA_PERSIST_PATH,
    collection_name=config.COLLECTION_NAME,
)


@mcp.tool()
def buscar_politica(
    query: str, top_k: int = 3, categoria: str | None = None, role: str | None = None
) -> dict:
    """
    Busca fragmentos relevantes en las políticas y contratos corporativos
    para responder una pregunta en lenguaje natural. Devuelve fragmentos
    citables con su documento y sección de origen, nunca una respuesta
    ya redactada — el llamador debe construir la respuesta final citando
    estos fragmentos, o reportar que no hay información si "encontrado"
    es false.
    """
    return _buscar_politica(_vectorstore, query, top_k=top_k, categoria=categoria, role=role)


@mcp.tool()
def citar_fuente(chunk_id: str, role: str | None = None) -> dict:
    """
    Recupera el fragmento exacto indexado bajo un chunk_id (obtenido de
    buscar_politica), para verificar que una cita es fiel al texto
    original antes de presentarla al usuario final.
    """
    return _citar_fuente(_vectorstore, chunk_id, role=role)


@mcp.tool()
def resumir_documento(doc_id: str, role: str | None = None) -> dict:
    """
    Devuelve un resumen por secciones de un documento completo del
    corpus, identificado por su doc_id (ver listar_documentos).
    """
    return _resumir_documento(doc_id, role=role)


@mcp.tool()
def listar_documentos(categoria: str | None = None, role: str | None = None) -> dict:
    """
    Lista los documentos disponibles en la base de conocimiento,
    opcionalmente filtrados por categoría (politicas, onboarding,
    contratos, faq, conducta), junto con cuántos fragmentos tiene cada
    uno indexados.
    """
    return _listar_documentos(_vectorstore, categoria=categoria, role=role)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
