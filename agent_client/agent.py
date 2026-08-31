"""
agent_client/agent.py

Agente LangGraph que consume el servidor MCP como un cliente delgado
(el README lo describe así: "agent_client/ — thin consumer"). No sabe
nada de Chroma, embeddings, ni chunking — solo ve las 4 tools que el
servidor expone y decide cuándo llamarlas.

Arquitectura:
    Ollama (qwen3.5, local)  <--tool calls-->  LangGraph react agent
                                                       |
                                                       | stdio subprocess
                                                       v
                                          mcp_server/server.py (MCP)
                                                       |
                                                       v
                                        VectorStore (Chroma) + corpus/

Uso:
    python -m agent_client.agent
    python -m agent_client.agent --pregunta "cuantos dias de vacaciones acumulo"
    python -m agent_client.agent --role legal   # ver guardrails.py
"""

from __future__ import annotations

import argparse
import asyncio
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent

from agent_client import config
from agent_client.prompts import SYSTEM_PROMPT


def build_mcp_client() -> MultiServerMCPClient:
    """
    Configura la conexión stdio al servidor MCP. El servidor se lanza
    como subproceso (`python -m mcp_server.server`) y se comunica por
    stdin/stdout — no hay puerto de red que abrir.

    *** GOTCHA REAL, ENCONTRADO PROBANDO ESTO ***
    `StdioConnection` de langchain-mcp-adapters, si no le pasas `env`
    explícitamente, NO hereda el entorno completo del proceso padre —
    solo un subconjunto (ver su propio docstring). Eso significa que
    variables como GROUNDEDKB_CHROMA_PATH, GROUNDEDKB_CORPUS_DIR o
    GROUNDEDKB_EMBEDDER_BACKEND, si las exportas antes de correr este
    agente, NO llegarían al servidor MCP y este caería silenciosamente
    a sus valores por defecto (índice equivocado, sin dar error). Por
    eso aquí se pasa `env=dict(os.environ)` explícitamente.
    """
    return MultiServerMCPClient(
        {
            "groundedkb": {
                "transport": "stdio",
                "command": config.MCP_SERVER_COMMAND,
                "args": config.MCP_SERVER_ARGS,
                "env": dict(os.environ),
            }
        }
    )


async def load_mcp_tools(client: MultiServerMCPClient) -> list[BaseTool]:
    """
    Descubre las tools expuestas por el servidor MCP (buscar_politica,
    citar_fuente, resumir_documento, listar_documentos) y las envuelve
    como BaseTool de LangChain, listas para bind_tools()/create_react_agent.
    """
    return await client.get_tools()


def build_llm() -> BaseChatModel:
    """
    Modelo local vía Ollama. Temperatura baja a propósito: se espera
    que el modelo cite fragmentos recuperados, no que "complete
    creativamente" — ver agent_client/prompts.py para el contrato completo.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.OLLAMA_TEMPERATURE,
    )


def build_agent_graph(llm: BaseChatModel, tools: list[BaseTool]) -> CompiledStateGraph:
    """
    Agente ReAct estándar de LangGraph: el LLM decide en cada paso si
    llamar una tool o responder; el grafo se encarga del loop
    tool-call -> tool-result -> siguiente decisión del LLM.
    """
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def ask(
    graph: CompiledStateGraph, pregunta: str, role: str | None = None
) -> str:
    """
    Ejecuta una pregunta contra el agente y devuelve el texto final.

    `role` se antepone a la pregunta como contexto para que el LLM lo
    pase como argumento `role` en sus tool calls (ver guardrails.py en
    el servidor) — el protocolo MCP no tiene un campo nativo de
    usuario/rol en la llamada a una tool, así que esto es lo que hay.
    """
    role = role or config.MCP_CLIENT_ROLE
    contextual_input = (
        f"[rol del solicitante: {role} — pásalo como argumento 'role' en "
        f"cada tool call que lo acepte]\n\n{pregunta}"
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=contextual_input)]})
    final_message = result["messages"][-1]
    return final_message.content


async def run_single_question(pregunta: str, role: str | None = None) -> str:
    client = build_mcp_client()
    tools = await load_mcp_tools(client)
    llm = build_llm()
    graph = build_agent_graph(llm, tools)
    return await ask(graph, pregunta, role=role)


async def run_repl(role: str | None = None) -> None:
    """Loop interactivo simple para probar el agente a mano."""
    client = build_mcp_client()
    tools = await load_mcp_tools(client)
    print(f"Conectado al servidor MCP. Tools disponibles: {[t.name for t in tools]}")
    llm = build_llm()
    graph = build_agent_graph(llm, tools)
    print(f"Modelo: {config.OLLAMA_MODEL}  |  Rol: {role or config.MCP_CLIENT_ROLE}")
    print("Escribe una pregunta (o 'salir' para terminar).\n")

    while True:
        try:
            pregunta = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit"}:
            break
        respuesta = await ask(graph, pregunta, role=role)
        print(f"\n{respuesta}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agente LangGraph que consume el servidor MCP.")
    parser.add_argument(
        "--pregunta", default=None, help="Si se pasa, responde una sola pregunta y termina."
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Rol del solicitante para permission scoping (employee, hr, legal, admin).",
    )
    args = parser.parse_args()

    if args.pregunta:
        respuesta = asyncio.run(run_single_question(args.pregunta, role=args.role))
        print(respuesta)
    else:
        asyncio.run(run_repl(role=args.role))


if __name__ == "__main__":
    main()
