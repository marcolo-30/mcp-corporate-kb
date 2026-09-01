"""
server.py

MCP server entrypoint. Registers the 4 knowledge-base tools using the
official MCP Python SDK (FastMCP), wrapping every call with the
guardrails defined in guardrails.py (rate limiting + structured audit
logging).

Run with:
    python -m mcp_server.server

Or, for local interactive testing with the MCP Inspector:
    mcp dev mcp_server/server.py
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from ingestion.embeddings import get_embedder
from ingestion.vectorstore import VectorStore
from mcp_server.guardrails import guarded_tool_call, log_result
from mcp_server.tools.search_policy import search_policy as _search_policy
from mcp_server.tools.cite_source import cite_source as _cite_source
from mcp_server.tools.list_documents import list_documents as _list_documents
from mcp_server.tools.summarize_document import summarize_document as _summarize_document

CORPUS_DIR = os.environ.get("GROUNDEDKB_CORPUS_DIR", "corpus")
PERSIST_PATH = os.environ.get("GROUNDEDKB_INDEX_PATH", ".chroma_index")
EMBEDDING_BACKEND = os.environ.get("GROUNDEDKB_EMBEDDING_BACKEND", "hashing")

mcp = FastMCP("groundedkb")

_embedder = get_embedder(EMBEDDING_BACKEND)
_store = VectorStore(embedder=_embedder, persist_path=PERSIST_PATH)


@mcp.tool()
def search_policy(query: str, top_k: int = 3) -> list[dict]:
    """Searches the corporate knowledge base (HR policies, vendor
    contracts, support FAQs) and returns up to `top_k` relevant passages,
    each with its source document, section, and relevance score.
    """
    guarded_tool_call("search_policy", {"query": query, "top_k": top_k})
    result = _search_policy(_store, query, top_k=top_k)
    log_result("search_policy", {"query": query, "top_k": top_k}, f"{len(result)} results")
    return result


@mcp.tool()
def cite_source(query: str) -> dict:
    """Returns a single, confidence-checked citation (document + exact
    fragment) that answers `query` — or an explicit "not found" if no
    passage is confident enough. Use this instead of search_policy when
    you need ONE trustworthy, citable answer rather than candidates to
    review.
    """
    guarded_tool_call("cite_source", {"query": query})
    result = _cite_source(_store, query)
    log_result("cite_source", {"query": query}, f"found={result['found']}")
    return result


@mcp.tool()
def summarize_document(doc_id: str) -> dict:
    """Summarizes a specific document from the corpus, identified by its
    doc_id (e.g. 'vacation_policy.md'). Use list_documents first if you
    don't know the exact doc_id.
    """
    guarded_tool_call("summarize_document", {"doc_id": doc_id})
    result = _summarize_document(CORPUS_DIR, doc_id)
    log_result("summarize_document", {"doc_id": doc_id}, f"found={result['found']}")
    return result


@mcp.tool()
def list_documents(category: str | None = None) -> list[dict]:
    """Lists all documents available in the knowledge base, optionally
    filtered by category (hr_policy, vendor_contract, onboarding,
    it_support, finance_policy, conduct_policy).
    """
    guarded_tool_call("list_documents", {"category": category})
    result = _list_documents(CORPUS_DIR, category=category)
    log_result("list_documents", {"category": category}, f"{len(result)} documents")
    return result


if __name__ == "__main__":
    mcp.run()
