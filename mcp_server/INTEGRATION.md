# Integrating mcp_server/ into your repo

Drop this `mcp_server/` folder (and `tests/test_mcp_tools.py`) into the
root of your `groundedkb/` repo, next to your existing `ingestion/`.

## One thing that's already wired, one thing you may want to check

1. **`mcp_server/embedder_factory.py`** — now calls your real
   `ingestion.embeddings.get_embedder(backend)` directly, no edit
   needed. It reads the backend from `GROUNDEDKB_EMBEDDER_BACKEND`
   (default `"hashing"`, matching `build_index.py`'s own default).
   **Set this env var to whatever `--backend` you actually used** when
   you last ran `build_index.py` — e.g.
   `GROUNDEDKB_EMBEDDER_BACKEND=sentence-transformers` — or the server
   will query the index with a different embedder than built it, and
   the distances (and the faithfulness threshold in
   `buscar_politica.py`) become meaningless.

2. **`mcp_server/doc_registry.py`** — maps `doc_id → (category, title)`
   for permission scoping. Built from the filenames in your README,
   **using the exact `doc_id` format your `build_index.py` produces**:
   `doc_id = md_file.name`, i.e. it includes the `.md` extension
   (`"politica_vacaciones.md"`, not `"politica_vacaciones"`). If your
   real `build_index.py` already computes a category and stores it as
   chunk metadata, swap this static table for a real metadata lookup
   (marked with `TODO` in the file) — and if you ever change `doc_id`
   to a bare stem, update these keys too.

## Known shortcuts, called out on purpose

- `citar_fuente.py` and `listar_documentos.py` reach into
  `vectorstore._collection` directly (Chroma's client) because the
  `VectorStore` interface you shared only exposes `query()` and
  `add_chunks()`, not a get-by-id or list-ids method. Marked `# noqa:
  SLF001` and `TODO` — cleanest fix is adding public methods to your
  `VectorStore`.
- `resumir_documento.py` reads the original file from
  `GROUNDEDKB_CORPUS_DIR/{doc_id}.md` rather than reconstructing from
  chunks, to avoid duplicated text if your chunker overlaps. It does an
  extractive (heading + first paragraph) summary — no LLM call, no API
  key required. There's a marked extension point if you want a real
  generated summary instead.
- Role-based access in `guardrails.py` is passed as a plain `role`
  argument to each tool, since MCP over stdio has no auth handshake to
  pull it from automatically. `agent_client/agent.py` would need to
  supply it.

## What's actually verified (not just written)

Ran end to end in a sandbox using your exact `vectorstore.py`, the real
`get_embedder("hashing")` factory shape from `build_index.py`, and
`doc_id`s with the `.md` extension exactly as `build_index.py`
produces them:
- All 4 tools register on `FastMCP`/`MCPServer` and are callable
  through `mcp.call_tool(...)` (the real dispatch path, not just the
  bare Python functions) — unmodified `embedder_factory.py` included.
- `buscar_politica`, `resumir_documento`, and `citar_fuente` all
  returned correct, grounded results against a real Chroma index.
- Role-based filtering blocks/allows `contratos` correctly for
  `employee` vs `legal`.
- `listar_documentos` correctly flags catalog/index mismatches.
- `pytest tests/test_mcp_tools.py` — 8/8 passing, no real Chroma or
  embedder needed (uses a stub `VectorStore`).

Also added a compatibility shim in `server.py` for the `mcp` SDK's
`FastMCP` → `MCPServer` rename (mcp<2 vs mcp>=2), since I don't know
which version is pinned in your `pyproject.toml`.
