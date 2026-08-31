# Integrating agent_client/ into your repo

Drop `agent_client/` into the root of `groundedkb/`, alongside
`mcp_server/` and `ingestion/`.

## Install

```bash
pip install langgraph langchain-ollama langchain-mcp-adapters "mcp<2"
```

**Pin `mcp<2` deliberately.** `langchain-mcp-adapters` 0.3.2 breaks on
`mcp` 2.1.1 — its `ClientSession` import chain crashes on an internal
bug in `mcp`'s new experimental "tasks" module, unrelated to anything
in this project. `mcp<2` (tested here on 1.29.1) works cleanly and is
also what `mcp_server/server.py`'s `FastMCP`/`MCPServer` compat shim
already supports.

## Ollama setup

```bash
ollama pull qwen3.5   # or set GROUNDEDKB_OLLAMA_MODEL to one you have
ollama serve          # if not already running
```

## Run it

```bash
# Interactive REPL
python -m agent_client.agent

# One-shot question
python -m agent_client.agent --pregunta "cuantos dias de vacaciones acumulo"

# As a different role (see mcp_server/guardrails.py)
python -m agent_client.agent --role legal --pregunta "plazo de terminacion con cloudtech"
```

## Config (env vars, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `GROUNDEDKB_OLLAMA_MODEL` | `qwen3.5` | Which local model reasons/answers |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `GROUNDEDKB_OLLAMA_TEMPERATURE` | `0.1` | Kept low on purpose — this agent should cite, not improvise |
| `GROUNDEDKB_MCP_COMMAND` / `GROUNDEDKB_MCP_ARGS` | `python -m mcp_server.server` | How to launch the MCP server subprocess |
| `GROUNDEDKB_CLIENT_ROLE` | `employee` | Default role passed to guardrails.py |

Any `GROUNDEDKB_*` / `OLLAMA_*` var you export before running the agent
also needs to reach the **spawned MCP server subprocess** (e.g.
`GROUNDEDKB_CHROMA_PATH`, `GROUNDEDKB_EMBEDDER_BACKEND` from the
mcp_server package) — see the real bug this surfaced, below.

## A real bug this surfaced, fixed rather than left silent

`StdioConnection` (from `langchain-mcp-adapters`) does **not** inherit
the full parent environment by default — only a documented subset. In
testing, this meant `GROUNDEDKB_CHROMA_PATH` etc. silently failed to
reach the spawned `mcp_server` subprocess, which fell back to its
default index path with no error — `buscar_politica` just always
returned `"encontrado": false`. Fixed in `build_mcp_client()` by
passing `env=dict(os.environ)` explicitly. If you add more env vars
later, this passthrough means you don't need to touch this file again.

## Two design choices worth knowing about

1. **Role is passed as a prompt hint, not a protocol field.** MCP tool
   calls have no native "current user" concept, so the agent prepends
   `[rol del solicitante: ...]` to the user's message and relies on the
   system prompt (`prompts.py`) telling the LLM to forward it as a
   `role` argument. This works but is soft enforcement — a model that
   ignores the instruction won't pass `role`, and the server then falls
   back to `DEFAULT_ROLE`. For real access control, enforce role at the
   MCP transport/auth layer instead, not by trusting the LLM to comply.
2. **Grounding is enforced by prompt, not by code**, per the README's
   own design: `buscar_politica` returns raw fragments only, never a
   pre-written answer, precisely so `eval/run_eval.py` can score
   faithfulness against something the LLM actually produced — a
   hard-coded "safe" answer would defeat the point of evaluating it.

## What's actually verified (not just written)

Ran end to end in a sandbox against the real `mcp_server` (from the
previous step) and a real Chroma-backed index:
- `build_mcp_client()` + `load_mcp_tools()` connect over stdio and
  correctly discover all 4 tools.
- Calling `buscar_politica` through the LangChain tool wrapper returns
  a correctly grounded, cited result — this is what caught the env
  passthrough bug above.
- The full `create_react_agent` graph runs a real tool-call round trip
  (tool call → real MCP tool execution → tool result fed back → final
  answer) using a scripted fake chat model standing in for Ollama,
  since Ollama isn't reachable from this sandbox (it runs on your
  machine). **The Ollama call itself (`ChatOllama`, actual qwen3.5
  inference and its tool-calling format) is not verified** — that part
  you'll need to run and confirm locally.
