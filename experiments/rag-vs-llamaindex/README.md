# RAG from scratch vs LlamaIndex

A controlled, reproducible comparison of the same retrieval pipeline built twice behind one Python interface:

- **A** — custom implementation (chunking, embeddings, vector store and retrieval written by hand).
- **B** — LlamaIndex (B1: vector, B2: hybrid BM25 + vector, B3: B2 + reranking).

It reuses `corpus/` from the parent repo (`mcp-corporate-kb`) and an extended copy of `eval/golden_set.json`.

> **Status: in progress.** No results exist yet. Every table below says "pending" until it is filled by a real run.

## Goal

Reach **honest conclusions I can defend in an interview**, not a pretty result. If there is no real difference between A and B, the README will say so.

## Hypotheses

> To be written and committed **before** running the evaluation (step 3). The two below are drafts to refine.

- H1 (draft): hybrid retrieval (B2) improves recall@3 on questions that contain exact terms.
- H2 (draft): reranking (B3) improves precision@3 but increases latency.
- H3 (to write): expected difference between A and B1 when chunking and embeddings are matched.

## Method

Controlled comparison: one variable changes at a time.

| Fixed across A and B | Value |
|---|---|
| Corpus | `corpus/` (8 synthetic documents) |
| Golden set | extended copy, dev/test split (pending step 1) |
| Embedding model | `nomic-embed-text` via Ollama (provisional) |
| Chunk size / overlap | 512 / 64 (provisional, matched as closely as possible) |
| Top-k | 5 |
| Generation model | pending (temperature 0) |
| RAGAS judge | pending (same judge for every config) |

Gold annotation: document + character range per question; a retrieved chunk is a hit if it overlaps the gold range, so metrics do not depend on chunk size.

Configurations: **A** (custom, vector), **B1** (LlamaIndex, vector), **B2** (LlamaIndex, hybrid), **B3** (B2 + reranking).

## Results

Retrieval metrics (no LLM), reported on the **test** split. Counts are shown next to percentages; small differences with few questions may be noise.

| Config | recall@1 | recall@3 | recall@5 | MRR | precision@3 | p50 latency |
|---|---|---|---|---|---|---|
| A | pending | pending | pending | pending | pending | pending |
| B1 | pending | pending | pending | pending | pending | pending |
| B2 | pending | pending | pending | pending | pending | pending |
| B3 | pending | pending | pending | pending | pending | pending |

RAGAS (A vs best B config, same judge):

| Metric | A | Best B | Note |
|---|---|---|---|
| Context precision | pending | pending | |
| Context recall | pending | pending | |
| Faithfulness | pending | pending | |
| Answer relevancy | pending | pending | |

## How to run

```bash
cd experiments/rag-vs-llamaindex
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env                                  # Windows: Copy-Item .env.example .env
ollama pull nomic-embed-text
```

Run commands will be added as each step lands.

## Limitations

- Small corpus (8 documents) and small golden set: differences can be noise.
- Local judge model for RAGAS may be weaker and noisier than a large hosted judge (to be assessed in step 6).
- Synthetic corpus: results may not transfer to real corporate documents.

## Roadmap

- [x] Step 0 — Environment (structure, pinned versions, `.env.example`, README skeleton)
- [ ] Step 1 — Data: review corpus and golden set, gold annotations, hard questions, dev/test split
- [ ] Step 2 — Custom RAG (A) behind a shared `Retriever` interface
- [ ] Step 3 — Retrieval evaluation (no LLM) and baseline for A
- [ ] Step 4 — LlamaIndex vector (B1)
- [ ] Step 5 — LlamaIndex hybrid and reranking (B2, B3)
- [ ] Step 6 — Generation + RAGAS
- [ ] Step 7 — Failure analysis and conclusions vs hypotheses
- [ ] Step 8 — Final README and LinkedIn draft

## What I would do next

To be written at the end, from real results.