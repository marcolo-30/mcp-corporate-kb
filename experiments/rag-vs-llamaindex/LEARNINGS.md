# LEARNINGS

Running log of what I learned, what failed and why. Raw material for LinkedIn posts.

## Step 0 — Environment

**What I did**
- Created the experiment folder inside the existing repo, with its own `pyproject.toml` so it does not touch the main project's dependencies.
- Pinned exact versions after checking them on PyPI (2026-10-05) instead of trusting memory.

**What I learned**
- In a comparative experiment the environment is part of the method: if a library version changes between runs, a difference in results could come from the library and not from the pipeline.
- Provisional fixed variables for A and B: embedding model `nomic-embed-text` (Ollama), chunk 512 / overlap 64, top-k 5. Choosing between embedding models will be a separate experiment, changing only that variable.
- The generator and the RAGAS judge cannot be loaded at the same time on a 6 GB GPU, so they will run one after the other.

**What failed / surprises**
- Pinning `numpy==2.5.3` forces Python >= 3.12 (PyPI metadata), which is stricter than the parent repo (>= 3.11). Needs the local Python version confirmed before the first install.