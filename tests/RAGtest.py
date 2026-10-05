from pathlib import Path

from ingestion.embeddings import get_embedder
from ingestion.vectorstore import VectorStore
from mcp_server.tools.search_policy import search_policy
from mcp_server.tools.cite_source import cite_source

# Resolve project root the same way build_index.py and server.py do,
# so this test always points at the same index those two use.
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tests/ -> mcp-corporate-kb/
INDEX_PATH = str(PROJECT_ROOT / ".chroma_index")

# Setup: connect to the already-built index
store = VectorStore(embedder=get_embedder("hashing"), persist_path=INDEX_PATH)

# --- 1. Confirm total chunk count ---
total_chunks = store._collection.count()
print(f"Total chunks in index: {total_chunks}")  # should be 43

# --- 2. Confirm all 8 documents are represented ---
results = store._collection.get(include=["metadatas"])
doc_ids = set(m["doc_id"] for m in results["metadatas"])
print(f"Unique documents found ({len(doc_ids)}):")
for doc_id in sorted(doc_ids):
    print(f"  - {doc_id}")

# --- 3. Test that the RAG actually answers a real question ---
query = "how many vacation days do I accumulate?"

print("\n--- search_policy ---")
search_results = search_policy(store, query, top_k=3)
for r in search_results:
    print(r)

print("\n--- cite_source ---")
cite_result = cite_source(store, query)
print(cite_result)

