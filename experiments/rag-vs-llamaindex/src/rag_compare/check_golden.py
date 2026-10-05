"""Check whether each golden-set fragment can be located in the corpus.

Usage (from experiments/rag-vs-llamaindex, venv active):
    python -m rag_compare.check_golden

Paths default to the parent repo's corpus/ and eval/golden_set.json; override with
the CORPUS_DIR and GOLDEN_PATH environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# src/rag_compare/check_golden.py -> parents[4] is the repo root (mcp-corporate-kb)
REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", REPO_ROOT / "corpus"))
GOLDEN_PATH = Path(os.getenv("GOLDEN_PATH", REPO_ROOT / "eval" / "golden_set.json"))


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    """Return {file name: full text} for every .md file under corpus_dir."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.rglob("*.md"))}


def split_parts(fragment: str) -> list[str]:
    """Golden fragments sometimes use '...' to join pieces; split them into parts."""
    return [part.strip() for part in fragment.split("...") if part.strip()]


def norm(text: str) -> str:
    """Normalize for lenient matching: drop markdown bold, collapse whitespace, lowercase."""
    return " ".join(text.replace("**", "").split()).lower()


def classify(fragment: str, text: str) -> str:
    """Classify how well a fragment matches a document's text.

    EXACT / EXACT_AMBIGUOUS : usable as gold as-is (raw substring of the document).
    PARTS_OK                : fragment uses '...'; every part is a raw substring.
    *_FORMAT                : would match after dropping markdown '**', whitespace and case.
    PARTS_PARTIAL / NOT_FOUND: wording differs; the gold must be re-annotated.
    """
    if fragment in text:
        return "EXACT" if text.count(fragment) == 1 else "EXACT_AMBIGUOUS"
    parts = split_parts(fragment)
    if len(parts) > 1:
        if all(part in text for part in parts):
            return f"PARTS_OK({len(parts)})"
        found = sum(norm(part) in norm(text) for part in parts)
        if found == len(parts):
            return f"PARTS_FORMAT({len(parts)})"
        return f"PARTS_PARTIAL({found}/{len(parts)})"
    if norm(fragment) in norm(text):
        return "FORMAT_ONLY"
    return "NOT_FOUND"


def main() -> None:
    corpus = load_corpus(CORPUS_DIR)
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["items"]

    print(f"Corpus dir : {CORPUS_DIR}")
    print(f"Golden set : {GOLDEN_PATH}\n")
    print("Documents:")
    for name, text in corpus.items():
        print(f"  {name:36s} {len(text):6d} chars  {len(text.split()):5d} words")

    print("\nPer-question check:")
    exact = 0
    factual = 0
    for item in golden:
        qid, kind = item["id"], item["type"]
        if kind != "factual":
            print(f"  {qid:10s} {kind:16s} (no gold expected)")
            continue
        factual += 1
        doc = item["expected_source_doc"]
        if doc not in corpus:
            print(f"  {qid:10s} {kind:16s} DOC_MISSING: {doc}")
            continue
        status = classify(item["expected_fragment"], corpus[doc])
        if status == "EXACT":
            exact += 1
        print(f"  {qid:10s} {kind:16s} {status:18s} {doc}")

    print(f"\nExact single-span matches: {exact}/{factual} factual questions")
    print("Anything other than EXACT needs a corrected gold annotation (step 1).")


if __name__ == "__main__":
    main()