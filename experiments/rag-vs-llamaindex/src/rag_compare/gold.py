"""Create and validate golden_v2.json, whose gold is annotated as text from the corpus.

Usage (from experiments/rag-vs-llamaindex, venv active):
    python -m rag_compare.gold init     # create data/golden_v2.json skeleton (never overwrites)
    python -m rag_compare.gold prefill  # fill EMPTY gold text from the old expected_fragment
    python -m rag_compare.gold check    # validate annotations and print resolved char spans
    python -m rag_compare.gold fix      # rewrite loosely-typed annotations as exact raw text

Each factual item has "gold": a list of {"doc": file name, "text": snippet}. A question needing
two separate pieces of evidence gets two entries. Trick questions keep "gold": [].

Matching is two-stage: first an exact substring match; if that fails, a normalized match that
ignores markdown bold (**), repeated whitespace/newlines, case, and curly quotes/dashes. The
span is always computed on the RAW document text, and the snippet must occur exactly once.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rag_compare.check_golden import CORPUS_DIR, GOLDEN_PATH, load_corpus, split_parts

V2_PATH = Path(
    os.getenv("GOLDEN_V2_PATH", Path(__file__).resolve().parents[2] / "data" / "golden_v2.json")
)

_CHAR_MAP = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}


def build_norm(text: str) -> tuple[str, list[int]]:
    """Normalize text and return (normalized, index_map) with index_map[i] = raw position."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            i += 2
            continue
        ch = _CHAR_MAP.get(text[i], text[i])
        if ch.isspace():
            if prev_space:
                i += 1
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch.lower())
        idx.append(i)
        i += 1
    return "".join(out), idx


def _balance_bold(text: str, start: int, end: int) -> tuple[int, int]:
    """If the span cuts a **bold** pair in half, extend it to include the missing marker."""
    if text[start:end].count("**") % 2 == 1:
        if text.startswith("**", end):
            end += 2
        elif start >= 2 and text[start - 2 : start] == "**":
            start -= 2
    return start, end


def resolve_span(text: str, snippet: str) -> tuple[int, int, str] | str:
    """Return (start, end, mode) on the raw text, or an error label.

    mode is "exact" or "normalized". The snippet must occur exactly once in either mode.
    """
    if not snippet.strip():
        return "EMPTY"
    count = text.count(snippet)
    if count == 1:
        start = text.index(snippet)
        return start, start + len(snippet), "exact"
    if count > 1:
        return f"AMBIGUOUS({count} matches)"

    norm_text, idx = build_norm(text)
    norm_snip = build_norm(snippet)[0].strip()
    if not norm_snip:
        return "EMPTY"
    starts, pos = [], norm_text.find(norm_snip)
    while pos != -1:
        starts.append(pos)
        pos = norm_text.find(norm_snip, pos + 1)
    if not starts:
        return "NOT_FOUND"
    if len(starts) > 1:
        return f"AMBIGUOUS({len(starts)} matches, normalized)"
    start = idx[starts[0]]
    end = idx[starts[0] + len(norm_snip) - 1] + 1
    start, end = _balance_bold(text, start, end)
    return start, end, "normalized"


def diagnose(text: str, snippet: str) -> str:
    """Explain where a NOT_FOUND snippet stops matching, in normalized form."""
    norm_text = build_norm(text)[0]
    norm_snip = build_norm(snippet)[0].strip()
    n = next((k for k in range(len(norm_snip), 0, -1) if norm_snip[:k] in norm_text), 0)
    if n == 0:
        return f"not even the start matches; your text starts with {norm_snip[:20]!r}"
    pos = norm_text.find(norm_snip[:n])
    return (
        f"(normalized) first {n} chars match; then your text has {norm_snip[n:n + 12]!r} "
        f"but the document has {norm_text[pos + n:pos + n + 12]!r}"
    )


def init(force: bool = False) -> None:
    if V2_PATH.exists() and not force:
        print(f"{V2_PATH} already exists; not overwriting (use --force to recreate it).")
        return
    old = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["items"]
    items = []
    for it in old:
        factual = it["type"] == "factual"
        items.append(
            {
                "id": it["id"],
                "question": it["question"],
                "expected_answer": it["expected_answer"],
                "category": it["category"],
                "difficulty": it["difficulty"],
                "type": it["type"],
                "split": None,
                "gold": [{"doc": it["expected_source_doc"], "text": ""}] if factual else [],
                "old_expected_fragment": it["expected_fragment"],
            }
        )
    V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "description": "Golden set v2: gold as text from the corpus, spans resolved by code.",
        "version": "2.0",
        "items": items,
    }
    V2_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Created {V2_PATH} with {len(items)} items. Fill in gold[].text, then run 'check'.")


def _load_v2() -> dict | None:
    if not V2_PATH.exists():
        print(f"{V2_PATH} does not exist yet. Run 'python -m rag_compare.gold init' first.")
        return None
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def prefill() -> None:
    """Copy old_expected_fragment into gold[].text where nothing is annotated yet.

    Never overwrites existing annotations. A fragment joined with '...' becomes one gold
    entry per part; merge them by hand if both parts turn out to be a single passage.
    """
    data = _load_v2()
    if data is None:
        return
    filled = skipped = 0
    for it in data["items"]:
        if it["type"] != "factual":
            continue
        if any(g["text"].strip() for g in it["gold"]):
            skipped += 1
            print(f"  {it['id']:10s} already annotated; left untouched")
            continue
        doc = it["gold"][0]["doc"] if it["gold"] else None
        parts = split_parts(it.get("old_expected_fragment") or "")
        if doc is None or not parts:
            print(f"  {it['id']:10s} nothing to prefill (no doc or no old fragment)")
            continue
        it["gold"] = [{"doc": doc, "text": part} for part in parts]
        filled += 1
        print(f"  {it['id']:10s} prefilled with {len(parts)} part(s)")
    V2_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPrefilled {filled} question(s), left {skipped} untouched. Now run 'check', then 'fix'.")


def check() -> None:
    data = _load_v2()
    if data is None:
        return
    corpus = load_corpus(CORPUS_DIR)
    ok_questions = factual = 0
    for it in data["items"]:
        if it["type"] != "factual":
            print(f"  {it['id']:10s} {it['type']:16s} (no gold expected)")
            continue
        factual += 1
        statuses, details = [], []
        for g in it["gold"]:
            if g["doc"] not in corpus:
                statuses.append(f"DOC_MISSING({g['doc']})")
                continue
            res = resolve_span(corpus[g["doc"]], g["text"])
            if isinstance(res, tuple):
                s, e, mode = res
                tag = "OK" if mode == "exact" else "OK~"
                statuses.append(f"{tag} {g['doc']}[{s}:{e}] len={e - s}")
                if mode == "normalized":
                    details.append(f"resolved from loose text; raw = {corpus[g['doc']][s:e]!r}")
            else:
                statuses.append(res)
                if res == "NOT_FOUND":
                    details.append("hint: " + diagnose(corpus[g["doc"]], g["text"]))
        if it["gold"] and all(s.startswith("OK") for s in statuses):
            ok_questions += 1
        print(f"  {it['id']:10s} {it['type']:16s} " + " | ".join(statuses or ["NO_GOLD_ENTRIES"]))
        for d in details:
            print(f"             -> {d}")
    print(f"\nAnnotated and valid: {ok_questions}/{factual} factual questions")
    print("OK~ = matched after normalization; run 'fix' to store the exact raw text.")


def fix() -> None:
    data = _load_v2()
    if data is None:
        return
    corpus = load_corpus(CORPUS_DIR)
    changed = 0
    for it in data["items"]:
        for g in it["gold"]:
            if g["doc"] not in corpus:
                continue
            res = resolve_span(corpus[g["doc"]], g["text"])
            if isinstance(res, tuple) and res[2] == "normalized":
                s, e, _ = res
                g["text"] = corpus[g["doc"]][s:e]
                changed += 1
                print(f"  {it['id']:10s} rewritten as exact raw text: {g['text']!r}")
    V2_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRewrote {changed} annotation(s). Run 'check' to confirm all are exact (OK).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "prefill", "check", "fix"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    {"init": lambda: init(args.force), "prefill": prefill, "check": check, "fix": fix}[args.command]()