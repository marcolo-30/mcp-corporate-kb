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

## Step 1 — Data (in progress): evaluation metrics

**What I learned**
- **Gold annotation must not depend on chunking.** Annotating "chunk 7" breaks as soon as chunk size changes. I annotate document + exact text, a script computes the character range, and a retrieved chunk is a hit if it overlaps that range.
- **Recall@k:** is the gold evidence inside the top-k? It says nothing about order or about whether the final answer is correct, only that the evidence reached the generator.
- **MRR:** average of 1/rank of the first relevant chunk (0 if absent). Worked example with positions 2, 1, absent, 4: recall@3 = 2/4, recall@5 = 3/4, MRR = 0.4375.
- **MRR is an average:** MRR ≈ 0.3 with recall@5 = 1 means the first relevant chunk appears around position 3 on average, but it hides the distribution. I need to look at per-question positions. Reranking only matters if few chunks reach the LLM; that has to be measured, not assumed.
- **Small samples:** with 14 questions, 12/14 (95% Wilson interval 60%–96%) and 10/14 (45%–88%) are statistically indistinguishable. Percentages are not "normalized" by sample size; the fix is to report counts, confidence intervals (Wilson for proportions, bootstrap for MRR) and a paired comparison of the questions where A and B disagree.
- **Trick questions** (no answer in the corpus) have no gold, so they are excluded from recall@k and MRR. They are evaluated in generation: correct abstention on trick questions AND no unwarranted abstention on answerable ones. A system that always abstains would score 100% on the trick questions and be useless.
- **RAGAS does not compute recall@k or MRR.** Per its documentation it offers LLM-judged metrics (context precision/recall, faithfulness, response relevancy, among others). My retrieval metrics are deterministic and based on my own gold annotation, so they do not depend on a judge model. Exact metric names and signatures for the pinned version (0.4.3) still have to be verified in step 6.

**What failed / surprises**
- I first argued that trick questions could explain a difference between A and B on 12/14 vs 10/14. Wrong: those 14 are only the factual questions. The real reasons are small-sample noise, which questions changed, and category effects.
- The golden set's `expected_fragment` values are not all exact substrings (some use "..." or paraphrase), so they cannot be used for recall@k as they are. To be confirmed with `check_golden`.

## Step 1 — Gold annotation findings

**What failed / surprises**
- `check_golden` reported 0/14 exact matches between the golden set's `expected_fragment` and the corpus. My first hypothesis (paraphrased or translated text) was wrong.
- After making the check markdown-aware, all 14 factual fragments match once markdown bold (`**`), whitespace and case are ignored: 8 single-span, 6 two-part fragments joined by "...". The content was faithful; only the formatting differed.
- Lesson: when a check fails everywhere, inspect one real example before theorizing. The first diff (Q01–Q03 against `vacation_policy.md`) revealed the cause in seconds.

**Decision**
- Gold is annotated as exact raw text from the corpus files (markdown included), in `data/golden_v2.json`; character spans are computed and validated by code, and must occur exactly once in the document. The original `eval/golden_set.json` is left untouched.

**Annotation cost and scaling**
- Annotating gold as exact raw text works for 14 questions and forced me to learn the corpus, but it does not scale to hundreds: every fragment must be copied from the source file (markdown markers included, not from the rendered preview), and one wrong character gives `NOT_FOUND`. The `hint` output of `gold check` made the errors quick to fix.
- Options considered for larger sets: coarser gold (whole sections; fast but inflates recall), tool-assisted matching (paste approximate text, code finds the span), answer-string matching (automatic, but false positives when the same number appears elsewhere), and LLM-generated questions (scale well, but tend to reuse the wording of the source chunk, which makes retrieval look easier than it is; a sample would need manual review).
- Statistical note: noise shrinks with the square root of n, so going from 14 to 60 questions narrows the confidence interval but does not remove it. With only 8 short documents, hundreds of questions would be near-duplicates; corpus variety matters more than question count.
- For now I annotate by hand. I will revisit tool-assisted matching only if the manual cost becomes a bottleneck.