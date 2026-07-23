# `data/` folder

This folder holds the AMR agent's document corpus.

## What it contains by default

No data file is required to run the agent: the demo corpus is embedded at the top
of `src/retrieval.py` (20 documents — 9 answer-bearing, split between
**surveillance** and **protocols**, and 11 **distractors** including 6
**adversarial** ones with high lexical overlap, which widen the MRR gap between
the baseline and the production pipeline).

The corpus lives next to the retriever that indexes it and the metrics that score
it, so that changing a document and re-measuring is a single file edit followed by
`python src/retrieval.py`.

## How to populate it with your own sources

1. Drop your text documents here (one `.txt` per source, or a `.jsonl` with `id`
   and `text` fields).
2. Replace the `CORPUS` dictionary in `src/retrieval.py` with a loader that reads
   this folder, **keeping distractors** (documents with close vocabulary but no
   answer) — this is what lets hit@k / MRR and RAGAS reveal the value of the
   hybrid + reranking pipeline.
3. Update `QUESTIONS`, `GOLD_DOC`, `GROUND_TRUTH` and `CLINICAL_QUESTION_INDICES`
   accordingly. Keep at least 10 questions: RAGAS needs that many to produce a
   stable score.
4. Re-run `python src/retrieval.py` to get the new baseline before touching the
   pipeline. Measure first, improve second.

## Recommended public AMR sources

- **WHO GLASS** — Global Antimicrobial Resistance and Use Surveillance System
- **ECDC EARS-Net** — European resistance surveillance
- **GRAM study** (Lancet, 2022) — global burden of bacterial resistance
- **WHO AWaRe classification** — Access / Watch / Reserve

> The demo corpus figures are pedagogical orders of magnitude; they are not a
> clinical reference.
