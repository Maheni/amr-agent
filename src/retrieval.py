"""
retrieval.py — AMR knowledge base + production retrieval pipeline + evaluation
==============================================================================

This module holds three things that belong together (the data, the retriever
that indexes it, and the metric that measures the retriever):

  1. CORPUS / QUESTIONS / GOLD_DOC / GROUND_TRUTH  — the AMR knowledge base and
     its evaluation set (10 questions).
  2. The retrieval chain, with no heavy dependency (no Chroma, no Ollama):

         hybrid (BM25 + TF-IDF + RRF)  ->  parent-child  ->  cross-encoder rerank

  3. The evaluation entry point (`python src/retrieval.py`): the offline
     hit@k / MRR proxy, plus the real RAGAS table (baseline vs final) when a key
     and the `ragas` package are available.

The corpus DELIBERATELY contains distractors — documents that share the
questions' vocabulary but do NOT hold the answer, including 5 *adversarial* ones
with very high lexical overlap. This is what lets us measure the gap between a
naive retriever and the production pipeline.

The cross-encoder is *simulated* (no sentence-transformers required). It weights
matches on the query's DISTINCTIVE (high-IDF) terms, which is what a real
cross-encoder captures. In production, replace `cross_encoder_score` with a real
`sentence_transformers.CrossEncoder`.

The figures are pedagogical orders of magnitude drawn from widely circulated
public sources (WHO, ECDC, GRAM/Lancet 2022 study). They are training material
for the agent, not a clinical reference.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# --------------------------------------------------------------------------- #
# 0. Knowledge base
# --------------------------------------------------------------------------- #
CORPUS = {
    # ── SURVEILLANCE: answer-bearing documents ──
    "gram_2019_deaths":
        "The GRAM study published in The Lancet in 2022 estimates that about "
        "1.27 million deaths were directly attributable to bacterial resistance to "
        "antibiotics in 2019, and nearly 5 million deaths were associated with it. "
        "Antimicrobial resistance (AMR) is among the leading causes of death worldwide.",
    "ecdc_ecoli_c3g":
        "According to the ECDC EARS-Net network, resistance of Escherichia coli to "
        "third-generation cephalosporins keeps rising in Europe, with wide disparities: "
        "the highest rates are seen in Southern and Eastern Europe. E. coli remains the "
        "most frequently reported pathogen.",
    "glass_carbapenem_klebsiella":
        "The WHO GLASS global surveillance system reports a worrying increase in the "
        "resistance of Klebsiella pneumoniae to carbapenems, last-resort antibiotics. "
        "Southern Europe and South Asia show the most unfavourable trends over five years.",
    "mrsa_trend_europe":
        "The proportion of methicillin-resistant Staphylococcus aureus (MRSA) has "
        "declined in several Western European countries over the past decade thanks to "
        "control programmes, but it remains high in the south and east of the continent. "
        "The overall European trend is a slow decline.",
    "gonorrhoea_ceftriaxone":
        "The WHO warns about the emergence of Neisseria gonorrhoeae strains resistant "
        "to ceftriaxone, the last reliable single-therapy option. Resistant cases have "
        "been reported in Asia-Pacific and Europe, raising fears of untreatable gonorrhoea.",

    # ── PROTOCOLS: answer-bearing documents ──
    "protocol_cre":
        "For carbapenem-resistant Enterobacteriaceae (CRE), recent protocols recommend "
        "the combinations ceftazidime-avibactam or meropenem-vaborbactam as first line "
        "depending on the resistance mechanism. Colistin is reserved as a last resort "
        "because of its renal toxicity.",
    "protocol_mrsa":
        "For a severe MRSA infection (bacteraemia, pneumonia), intravenous vancomycin "
        "or linezolid is the first-line treatment. Monitoring of vancomycin levels is "
        "recommended to avoid nephrotoxicity.",
    "protocol_esbl":
        "For severe infections caused by extended-spectrum beta-lactamase (ESBL) "
        "producing Enterobacteriaceae, carbapenems (meropenem, imipenem) remain the "
        "treatment of choice. For an uncomplicated ESBL cystitis, nitrofurantoin or "
        "fosfomycin may suffice.",
    "aware_classification":
        "The WHO AWaRe classification sorts antibiotics into three categories: Access "
        "(first line, to be favoured), Watch (to be monitored, high resistance potential) "
        "and Reserve (last resort, to be preserved). The goal is that at least 60% of "
        "consumption falls in the Access category.",

    # ── DISTRACTORS: same theme, close vocabulary, but NOT the answer ──
    "distractor_pipeline":
        "The development pipeline for new antibiotics remains insufficient. Few new "
        "classes have reached the market since the 1980s, and the economic model "
        "discourages pharmaceutical investment.",
    "distractor_carbx_gardp":
        "Initiatives such as CARB-X and GARDP fund research into new antimicrobial "
        "treatments. Public-private funding seeks to offset the poor return on "
        "investment of antibiotics.",
    "distractor_vaccines":
        "Vaccination reduces the use of antibiotics by preventing infections. "
        "Pneumococcal vaccines lower resistant infections but do not treat a resistance "
        "that is already established.",
    "distractor_agriculture":
        "The use of antibiotics in livestock accounts for a large share of global "
        "consumption. Reducing antibiotics as animal growth promoters is a lever of the "
        "One Health approach.",
    "distractor_hand_hygiene":
        "Hand-hygiene and healthcare-associated infection prevention programmes reduce "
        "the transmission of bacteria in hospitals, but belong to prevention, not to the "
        "treatment of a resistant strain.",

    # ── ADVERSARIAL DISTRACTORS: high density of the question's keywords, but no
    #    answer. Designed to trap a naive TF-IDF (high lexical overlap) where BM25 +
    #    RRF + reranking stay robust. ──
    "adv_deaths_debate":
        "The number of deaths attributable to bacterial resistance in 2019 is debated. "
        "Deaths attributable to and deaths associated with resistance are often confused "
        "in mortality estimates, and calculation methods vary across studies, which "
        "complicates any numeric comparison of resistance deaths.",
    "adv_ecoli_agri":
        "E. coli resistance to cephalosporins in Europe. E. coli cephalosporin "
        "resistance in Europe is monitored. Resistance of E. coli to cephalosporins "
        "across Europe in livestock. E. coli, cephalosporins, resistance, Europe.",
    "adv_mrsa_history":
        "MRSA (methicillin-resistant Staphylococcus aureus) was first described back in "
        "the 1960s. Its history, hospital epidemiology and transmission are widely "
        "documented, but the choice of a first-line treatment depends on the infection "
        "site and the local context.",
    "adv_carbapenem_mechanism":
        "Becoming resistant to carbapenems is a concern. Carbapenem resistance and "
        "carbapenems. Bacteria becoming resistant to carbapenems. Carbapenem-resistant "
        "bacteria and carbapenems, resistance to carbapenems.",
    "adv_aware_history":
        "The notion of good antibiotic stewardship is old. The WHO has published "
        "essential-medicines lists and discussed their classification for years, within "
        "the broader fight against antimicrobial resistance.",
    "adv_esbl_epidemiology":
        "ESBL-producing Enterobacteriaceae are spreading. ESBL epidemiology, ESBL "
        "carriage and ESBL infections are described in many cohorts, but carriage "
        "prevalence says nothing about which antibiotic to choose for a severe ESBL "
        "infection.",
}


# --------------------------------------------------------------------------- #
# Evaluation set — 10 questions (RAGAS requires >= 10)
# --------------------------------------------------------------------------- #
QUESTIONS = [
    "How many deaths were directly attributable to bacterial resistance in 2019?",
    "Is E. coli resistance to cephalosporins rising in Europe?",
    "What is the first-line treatment for a severe MRSA infection?",
    "What do protocols recommend for carbapenem-resistant Enterobacteriaceae?",
    "What is the purpose of the WHO AWaRe classification of antibiotics?",
    "Is Klebsiella pneumoniae becoming resistant to carbapenems?",
    "Is MRSA declining in Western Europe?",
    "Is Neisseria gonorrhoeae becoming resistant to ceftriaxone?",
    "Which antibiotic is the treatment of choice for a severe ESBL infection?",
    "How many deaths were associated with antimicrobial resistance in 2019?",
]

GOLD_DOC = [
    "gram_2019_deaths",
    "ecdc_ecoli_c3g",
    "protocol_mrsa",
    "protocol_cre",
    "aware_classification",
    "glass_carbapenem_klebsiella",
    "mrsa_trend_europe",
    "gonorrhoea_ceftriaxone",
    "protocol_esbl",
    "gram_2019_deaths",
]

GROUND_TRUTH = [
    "about 1.27 million deaths directly attributable to resistance in 2019",
    "yes, E. coli resistance to third-generation cephalosporins is increasing in Europe",
    "intravenous vancomycin or linezolid as first line",
    "ceftazidime-avibactam or meropenem-vaborbactam depending on the mechanism",
    "sort antibiotics into Access, Watch and Reserve to preserve the molecules",
    "increasing in Southern Europe and South Asia over five years",
    "yes, a slow decline in Western Europe but still high in the south and east",
    "yes, ceftriaxone-resistant strains are emerging",
    "carbapenems, meropenem or imipenem, are the treatment of choice",
    "nearly 5 million deaths were associated with bacterial resistance in 2019",
]

# "Clinical" subset: questions whose answer touches a treatment recommendation.
# The agent handles these with extra caution (L4 CONFIRM gate + EU AI Act section).
CLINICAL_QUESTION_INDICES = [2, 3, 8]


# --------------------------------------------------------------------------- #
# 1. Tokenisation helpers
# --------------------------------------------------------------------------- #
_TOKEN_RE = r"[a-zàâçéèêëîïôûùüÿœ0-9]+"


def _tokenise(text: str) -> list:
    return re.findall(_TOKEN_RE, text.lower())


def _tokset(text: str) -> set:
    return set(re.findall(_TOKEN_RE, text.lower()))


def _build_corpus_idf(docs: dict) -> dict:
    """Corpus-level IDF, used to weight distinctive terms in the reranker."""
    tok = [set(_tokenise(t)) for t in docs.values()]
    n = len(tok)
    df = Counter(w for ts in tok for w in ts)
    return {w: math.log((1 + n) / (1 + df[w])) + 1 for w in df}


_CORPUS_IDF = _build_corpus_idf(CORPUS)


# --------------------------------------------------------------------------- #
# 2. Dense TF-IDF (baseline)
# --------------------------------------------------------------------------- #
class TinyTfidf:
    """Minimal TF-IDF retriever (cosine), no external dependency."""

    def __init__(self, docs: dict):
        self.names = list(docs.keys())
        self.texts = list(docs.values())
        tok = [_tokenise(t) for t in self.texts]
        n = len(self.texts)
        df = Counter(w for ts in tok for w in set(ts))
        self.idf = {w: math.log((1 + n) / (1 + df[w])) + 1 for w in df}
        self.vecs = [self._vec(ts) for ts in tok]

    def _vec(self, tokens):
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {w: tf[w] / n * self.idf.get(w, 0.0) for w in tf}

    @staticmethod
    def _cos(a, b):
        num = sum(a[w] * b[w] for w in set(a) & set(b))
        na = math.sqrt(sum(v ** 2 for v in a.values()))
        nb = math.sqrt(sum(v ** 2 for v in b.values()))
        return num / (na * nb) if na * nb else 0.0

    def search(self, query: str, k: int = 3) -> list:
        qv = self._vec(_tokenise(query))
        scores = [(self._cos(qv, v), t) for v, t in zip(self.vecs, self.texts)]
        return sorted(scores, reverse=True, key=lambda x: x[0])[:k]


# --------------------------------------------------------------------------- #
# 3. BM25 (exact terms: numbers, acronyms, proper nouns)
# --------------------------------------------------------------------------- #
class TinyBM25:
    """Minimal BM25-Okapi, no external dependency."""

    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.names = list(docs.keys())
        self.texts = list(docs.values())
        self.tok = [_tokenise(t) for t in self.texts]
        n = len(self.texts)
        avgdl = sum(len(ts) for ts in self.tok) / n
        df = Counter(w for ts in self.tok for w in set(ts))
        self.idf = {w: math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1) for w in df}
        self._k1, self._b, self._avgdl = k1, b, avgdl

    def _score(self, tok_q, idx):
        dl = len(self.tok[idx])
        sc = 0.0
        tf_d = Counter(self.tok[idx])
        for w in tok_q:
            if w not in self.idf:
                continue
            f = tf_d[w]
            num = self.idf[w] * f * (self._k1 + 1)
            den = f + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            sc += num / den
        return sc

    def search(self, query: str, k: int = 3) -> list:
        q = _tokenise(query)
        ranked = sorted(range(len(self.texts)),
                        key=lambda i: self._score(q, i), reverse=True)
        return [(self._score(q, i), self.texts[i]) for i in ranked[:k]]


# --------------------------------------------------------------------------- #
# 4. Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_fusion(lists: list, K: int = 60) -> list:
    """Reciprocal Rank Fusion — fuses several lists of (score, text) by rank."""
    scores: dict = {}
    for lst in lists:
        for rank, (_, text) in enumerate(lst):
            scores[text] = scores.get(text, 0.0) + 1.0 / (K + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- #
# 5. Parent-child index: index small, return large
# --------------------------------------------------------------------------- #
def _split_words(text: str, size: int, overlap: int) -> list:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


children: dict = {}          # child_id  -> short text
parents: dict = {}           # parent_id -> long text
child_to_parent: dict = {}

for _doc_id, _text in CORPUS.items():
    for _p_idx, _parent_text in enumerate(_split_words(_text, size=60, overlap=10)):
        _parent_id = f"{_doc_id}_p{_p_idx}"
        parents[_parent_id] = _parent_text
        for _c_idx, _child_text in enumerate(_split_words(_parent_text, size=20, overlap=3)):
            _child_id = f"{_parent_id}_c{_c_idx}"
            children[_child_id] = _child_text
            child_to_parent[_child_id] = _parent_id

retriever_children = TinyTfidf(children)
_bm25_children = TinyBM25(children)


# --------------------------------------------------------------------------- #
# 6. Cross-encoder (simulated) + reranking
# --------------------------------------------------------------------------- #
def cross_encoder_score(query: str, document: str) -> float:
    """Simulated cross-encoder — in production: sentence_transformers.CrossEncoder.

    A real cross-encoder reads the query and the document TOGETHER and rewards
    matches on the query's meaningful, distinctive terms. We approximate that by
    weighting the query∩document overlap with corpus IDF, so a long keyword-dense
    distractor does not beat the document that actually matches the rare entities
    of the question (e.g. "Klebsiella", "AWaRe", "gonorrhoeae").
    Replace with: reranker.predict([(query, doc)]).
    """
    q_tok = _tokset(query)
    d_tok = _tokset(document)
    if not q_tok or not d_tok:
        return 0.0
    q_weight = sum(_CORPUS_IDF.get(w, 1.0) for w in q_tok) or 1.0
    matched = sum(_CORPUS_IDF.get(w, 1.0) for w in (q_tok & d_tok))
    coverage = matched / q_weight            # IDF-weighted term coverage
    doc_bonus = min(1.0, len(d_tok) / 60)    # small richer-context bonus
    return round(coverage * 0.85 + doc_bonus * 0.15, 4)


def rerank(query: str, candidates: list, top_k: int = 3) -> list:
    scored = [(cross_encoder_score(query, doc), doc) for doc in candidates]
    return [doc for _, doc in sorted(scored, reverse=True, key=lambda x: x[0])[:top_k]]


# --------------------------------------------------------------------------- #
# 7. Exposed retrievers
# --------------------------------------------------------------------------- #
_dense = TinyTfidf(CORPUS)
_bm25 = TinyBM25(CORPUS)


def baseline_retrieve(query: str, k: int = 3) -> list:
    """Baseline: plain TF-IDF over whole documents (pre-Block-1 pipeline)."""
    return [txt for _, txt in _dense.search(query, k)]


def hybrid_retrieve(query: str, k: int = 3) -> list:
    """Hybrid BM25 + TF-IDF fused by RRF (whole documents)."""
    dense_results = _dense.search(query, k=k * 2)
    bm25_results = _bm25.search(query, k=k * 2)
    fused = rrf_fusion([dense_results, bm25_results])
    return [text for text, _ in fused[:k]]


def production_retrieve(query: str, k_final: int = 3) -> list:
    """Full pipeline: hybrid over children -> parents -> cross-encoder rerank."""
    dense_c = retriever_children.search(query, k=10)
    bm25_c = _bm25_children.search(query, k=10)
    fused_c = rrf_fusion([dense_c, bm25_c])

    seen, candidates = set(), []
    for text, _ in fused_c:
        child_id = next((cid for cid, ct in children.items() if ct == text), None)
        if child_id:
            pid = child_to_parent[child_id]
            if pid not in seen:
                seen.add(pid)
                candidates.append(parents[pid])
    return rerank(query, candidates, top_k=k_final)


# --------------------------------------------------------------------------- #
# 8. Metrics (distractor-sensitive)
# --------------------------------------------------------------------------- #
def hit_at_k(retriever_fn, k: int = 3) -> float:
    """Fraction of questions whose GOLD document appears in the top-k."""
    hits = 0
    for q, gold_id in zip(QUESTIONS, GOLD_DOC):
        chunks = retriever_fn(q, k)
        gold_tok = _tokset(CORPUS[gold_id])
        found = any(len(_tokset(c) & gold_tok) / max(1, len(gold_tok)) > 0.5 for c in chunks)
        hits += found
    return round(hits / len(QUESTIONS), 3)


def mrr(retriever_fn, k: int = 5) -> float:
    """Mean Reciprocal Rank: rewards having the right document AT THE TOP."""
    total = 0.0
    for q, gold_id in zip(QUESTIONS, GOLD_DOC):
        chunks = retriever_fn(q, k)
        gold_tok = _tokset(CORPUS[gold_id])
        rank = 0
        for pos, c in enumerate(chunks, start=1):
            if len(_tokset(c) & gold_tok) / max(1, len(gold_tok)) > 0.5:
                rank = pos
                break
        total += (1.0 / rank) if rank else 0.0
    return round(total / len(QUESTIONS), 3)


def compare_retrievers() -> dict:
    """Return hit@3 and MRR for baseline / hybrid / production."""
    out = {}
    for name, fn in [
        ("baseline", baseline_retrieve),
        ("hybrid", hybrid_retrieve),
        ("production", production_retrieve),
    ]:
        out[name] = {"hit@3": hit_at_k(fn, 3), "mrr": mrr(fn, 5)}
    return out


# --------------------------------------------------------------------------- #
# 9. RAGAS — the 4 reference metrics, BASELINE vs FINAL
# --------------------------------------------------------------------------- #
def _build_ragas_rows(retriever_fn, synthesise: bool):
    """Build the RAGAS dataset rows for one retriever configuration.

    `synthesise=True` runs the real few-shot-CoT synthesis on the retrieved
    context, so `faithfulness` and `answer_relevancy` measure the ANSWER the
    agent would actually produce — not just the top passage.
    """
    from reasoning import answer as synthesise_answer  # local import: avoids a cycle

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q, gt in zip(QUESTIONS, GROUND_TRUTH):
        ctx = retriever_fn(q, 3)
        if synthesise:
            ans = synthesise_answer(q, "\n---\n".join(ctx))
        else:
            ans = ctx[0] if ctx else ""
        rows["question"].append(q)
        rows["answer"].append(ans)
        rows["contexts"].append(ctx)
        rows["ground_truth"].append(gt)
    return rows


def run_ragas(synthesise: bool = True) -> dict | None:
    """Compute the 4 RAGAS metrics for BASELINE and FINAL (production).

    Requires: an API key in .env AND `pip install ragas langchain-openai datasets`
    (see the optional block in requirements.txt). Returns a dict of results, or
    None if the environment does not allow it.

    Note: `answer_relevancy` needs an embeddings endpoint. Providers without one
    (e.g. Groq) will run only the 3 LLM-judged metrics — this is reported as-is
    in REPORT.md rather than filled with an invented number.
    """
    import os

    from llm_helpers import credentials_available

    if not credentials_available():
        print("\nOFFLINE — RAGAS not run. The hit@3 / MRR proxy above is the measurement.")
        print("Fill .env with a key and install the optional block of "
              "requirements.txt to get the 4 metrics.")
        return None

    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (answer_relevancy, context_precision,
                                   context_recall, faithfulness)

        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")

        judge = ChatOpenAI(model=model, api_key=api_key, base_url=base_url,
                           temperature=0)
        has_embeddings = (not base_url) or ("openai.com" in base_url)
        emb = OpenAIEmbeddings(api_key=api_key) if has_embeddings else None

        metrics = [context_recall, context_precision, faithfulness]
        kwargs = {"llm": judge}
        if emb is not None:
            metrics.append(answer_relevancy)
            kwargs["embeddings"] = emb
        else:
            print("\n[RAGAS] provider without an embeddings endpoint: "
                  "answer_relevancy is skipped (documented in REPORT.md §3.2).")

        results = {}
        for label, fn in [("BASELINE (plain TF-IDF)", baseline_retrieve),
                          ("FINAL (hybrid + parent-child + rerank)", production_retrieve)]:
            print(f"\n[RAGAS] evaluating: {label} — {len(QUESTIONS)} questions…")
            ds = Dataset.from_dict(_build_ragas_rows(fn, synthesise))
            results[label] = evaluate(ds, metrics=metrics, **kwargs)

        print("\n=== RAGAS — BASELINE vs FINAL "
              f"({len(QUESTIONS)} questions) ===")
        for label, scores in results.items():
            print(f"\n{label}\n  {scores}")
        print("\nCopy these two rows into REPORT.md §3.2.")
        return results

    except ImportError as exc:
        print(f"\n[RAGAS] package missing ({exc.name}). Install the optional block "
              f"of requirements.txt.")
    except Exception as exc:  # network, quota, provider incompatibility…
        print(f"\n[RAGAS] not run here ({type(exc).__name__}: {exc}). "
              f"The hit@3 / MRR proxy above remains the offline measurement.")
    return None


# --------------------------------------------------------------------------- #
# 10. Evaluation entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _distract_prefixes = ("distractor_", "adv_")
    _n_distract = sum(1 for k in CORPUS if k.startswith(_distract_prefixes))
    print(f"Corpus: {len(CORPUS)} documents "
          f"({len(CORPUS) - _n_distract} answer-bearing, {_n_distract} distractors "
          f"incl. adversarial) · {len(QUESTIONS)} evaluation questions\n")

    print("=== RETRIEVER — offline proxy (distractor-sensitive) ===\n")
    print(f'{"Retriever":<38}{"hit@3":>8}{"MRR":>8}')
    print("-" * 54)
    for _name, _m in compare_retrievers().items():
        print(f"{_name:<38}{_m['hit@3']:>8}{_m['mrr']:>8}")
    print("\nReading: cross-encoder reranking lifts the right document to the TOP")
    print("(baseline MRR < production MRR); hit@3 already saturates on this corpus.")

    run_ragas()
