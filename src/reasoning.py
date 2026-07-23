"""
reasoning.py — Reasoning strategies (Lab 3)
===========================================

  - Few-shot CoT in the EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE format,
    adapted to the AMR domain.
  - Self-Consistency (k voices, majority vote by stance signature).

The CONFIDENCE tag is non-negotiable: an agent that expresses uncertainty is more
reliable than one that hallucinates confidently — all the more critical in health.
"""
from __future__ import annotations

import re
from collections import Counter

from llm_helpers import make_client, credentials_available

ONLINE = credentials_available()


def _demo(script=None):
    return make_client(offline_script=script, quiet=True)


# --------------------------------------------------------------------------- #
# Few-shot CoT prompt — AMR domain
# --------------------------------------------------------------------------- #
SYSTEM_SYNTHESIS = """
You are an antimicrobial-resistance (AMR) synthesis agent.
Answer ONLY from the provided CONTEXT. Always use this format:

EVIDENCE:
  - [fact 1 with its source]
  - [fact 2 with its source]

ANALYSIS:
  Step 1: [first reasoning step]
  Step 2: [second reasoning step]
  Step 3: [reconcile any contradiction]

CONCLUSION: [your answer]
CONFIDENCE: HIGH / MEDIUM / LOW — [one-sentence justification]

Rules:
- Never cite a source absent from the context.
- If the context is insufficient, say so and set CONFIDENCE: LOW.
- For any TREATMENT recommendation, add the note:
  "Decision support — clinical validation required."

Example:
Question: Is carbapenem resistance increasing?
EVIDENCE:
  - WHO GLASS: rise of carbapenem-resistant Klebsiella pneumoniae
  - Unfavourable 5-year trend in Southern Europe and South Asia
ANALYSIS:
  Step 1: a global surveillance source documents the rise
  Step 2: the trend is regional but convergent
  Step 3: no contradicting source in the context
CONCLUSION: Yes, carbapenem resistance is increasing, mostly in Southern Europe and South Asia
CONFIDENCE: HIGH — dedicated surveillance source, consistent trend
"""


def answer(question: str, context: str, system: str = SYSTEM_SYNTHESIS, script=None) -> str:
    client = _demo(script or [f"[simulated response] {question[:40]}…"])
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ]
    return client.complete(msg).content


# --------------------------------------------------------------------------- #
# Self-Consistency — k voices, vote by stance signature
# --------------------------------------------------------------------------- #
def _stance_signature(text: str) -> str:
    """Reduce a conclusion to a coarse signature (polarity + key entities)."""
    t = text.lower()
    polarity = "yes" if ("yes" in t[:20] or "increas" in t or "rising" in t
                         or "declin" in t) else "other"
    keywords = sorted(kw for kw in ["carbapenem", "klebsiella", "e. coli", "ecoli",
                                    "mrsa", "vancomycin", "ceftazidime", "aware",
                                    "glass", "1.27", "gonorrhoea"] if kw in t)
    return polarity + "|" + ",".join(keywords[:3])


def _mock_from_context(question: str, context: str, k: int) -> list:
    """Build k simulated answers GROUNDED in the retrieved context.

    In offline mode the MockLLMClient merely replays a script, so we build a
    correctly-formatted synthesis from the passages actually returned by the
    retriever — this keeps the demo coherent and lets the critic measure a
    realistic grounding score.
    """
    passages = [p.strip() for p in context.split("\n---\n") if p.strip()
                and not p.strip().startswith("[")]
    if not passages:
        passages = [context.strip() or "context unavailable"]

    def _clip(text, n=200):
        return (text[:n].rsplit(" ", 1)[0] + "…") if len(text) > n else text

    top = _clip(passages[0])
    ev_lines = "\n".join(f"  - {_clip(p, 140)}" for p in passages[:2])
    confidences = ["HIGH — several concordant passages",
                   "HIGH — dedicated surveillance source",
                   "MEDIUM — one main source"]
    scripts = []
    for i in range(k):
        conf = confidences[i % len(confidences)] if len(passages) >= 2 else confidences[2]
        block = (f"EVIDENCE:\n{ev_lines}\n"
                 f"ANALYSIS:\n  Step 1: the retrieved passages converge\n"
                 f"  Step 2: no contradicting source in the context\n"
                 f"CONCLUSION: {top}\n"
                 f"CONFIDENCE: {conf}")
        scripts.append([block])
    return scripts


def self_consistent_answer(question: str, context: str, k: int = 3,
                           mock_scripts=None, tracer=None) -> dict:
    """Generate k independent reasoning chains and vote the majority.

    k = 3 costs 3x the tokens — reserve it for the final synthesis.
    In offline mode the simulated answers are grounded in the context.
    If `tracer` is provided, each voice (LLM call) emits its own span.
    """
    scripts = mock_scripts or _mock_from_context(question, context, k)

    answers = []
    for i in range(k):
        script = None if ONLINE else scripts[i % len(scripts)]
        if tracer is not None:
            with tracer.span(f"llm_call:voice_{i + 1}", kind="llm", voice=i + 1):
                resp = answer(question, context, SYSTEM_SYNTHESIS, script=script)
        else:
            resp = answer(question, context, SYSTEM_SYNTHESIS, script=script)
        m = re.search(r"CONCLUSION\s*:\s*(.+?)(?:\nCONFIDENCE:|$)", resp, re.DOTALL)
        conclusion = m.group(1).strip() if m else resp[-200:]
        answers.append({"conclusion": conclusion, "full": resp})

    signatures = [_stance_signature(r["conclusion"]) for r in answers]
    winning_sig, count = Counter(signatures).most_common(1)[0]
    majority = next(r["conclusion"] for r, s in zip(answers, signatures) if s == winning_sig)

    return {
        "answer": majority,
        "confidence": count / k,
        "k": k,
        "agreement": count,
        "all": answers,
    }


if __name__ == "__main__":
    ctx = ("WHO GLASS: rise of carbapenem-resistant Klebsiella pneumoniae. "
           "ECDC: E. coli resistance to cephalosporins rising in Europe.")
    q = "Is antibiotic resistance increasing?"
    res = self_consistent_answer(q, ctx, k=3)
    print(f"Answer:    {res['answer']}")
    print(f"Agreement: {res['agreement']}/{res['k']} ({res['confidence']:.0%})")
