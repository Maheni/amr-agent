"""mcp_server.py — A 4-tool MCP server for AMR research (Lab 1, part 8).

Run standalone:  python src/mcp_server.py
Inspect:         npx @modelcontextprotocol/inspector python src/mcp_server.py

Tools exposed:
  - recall_memory              : read the internal base (surveillance + protocols)
  - search_surveillance        : targeted search in the surveillance reports
  - lookup_treatment_protocol  : antibiotic-therapy recommendation (DECISION SUPPORT)
  - store_finding              : store a verified fact

Each tool is wrapped in try/except and ALWAYS returns a string: a tool that
raises an exception can disconnect the whole server.

The tools are defined as plain functions and registered with FastMCP at the
bottom of the file. This keeps them importable and callable in-process (used by
`agent.py` as an explicit, labelled fallback when the `mcp` package is absent),
while the canonical path remains a real stdio MCP session.
"""
from __future__ import annotations

try:  # full base if launched from src/
    from retrieval import CORPUS
except Exception:  # fallback: embedded mini-corpus, the server runs anywhere
    CORPUS = {
        "gram_2019_deaths": "The GRAM study (Lancet 2022) estimates ~1.27 million deaths "
                            "directly attributable to bacterial resistance in 2019.",
        "ecdc_ecoli_c3g": "ECDC EARS-Net: E. coli resistance to third-generation "
                          "cephalosporins is rising in Europe, mostly South and East.",
        "glass_carbapenem_klebsiella": "WHO GLASS: rise of carbapenem-resistant Klebsiella "
                                       "pneumoniae in Southern Europe and South Asia.",
        "protocol_mrsa": "Severe MRSA: IV vancomycin or linezolid as first line.",
        "protocol_cre": "Carbapenem-resistant Enterobacteriaceae: ceftazidime-avibactam "
                        "or meropenem-vaborbactam depending on the mechanism.",
    }

# Treatment protocols (the "care" subset)
PROTOCOLS = {
    "mrsa": "Severe MRSA (bacteraemia/pneumonia): IV vancomycin or linezolid as first "
            "line; monitor vancomycin levels.",
    "cre": "Carbapenem-resistant Enterobacteriaceae (CRE): ceftazidime-avibactam or "
           "meropenem-vaborbactam; colistin as a last resort.",
    "carbapenem": "CRE: ceftazidime-avibactam or meropenem-vaborbactam; colistin as a "
                  "last resort.",
    "esbl": "ESBL, severe infection: carbapenems (meropenem/imipenem); simple cystitis: "
            "nitrofurantoin or fosfomycin.",
    "klebsiella": "Carbapenem-resistant Klebsiella (CRE): ceftazidime-avibactam or "
                  "meropenem-vaborbactam; colistin as a last resort.",
}

# In-memory store for findings saved during a session
_STORE: dict = {}


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def recall_memory(query: str) -> str:
    """Retrieve relevant passages from the internal AMR knowledge base.

    Use when: you need a fact already held in the internal base.
    Use FIRST, before any external search: it is free and instant.
    Do NOT use for: a treatment recommendation (see lookup_treatment_protocol).
    Returns: matching passages with their source id, or a message pointing to
    search_surveillance.
    Example: query="what treatment for MRSA"
    """
    try:
        q = set(query.lower().split())
        scored = []
        for doc_id, text in {**CORPUS, **_STORE}.items():
            overlap = len(q & set(text.lower().split()))
            if overlap:
                scored.append((overlap, doc_id, text))
        scored.sort(reverse=True)
        if not scored:
            return "No relevant memory. Try search_surveillance."
        return "\n---\n".join(f"[{doc_id}] {text}" for _, doc_id, text in scored[:3])
    except Exception as e:
        return f"Recall error: {e}"


def search_surveillance(query: str) -> str:
    """Targeted search in the SURVEILLANCE reports (WHO GLASS, ECDC, GRAM).

    Use when: you need trends, epidemiological figures, evolution by region or germ.
    Do NOT use for: a treatment recommendation (see lookup_treatment_protocol).
    Returns: the matching surveillance passages, or a not-found message.
    Example: query="E. coli resistance Europe"
    """
    try:
        q = set(query.lower().split())
        surveillance = {k: v for k, v in CORPUS.items()
                        if not k.startswith(("protocol_", "aware", "distractor_", "adv_"))}
        scored = [(len(q & set(t.lower().split())), i, t) for i, t in surveillance.items()]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        if not scored:
            return "No matching surveillance report."
        return "\n---\n".join(f"[{i}] {t}" for _, i, t in scored[:3])
    except Exception as e:
        return f"Surveillance error: {e}"


def lookup_treatment_protocol(pathogen: str) -> str:
    """Return an antibiotic-therapy recommendation for a given resistant germ.

    DECISION SUPPORT only — the output must be validated by a clinician.
    Use when: the question asks which antibiotic to use against MRSA, CRE /
    carbapenem-resistant strains, ESBL producers or resistant Klebsiella.
    Do NOT use for: making a diagnosis, prescribing autonomously, or answering a
    pure surveillance-trend question.
    Returns: the matching protocol plus a clinical-validation reminder.
    Example: pathogen="MRSA"
    """
    try:
        key = pathogen.lower().strip()
        hit = next((v for k, v in PROTOCOLS.items() if k in key or key in k), None)
        if not hit:
            return (f"No protocol for '{pathogen}'. Covered germs: "
                    f"MRSA, CRE/carbapenems, ESBL, Klebsiella.")
        return f"{hit}\n[DECISION SUPPORT — clinical validation required]"
    except Exception as e:
        return f"Protocol error: {e}"


def store_finding(finding: str, source: str) -> str:
    """Store a verified fact so recall_memory can return it later.

    Use when: a search produced a credible, relevant fact worth reusing.
    Do NOT store: speculation, unverified claims, or a patient's personal data.
    Returns: a confirmation with the assigned key.
    Example: finding="Colistin as last resort for CRE", source="WHO 2023"
    """
    try:
        key = f"finding_{len(_STORE) + 1}"
        _STORE[key] = f"{finding} (source: {source})"
        return f"Stored as {key}: {_STORE[key]}"
    except Exception as e:
        return f"Store error: {e}"


# Canonical registry — also used by agent.py's labelled in-process fallback.
TOOLS = {
    "recall_memory": recall_memory,
    "search_surveillance": search_surveillance,
    "lookup_treatment_protocol": lookup_treatment_protocol,
    "store_finding": store_finding,
}


# --------------------------------------------------------------------------- #
# FastMCP registration (optional import: the tools stay callable without `mcp`)
# --------------------------------------------------------------------------- #
try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("amr-research-tools")
    for _name, _fn in TOOLS.items():
        mcp.tool()(_fn)
except ImportError:  # pragma: no cover - the package is listed in requirements.txt
    mcp = None


if __name__ == "__main__":
    if mcp is None:
        raise SystemExit("The `mcp` package is required to run the server: "
                         "pip install -r requirements.txt")
    mcp.run(transport="stdio")
