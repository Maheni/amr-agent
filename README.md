# 💊 AMR Agent — Antimicrobial-resistance monitoring & treatment protocols

A production agent that **synthesises antimicrobial-resistance (AMR) surveillance
reports** and **helps research treatment protocols** against a resistant germ.
It assembles the four blocks of the module: advanced RAG, security, reasoning,
production.

> **Topic #6 — Drug resistance** (combined angle: trend monitoring + treatment
> protocols). Offline-first code: it runs from a fresh clone with no API key.

---

## What the agent does (that a chatbot cannot)

For a question like *"What do protocols recommend for carbapenem-resistant
Enterobacteriaceae?"*, the agent:

1. **filters** the input (injection / homoglyphs) — L1 layer;
2. **retrieves** the relevant passages via a hybrid pipeline
   (BM25 + TF-IDF + RRF) → parent-child → **cross-encoder reranking**;
3. **plans its tool calls**, passes each through the **L4 gate**, then executes
   the approved ones on the **MCP server** in a single stdio session — with
   `lookup_treatment_protocol` classified `CONFIRM` (human validation);
4. **sanitises** every tool result before it enters the context (indirect-injection
   defence);
5. **synthesises** in the EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE format,
   with **Self-Consistency k=3**;
6. has the answer **reviewed by a second critic agent** (grounding + form), and
   only an `ACCEPT` unlocks the `store_finding` memory write;
7. **traces** every step (Langfuse or local spans), with a **token budget** and
   **5 monitoring metrics**.

---

## Quick start

```bash
git clone <your-repo>
cd amr-agent
cp .env.example .env          # optional: the agent runs WITHOUT a key
pip install -r requirements.txt
python src/agent.py           # full demo, produces output
```

The agent starts in **offline mode** (`MockLLMClient`): it runs with no API key.
As soon as a key is present in `.env`, the same code calls the real model — no
code change.

### Other commands

```bash
python src/retrieval.py            # evaluation: hit@3 / MRR table + RAGAS baseline vs final
python -m pytest tests/test_security.py -q   # the 5 injection tests (must all pass)
python src/mcp_server.py           # launch the MCP server standalone (stdio)
python src/guardrails.py           # L1 demo, including the homoglyph attack
python src/reasoning.py            # Self-Consistency demo
```

`python src/agent.py` runs the **10 evaluation questions**, an injection attempt,
a deliberately-triggered token budget, the monitoring report, and writes an
observable trace to `trace/` (`run_trace.json` + a self-contained
`run_trace.html` timeline). A clinical run's trace contains **13 spans**
(guardrails + 5 tool calls + 4 LLM calls), well above the "agent + 2 LLM + 2
tools" minimum. `trace/` is generated at runtime and git-ignored: the proof is
reproducible rather than committed.

### MCP transport

`agent.py` spawns `src/mcp_server.py` as a subprocess and talks the MCP stdio
protocol; each run reports its transport in the `[Obs]` line (`MCP=stdio` or
`MCP=fallback`). If the `mcp` package is missing, or if you set `AMR_USE_MCP=0`,
the agent falls back to **labelled** in-process calls so a fresh clone always
produces output. Optional visual inspection (needs Node):

```bash
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

---

## Architecture (summary)

```
Question ──▶ L1 (filter) ──▶ Retrieval (hybrid + parent-child + rerank)
                                   │
                                   ▼
                       Plan tools ──▶ L4 gate ──▶ MCP stdio session
                                   │                    │
                                   │              sanitise results
                                   ▼                    │
                   Synthesis few-shot CoT + Self-Consistency k=3
                                   │
                                   ▼
              Critic agent (verdict) ──ACCEPT──▶ store_finding (MCP)
                                   │
                                   ▼
                                Answer
             Observability: spans + budget + 5 metrics
```

Full details and Mermaid diagram: [`docs/architecture.md`](docs/architecture.md).

---

## Repository structure

```
amr-agent/
├── README.md              # this file
├── REPORT.md              # report (7 sections)
├── requirements.txt       # pinned dependencies
├── .env.example           # required keys (no values)
├── .gitignore
├── src/
│   ├── agent.py           # main loop + critic agent + observability + EU AI Act
│   ├── mcp_server.py      # MCP server (4 tools)
│   ├── retrieval.py       # corpus + hybrid/parent-child/reranking + metrics + RAGAS
│   ├── guardrails.py      # L1 + L4 + TokenBudget
│   ├── reasoning.py       # few-shot CoT + Self-Consistency
│   └── llm_helpers.py     # provider-agnostic LLM layer
├── tests/
│   └── test_security.py   # 5 injection tests
├── docs/
│   └── architecture.md    # Mermaid diagram + component descriptions
└── data/
    └── README.md          # how to plug in a real corpus
```

> `src/llm_helpers.py` is the **unmodified course helper**, vendored in the repo so
> that it runs from a fresh clone with no extra setup. Every other file is ours.

---

## Important note

The corpus figures are pedagogical orders of magnitude drawn from public sources
(WHO, ECDC, GRAM/Lancet 2022 study). The agent is a **monitoring and
decision-support** tool: any treatment recommendation must be **validated by a
clinician**.
