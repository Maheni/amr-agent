"""
llm_helpers.py — Couche d'abstraction LLM agnostique (OpenAI / Mistral / Anthropic)
====================================================================================

Module utilitaire partagé par les notebooks du module "Agentic AI & AI Agents" (PGE5).

Objectif pédagogique
--------------------
Permettre d'écrire des agents *une seule fois* et de basculer de fournisseur en
changeant uniquement une variable d'environnement, sans réécrire la logique
d'appel d'outils (function calling / tool use).

Ce que fournit ce module
-------------------------
- ``LLMClient``           : client de chat unifié (function calling, sortie structurée,
                             comptage de tokens et estimation de coût).
- ``MockLLMClient``       : client *hors-ligne* piloté par un script, pour faire tourner
                             les TP sans clé d'API ni connexion (utile en salle).
- ``make_client``         : fabrique qui choisit automatiquement le vrai client ou le mock.
- ``ToolRegistry``        : associe un nom d'outil à une fonction Python + son schéma.
- ``run_agent``           : boucle ReAct minimale réutilisable, avec trace optionnelle.
- ``safe_calc``           : évaluateur arithmétique SÛR (AST, pas de ``eval``).

Configuration
-------------
Variables d'environnement (un fichier .env est chargé automatiquement si présent) :

    LLM_PROVIDER = openai | mistral | anthropic | mock     (défaut: openai)
    LLM_MODEL    = nom du modèle (optionnel, sinon défaut par fournisseur)

    OPENAI_API_KEY / MISTRAL_API_KEY / ANTHROPIC_API_KEY selon le fournisseur.

Format de messages AGNOSTIQUE (liste de dict)
---------------------------------------------
    {"role": "system",    "content": "..."}
    {"role": "user",      "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall, ...]}
    {"role": "tool",      "tool_call_id": "...", "name": "...", "content": "..."}

Un ToolCall est un dict : {"id": str, "name": str, "arguments": dict}

Format des OUTILS (style OpenAI, traduit en interne pour chaque fournisseur)
----------------------------------------------------------------------------
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Donne la météo d'une ville",
        "parameters": { ... JSON Schema ... }
      }
    }
"""
from __future__ import annotations

import ast
import json
import math
import operator
import os
from dataclasses import dataclass, field
from typing import Any, Callable

# Chargement optionnel de .env puis .secrets (ne casse pas si python-dotenv absent)
try:
    from dotenv import load_dotenv

    load_dotenv()
    load_dotenv(".secrets", override=True)   # clés sensibles, non committées
except Exception:  # pragma: no cover
    pass


# Google/Gemini est exposé via son endpoint COMPATIBLE OpenAI : on réutilise le SDK openai.
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "mistral": "mistral-large-latest",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-2.5-flash",
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Tarifs INDICATIFS en USD par million de tokens (entrée, sortie).
# À adapter : les prix évoluent. Sert uniquement à illustrer le FinOps en TP.
PRICING_USD_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mistral-large-latest": (2.00, 6.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-flash-latest": (0.30, 2.50),
}


# --------------------------------------------------------------------------- #
# Évaluateur arithmétique SÛR (remplace eval) — utilisé comme outil "calculator"
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor,
    "ceil": math.ceil, "pow": pow,
}


def safe_calc(expression: str) -> float:
    """Évalue une expression arithmétique SANS ``eval`` (parsing AST, liste blanche).

    Autorise : nombres, + - * / // % **, parenthèses, et un petit jeu de fonctions
    (sqrt, abs, round, min, max, log, exp, sin, cos, ...) et constantes (pi, e).
    Tout le reste lève ``ValueError`` — c'est le comportement attendu d'un outil sûr.
    """

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("constante non autorisée")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FUNCS and not node.keywords:
                return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
            raise ValueError(f"fonction non autorisée: {node.func.id}")
        raise ValueError("expression non autorisée")

    return _eval(ast.parse(expression, mode="eval"))


# --------------------------------------------------------------------------- #
# Disponibilité des clés d'API
# --------------------------------------------------------------------------- #
def credentials_available(provider: str | None = None) -> bool:
    """Vrai si la clé d'API du fournisseur visé est présente dans l'environnement."""
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    key = ENV_KEYS.get(provider)
    return bool(key and os.getenv(key))


# --------------------------------------------------------------------------- #
# Structures normalisées
# --------------------------------------------------------------------------- #
@dataclass
class AssistantMessage:
    """Réponse normalisée du modèle, indépendante du fournisseur."""

    content: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict | None = None  # {"input_tokens": int, "output_tokens": int}

    def to_message(self) -> dict:
        """Reconvertit en message agnostique pour réinjection dans l'historique."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class _UsageMixin:
    """Comptabilité tokens / coût partagée par les clients."""

    model: str

    def _init_usage(self) -> None:
        self.last_usage: dict = {"input_tokens": 0, "output_tokens": 0}
        self.total_usage: dict = {"input_tokens": 0, "output_tokens": 0}
        self.n_calls: int = 0

    def _record_usage(self, usage: dict | None) -> dict:
        usage = usage or {"input_tokens": 0, "output_tokens": 0}
        self.last_usage = usage
        self.total_usage["input_tokens"] += usage.get("input_tokens", 0)
        self.total_usage["output_tokens"] += usage.get("output_tokens", 0)
        self.n_calls += 1
        return usage

    def estimated_cost(self, model: str | None = None) -> float:
        """Coût USD estimé du cumul des appels, d'après PRICING_USD_PER_MTOK."""
        model = model or getattr(self, "price_as", None) or self.model
        price_in, price_out = PRICING_USD_PER_MTOK.get(model, (0.0, 0.0))
        return (
            self.total_usage["input_tokens"] / 1e6 * price_in
            + self.total_usage["output_tokens"] / 1e6 * price_out
        )

    def usage_report(self) -> str:
        c = self.estimated_cost()
        return (
            f"{self.n_calls} appel(s) · "
            f"{self.total_usage['input_tokens']} tok in / "
            f"{self.total_usage['output_tokens']} tok out · "
            f"≈ ${c:.5f}"
        )


# --------------------------------------------------------------------------- #
# Client unifié
# --------------------------------------------------------------------------- #
class LLMClient(_UsageMixin):
    """Client de chat unifié avec support du function calling + sortie structurée."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Fournisseur inconnu: {self.provider!r}. "
                f"Choisir parmi {list(DEFAULT_MODELS)} (ou 'mock')."
            )
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self._init_usage()
        self._client = self._build_client()

    # ----- initialisation du SDK natif ------------------------------------- #
    def _build_client(self):
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI()
        if self.provider == "google":
            # Endpoint compatible OpenAI de Google → on réutilise le SDK openai.
            from openai import OpenAI

            return OpenAI(api_key=os.environ["GOOGLE_API_KEY"],
                          base_url=GOOGLE_OPENAI_BASE_URL)
        if self.provider == "mistral":
            from mistralai import Mistral

            return Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        if self.provider == "anthropic":
            from anthropic import Anthropic

            return Anthropic()
        raise RuntimeError("unreachable")

    # ----- appel principal -------------------------------------------------- #
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        tool_choice: Any = None,
    ) -> AssistantMessage:
        """Envoie l'historique au modèle et renvoie une réponse normalisée."""
        if self.provider in ("openai", "mistral", "google"):
            reply = self._complete_openai_like(
                messages, tools, temperature, max_tokens, tool_choice
            )
        else:
            reply = self._complete_anthropic(
                messages, tools, temperature, max_tokens, tool_choice
            )
        self._record_usage(reply.usage)
        return reply

    def complete_structured(
        self,
        messages: list[dict],
        schema: dict,
        name: str = "emit",
        description: str = "Renvoie le résultat au format structuré demandé.",
        **kwargs: Any,
    ) -> dict:
        """Force le modèle à renvoyer un JSON conforme à ``schema`` (agnostique).

        Astuce : on déclare un unique outil dont les paramètres SONT le schéma, et
        on force son appel. Les arguments de l'outil constituent la sortie structurée.
        """
        tool = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": schema},
        }
        if self.provider == "anthropic":
            choice: Any = {"type": "tool", "name": name}
        elif self.provider == "mistral":
            choice = "any"
        else:  # openai
            choice = {"type": "function", "function": {"name": name}}

        reply = self.complete(messages, tools=[tool], tool_choice=choice, **kwargs)
        if reply.tool_calls:
            return reply.tool_calls[0]["arguments"]
        # Repli : tenter de parser un JSON dans le texte (au cas où le forçage échoue).
        return _loads_loose(reply.content or "{}")

    # ----- implémentation OpenAI / Mistral (format compatible) ------------- #
    def _complete_openai_like(self, messages, tools, temperature, max_tokens, tool_choice):
        payload_messages = [_to_openai_message(m) for m in messages]
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=payload_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self.provider == "mistral":
            resp = self._client.chat.complete(**kwargs)
        else:  # openai, google (endpoint compatible OpenAI)
            resp = self._client.chat.completions.create(**kwargs)

        choice = resp.choices[0].message
        tool_calls = []
        for tc in getattr(choice, "tool_calls", None) or []:
            args = tc.function.arguments
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(args) if isinstance(args, str) else args,
                }
            )
        usage = _normalize_usage(getattr(resp, "usage", None), self.provider)
        return AssistantMessage(content=choice.content, tool_calls=tool_calls, usage=usage)

    # ----- implémentation Anthropic ---------------------------------------- #
    def _complete_anthropic(self, messages, tools, temperature, max_tokens, tool_choice):
        system_txt, conv = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=conv,
        )
        if system_txt:
            kwargs["system"] = system_txt
        if tools:
            kwargs["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        resp = self._client.messages.create(**kwargs)
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "arguments": block.input}
                )
        usage = _normalize_usage(getattr(resp, "usage", None), self.provider)
        return AssistantMessage(
            content="".join(text_parts) or None, tool_calls=tool_calls, usage=usage
        )


# --------------------------------------------------------------------------- #
# Client HORS-LIGNE piloté par script (aucune clé requise)
# --------------------------------------------------------------------------- #
class MockLLMClient(_UsageMixin):
    """Imite ``LLMClient`` sans appel réseau, pour les TP en l'absence de clé d'API.

    Pilotage par ``script`` : une liste d'étapes consommées à chaque ``complete()`` :
        {"tool": "calculator", "arguments": {"expression": "2+2"}}   # un appel d'outil
        {"tools": [ {...}, {...} ]}                                  # plusieurs appels
        {"final": "réponse finale"}                                  # réponse texte
        "réponse finale"                                             # idem (raccourci)

    Sans script (ou script épuisé), un petit comportement heuristique prend le relais
    pour rester utilisable. Le coût est SIMULÉ via ``price_as`` (par défaut gpt-4o-mini).
    """

    def __init__(self, script: list | None = None, price_as: str = "gpt-4o-mini"):
        self.provider = "mock"
        self.model = "mock-model"
        self.price_as = price_as
        self.script = list(script or [])
        self._i = 0
        self._init_usage()

    def reset(self) -> None:
        self._i = 0

    def complete(self, messages, tools=None, temperature=0.0, max_tokens=1024,
                 tool_choice=None) -> AssistantMessage:
        if self._i < len(self.script):
            reply = self._from_script(self.script[self._i])
            self._i += 1
        else:
            reply = self._heuristic(messages, tools)
        reply.usage = self._simulate_usage(messages, reply)
        self._record_usage(reply.usage)
        return reply

    def complete_structured(self, messages, schema, name="emit", description="", **kwargs):
        reply = self.complete(messages, tools=None)
        if reply.tool_calls:
            return reply.tool_calls[0]["arguments"]
        return _loads_loose(reply.content or "{}")

    # ----- helpers internes ------------------------------------------------- #
    @staticmethod
    def _from_script(step) -> AssistantMessage:
        if isinstance(step, str):
            return AssistantMessage(content=step)
        if "final" in step:
            return AssistantMessage(content=step["final"])
        calls = step.get("tools") or [step]
        tool_calls = [
            {"id": f"mock_{i}", "name": c["tool"], "arguments": c.get("arguments", {})}
            for i, c in enumerate(calls)
        ]
        return AssistantMessage(content=step.get("content"), tool_calls=tool_calls)

    @staticmethod
    def _heuristic(messages, tools) -> AssistantMessage:
        import re

        last = messages[-1] if messages else {}
        if last.get("role") == "tool":
            return AssistantMessage(
                content=f"[réponse simulée] D'après l'outil : {last.get('content')}"
            )
        user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        names = [t["function"]["name"] for t in (tools or [])]
        if "calculator" in names:
            m = re.search(r"[-+*/0-9.\s()]{3,}", user)
            if m and re.search(r"\d\s*[-+*/]\s*\d", m.group()):
                return AssistantMessage(
                    tool_calls=[{"id": "mock_h", "name": "calculator",
                                 "arguments": {"expression": m.group().strip()}}]
                )
        return AssistantMessage(content="[réponse simulée hors-ligne] " + (user[:160] or "ok"))

    def _simulate_usage(self, messages, reply) -> dict:
        def toks(txt) -> int:
            return max(1, len(str(txt)) // 4)

        return {
            "input_tokens": sum(toks(m.get("content")) for m in messages),
            "output_tokens": toks(reply.content) + 8 * len(reply.tool_calls),
        }


# --------------------------------------------------------------------------- #
# Fabrique : vrai client si une clé est dispo, sinon mock
# --------------------------------------------------------------------------- #
def make_client(
    offline_script: list | None = None,
    provider: str | None = None,
    model: str | None = None,
    force_mock: bool = False,
    quiet: bool = False,
):
    """Renvoie un ``LLMClient`` si la clé est présente, sinon un ``MockLLMClient``.

    Idéal en tête de notebook : les cellules s'exécutent en salle même sans clé.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    if force_mock or provider == "mock" or not credentials_available(provider):
        if not quiet:
            print("⚙️  Mode HORS-LIGNE (MockLLMClient) — aucune clé d'API détectée. "
                  "Renseignez .env pour appeler le vrai modèle.")
        return MockLLMClient(script=offline_script)
    if not quiet:
        print(f"🌐 Mode EN LIGNE — fournisseur={provider}.")
    return LLMClient(provider, model)


# --------------------------------------------------------------------------- #
# Conversions de format
# --------------------------------------------------------------------------- #
def _normalize_usage(usage, provider) -> dict:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    if provider == "anthropic":
        return {
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        }
    # openai / mistral
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }


def _loads_loose(text: str) -> dict:
    """Parse JSON tolérant : retire d'éventuelles clôtures markdown ```json ... ```."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    try:
        return json.loads(t)
    except Exception:
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(t[start : end + 1])
            except Exception:
                pass
    return {}


def _to_openai_message(m: dict) -> dict:
    role = m["role"]
    if role == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m.get("content") or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in m["tool_calls"]
            ],
        }
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m["tool_call_id"],
            "content": str(m["content"]),
        }
    return {"role": role, "content": m.get("content", "")}


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Renvoie (system, conversation) au format Anthropic.

    Les messages 'tool' consécutifs sont fusionnés en un seul tour 'user'
    contenant des blocs tool_result (exigence de l'API Anthropic).
    """
    system_txt: str | None = None
    conv: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_txt = (system_txt + "\n\n" if system_txt else "") + m["content"]
        elif role == "user":
            conv.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m.get("tool_calls", []):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    }
                )
            conv.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            result_block = {
                "type": "tool_result",
                "tool_use_id": m["tool_call_id"],
                "content": str(m["content"]),
            }
            # fusion avec le tour user précédent s'il contient déjà des tool_result
            if (
                conv
                and conv[-1]["role"] == "user"
                and isinstance(conv[-1]["content"], list)
            ):
                conv[-1]["content"].append(result_block)
            else:
                conv.append({"role": "user", "content": [result_block]})
    return system_txt, conv


def _openai_tool_to_anthropic(tool: dict) -> dict:
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


# --------------------------------------------------------------------------- #
# Mini-registre d'outils + boucle d'agent réutilisable
# --------------------------------------------------------------------------- #
def tool_schema(name: str, description: str, properties: dict | None = None,
                required: list | None = None) -> dict:
    """Petit raccourci pour écrire un schéma d'outil (style OpenAI)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


class ToolRegistry:
    """Associe un nom d'outil à une fonction Python et à son schéma."""

    def __init__(self):
        self._specs: list[dict] = []
        self._funcs: dict[str, Callable[..., Any]] = {}

    def register(self, schema: dict, func: Callable[..., Any]) -> None:
        self._specs.append(schema)
        self._funcs[schema["function"]["name"]] = func

    @property
    def specs(self) -> list[dict]:
        return self._specs

    @property
    def names(self) -> list[str]:
        return [s["function"]["name"] for s in self._specs]

    def call(self, name: str, arguments: dict) -> str:
        if name not in self._funcs:
            return f"ERREUR: outil inconnu '{name}'"
        try:
            return str(self._funcs[name](**arguments))
        except Exception as exc:  # remonte l'erreur au modèle plutôt que de planter
            return f"ERREUR pendant l'exécution de '{name}': {exc}"


def run_agent(
    client,
    registry: ToolRegistry,
    messages: list[dict],
    max_steps: int = 8,
    verbose: bool = True,
    on_event: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Boucle ReAct minimale : appelle le modèle, exécute les outils, recommence.

    ``on_event`` (optionnel) reçoit des dicts {"type": ..., ...} à chaque étape :
    pratique pour tracer/observer sans modifier la boucle. Renvoie l'historique complet.
    """
    for step in range(1, max_steps + 1):
        reply = client.complete(messages, tools=registry.specs)
        messages.append(reply.to_message())

        if not reply.has_tool_calls:
            if verbose:
                print(f"[étape {step}] réponse finale → {reply.content}")
            if on_event:
                on_event({"type": "final", "step": step, "content": reply.content})
            return messages

        for tc in reply.tool_calls:
            if verbose:
                print(f"[étape {step}] outil: {tc['name']}({tc['arguments']})")
            result = registry.call(tc["name"], tc["arguments"])
            if verbose:
                print(f"          ↳ {result}")
            if on_event:
                on_event({"type": "tool", "step": step, "name": tc["name"],
                          "arguments": tc["arguments"], "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": result,
                }
            )
    raise RuntimeError(f"Limite de {max_steps} étapes atteinte sans réponse finale.")


if __name__ == "__main__":
    # Petit test de fumée. Fonctionne SANS clé grâce au mock.
    print("safe_calc('sqrt(1764) + 2**3') =", safe_calc("sqrt(1764) + 2**3"))
    client = make_client(offline_script=[
        {"tool": "calculator", "arguments": {"expression": "21*2"}},
        {"final": "La réponse est 42."},
    ])
    reg = ToolRegistry()
    reg.register(tool_schema("calculator", "Évalue une expression.",
                             {"expression": {"type": "string"}}, ["expression"]),
                 safe_calc)
    hist = run_agent(client, reg, [{"role": "user", "content": "Combien font 21*2 ?"}])
    print("Usage :", client.usage_report())
