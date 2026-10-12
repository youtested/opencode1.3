"""Tool-schema helpers shared by the provider and tool layers.

These two helpers used to live in ``providers.base`` and ``tools.registry``,
which made those two modules import each other. Both implementations are pure
(stdlib only), so they live here as a leaf module instead: anything can import
them without creating a cycle.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["sanitize_function_name", "provider_schema"]


def sanitize_function_name(name: str) -> str:
    """Wire-safe function name for strict validators (Nemotron 400s on ``:``).

    Only ``a-z, A-Z, 0-9, _, -`` allowed, max 64 chars. Lenient models accept
    anything, strict ones reject the whole request — so sanitize on the wire.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name or ""))
    safe = safe.strip("_-") or "tool"
    return safe[:64]


def provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a tool schema for a provider, dropping non-standard keywords.

    ``schema_with``/``_param`` add an ``"optional": True`` hint for the TUI
    (and tool builders); it is NOT part of JSON Schema, and strict provider
    tool-schema validators (Anthropic, some OpenAI-compatible gateways) reject
    unknown keywords. Return a copy so the stored tool definition keeps its
    hint while the wire format stays valid.
    """
    if not isinstance(schema, dict):
        return schema
    clean: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "optional":
            continue
        if key == "properties" and isinstance(value, dict):
            clean[key] = {
                name: provider_schema(prop) if isinstance(prop, dict) else prop
                for name, prop in value.items()
            }
        else:
            clean[key] = value
    return clean
