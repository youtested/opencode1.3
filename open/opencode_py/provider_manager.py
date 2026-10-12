"""Provider/key management for the /provider popup."""
from __future__ import annotations

from typing import Any

from .auth import Auth
from .providers.tables import FREE_DEFAULT_MODELS

FREE_INFO = {
    "groq": ("Groq", "GROQ_API_KEY", "https://console.groq.com/keys"),
    "cerebras": ("Cerebras", "CEREBRAS_API_KEY", "https://cloud.cerebras.ai/"),
    "google": ("Google AI Studio", "GOOGLE_API_KEY", "https://aistudio.google.com/app/apikey"),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"),
    "nvidia": ("NVIDIA NIM", "NVIDIA_API_KEY", "https://build.nvidia.com/"),
    "mistral": ("Mistral", "MISTRAL_API_KEY", "https://console.mistral.ai/api-keys"),
    "github": ("GitHub Models", "GITHUB_TOKEN", "https://github.com/settings/tokens"),
    "sambanova": ("SambaNova", "SAMBANOVA_API_KEY", "https://cloud.sambanova.ai/apis"),
    "togetherai": ("Together AI", "TOGETHER_API_KEY", "https://api.together.ai/settings/api-keys"),
}

PAID_INFO = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "anthropic": ("Anthropic Claude", "ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
    "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys"),
    "xai": ("xAI", "XAI_API_KEY", "https://console.x.ai/"),
    "deepinfra": ("DeepInfra", "DEEPINFRA_API_KEY", "https://app.deepinfra.com/keys"),
}


def all_providers() -> dict[str, dict[str, str]]:
    out = {key: {"name": value[0], "env": value[1], "url": value[2], "kind": "free"} for key, value in FREE_INFO.items()}
    out.update({key: {"name": value[0], "env": value[1], "url": value[2], "kind": "paid"} for key, value in PAID_INFO.items()})
    return out


def has_key(auth: Auth, provider: str) -> bool:
    return bool(auth.get(provider))


def status(auth: Auth, provider: str) -> str:
    return "Connected" if has_key(auth, provider) else "Key needed"


def default_model(provider: str) -> str:
    return FREE_DEFAULT_MODELS.get(provider, "")


def set_key(auth: Auth, provider: str, key: str) -> None:
    if not key.strip():
        raise ValueError("API key cannot be empty")
    auth.set(provider, key.strip())


def add_custom(auth: Auth, cfg: Any, name: str, url: str, key: str = "") -> str:
    from .config import save_config

    name = str(name or "").strip()
    url = str(url or "").strip()
    if not name or not url:
        raise ValueError("Provider name and URL are required")
    raw = getattr(cfg, "raw", None)
    if not isinstance(raw, dict):
        raw = {}
        cfg.raw = raw
    custom = raw.setdefault("customProviders", {})
    custom[name] = {"name": name, "url": url}
    if key.strip():
        set_key(auth, name, key)
    save_config(cfg)
    return name


def remove_key(auth: Auth, provider: str) -> bool:
    if not has_key(auth, provider):
        return False
    auth.remove(provider)
    return True


def rows(auth: Auth, cfg: Any = None) -> list[dict[str, str]]:
    providers = dict(all_providers())
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    custom = raw.get("customProviders") or {}
    if isinstance(custom, dict):
        for name, info in custom.items():
            if isinstance(info, dict):
                providers[str(name)] = {"name": str(info.get("name") or name), "env": "", "url": str(info.get("url") or ""), "kind": "custom"}
    result = []
    for provider, info in providers.items():
        result.append({
            "id": provider,
            "name": info["name"],
            "env": info.get("env", ""),
            "url": info.get("url", ""),
            "kind": info.get("kind", "custom"),
            "status": status(auth, provider),
            "model": default_model(provider),
        })
    return result


__all__ = ["add_custom", "all_providers", "default_model", "has_key", "remove_key", "rows", "set_key", "status"]
