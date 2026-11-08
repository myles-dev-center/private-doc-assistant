"""Model presets ("profiles") the UI can switch between, e.g. local models and a cloud API.

A profile overrides settings: endpoints, model names, the embedding model. API keys are
written as ${ENV_VAR} and resolved on the server, so they never reach the browser. Each
profile keeps its documents in its own library (a folder under DOCASSIST_DATA_DIR),
because vectors from different embedding models can't be compared; profiles that share
an embedding model may share a library - e.g. local and cloud chat over the same index.

Without a profiles file there is one profile, built from the environment / .env.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import Settings
from .llm import PRICES, is_local_url

_VAR = re.compile(r"\$\{(\w+)\}")


@dataclass
class Profile:
    name: str
    label: str
    overrides: dict
    library: str
    missing: list[str]  # environment variables the profile needs but are not set


def load_profiles(path: Path | None) -> tuple[list[Profile], str]:
    """(profiles, name of the default profile)."""
    if path is None:
        return [Profile("default", "", {}, "", [])], "default"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for model, (p_in, p_cached, p_out) in (raw.get("prices") or {}).items():
        PRICES.setdefault(model, (p_in, p_cached, p_out, 0))
    profiles = []
    for name, spec in raw["profiles"].items():
        spec = dict(spec)
        label, library = spec.pop("label", name), spec.pop("library", name)
        unknown = set(spec) - set(Settings.model_fields)
        if unknown:
            raise ValueError(f"{path}: profile {name} has unknown settings {sorted(unknown)}")
        missing: list[str] = []
        overrides = {k: _resolve(v, missing) for k, v in spec.items()}
        profiles.append(Profile(name, label, overrides, library, missing))
    default = raw.get("default", profiles[0].name)
    if default not in {p.name for p in profiles}:
        raise ValueError(f"{path}: default profile {default} is not defined")
    return profiles, default


def _resolve(value, missing: list[str]):
    if not isinstance(value, str):
        return value

    def sub(m: re.Match) -> str:
        if m.group(1) not in os.environ:
            missing.append(m.group(1))
        return os.environ.get(m.group(1), "")

    return _VAR.sub(sub, value)


def settings_for(base: Settings, profile: Profile) -> Settings:
    return base.model_copy(update={**profile.overrides, "data_dir": base.data_dir / profile.library})


def describe(profile: Profile, s: Settings) -> dict:
    """What the UI may know about a profile - never keys or URLs."""
    chat_local = is_local_url(s.llm_base_url)
    embed_local = is_local_url(s.embed_base_url or s.llm_base_url)
    return {
        "name": profile.name,
        "label": profile.label or s.llm_model,
        "chat_model": s.llm_model,
        "embed_model": s.embed_model,
        "library": profile.library or "default",
        "local": chat_local and embed_local,
        "chat_local": chat_local,  # answers (and extraction at upload) run on this machine
        "embed_local": embed_local,  # documents and questions are embedded on this machine
        "priced": chat_local or s.llm_model in PRICES,
        "available": not profile.missing,
        "missing": profile.missing,
    }
