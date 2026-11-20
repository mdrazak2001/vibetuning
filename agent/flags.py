"""Feature flags. Every integration is opt-in and must fail closed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None

from agent import ROOT


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    if _dotenv_load:
        _dotenv_load(path, override=False)
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file(ROOT / ".env")
_load_env_file(Path.home() / ".vibetuning" / "secrets.env")
_load_env_file(Path.home() / ".jl.env")


def _truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class FlagStatus:
    cognee: bool = False
    singlestore: bool = False
    airbyte: bool = False
    amplitude: bool = False
    statsig: bool = False
    superlinked: bool = False
    whisper_gpu: bool = False
    notes: dict[str, str] = field(default_factory=dict)
    budget_cap_inr: float = 150.0
    warm_resume_enabled: bool = True
    statsig_source: str = "env"


_STATUS: FlagStatus | None = None
_STATSIG = None


def _try_statsig(status: FlagStatus) -> None:
    global _STATSIG
    if not _truthy("STATSIG_ENABLED", default=bool(os.getenv("STATSIG_SDK_KEY") or os.getenv("STATSIG_SECRET_KEY"))):
        status.statsig = False
        return
    key = os.getenv("STATSIG_SDK_KEY") or os.getenv("STATSIG_SECRET_KEY") or ""
    if not key:
        status.statsig = False
        status.notes["statsig"] = "no STATSIG_SDK_KEY"
        return
    try:
        from statsig_python_core import Statsig, StatsigOptions, StatsigUser

        options = StatsigOptions()
        options.environment = os.getenv("STATSIG_ENV", "development")
        client = Statsig(key, options)
        client.initialize().wait()
        _STATSIG = client
        user = StatsigUser(os.getenv("STATSIG_USER", "vibetuning-demo"))
        gate_name = os.getenv("STATSIG_WARM_RESUME_GATE", "warm_resume_enabled")
        cfg_name = os.getenv("STATSIG_BUDGET_CONFIG", "budget_cap_inr")
        status.warm_resume_enabled = bool(client.check_gate(user, gate_name))
        try:
            cfg = client.get_dynamic_config(user, cfg_name)
            cap = cfg.get_float("value", status.budget_cap_inr)
            status.budget_cap_inr = float(cap)
        except Exception:
            # Config missing → keep env default; gate still counts as a live read.
            pass
        status.statsig = True
        status.statsig_source = f"statsig:{gate_name}/{cfg_name}"
        status.notes["statsig"] = f"gate {gate_name}={status.warm_resume_enabled} cap={status.budget_cap_inr}"
    except Exception as exc:
        status.statsig = False
        status.notes["statsig"] = f"init failed: {exc.__class__.__name__}"
        _STATSIG = None


def status() -> FlagStatus:
    global _STATUS
    if _STATUS is not None:
        return _STATUS
    s = FlagStatus()
    s.budget_cap_inr = _num("BUDGET_CAP_INR", 150.0)
    s.warm_resume_enabled = _truthy("WARM_RESUME_ENABLED", True)
    s.cognee = _truthy("COGNEE_ENABLED", True)
    s.singlestore = _truthy(
        "SINGLESTORE_ENABLED",
        default=bool(os.getenv("SINGLESTORE_HOST") or os.getenv("DB_HOST")),
    )
    s.airbyte = _truthy("AIRBYTE_ENABLED", False)
    s.amplitude = _truthy(
        "AMPLITUDE_ENABLED",
        default=bool(os.getenv("AMPLITUDE_API_KEY")),
    )
    s.superlinked = _truthy(
        "SUPERLINKED_ENABLED",
        default=bool(os.getenv("SIE_API_KEY")),
    )
    s.whisper_gpu = _truthy("WHISPER_GPU_FALLBACK", False)
    _try_statsig(s)
    _STATUS = s
    return s


def refresh() -> FlagStatus:
    global _STATUS
    _STATUS = None
    return status()
