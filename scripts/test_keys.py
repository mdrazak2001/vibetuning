"""Probe Amplitude + Superlinked SIE with the configured keys. Prints status only, never keys."""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.events import _amplitude  # noqa: E402
from agent.flags import status  # noqa: E402

f = status()
print("amplitude flag:", f.amplitude, "superlinked flag:", f.superlinked)
_amplitude("key_probe", {"note": "mission control key check"}, "probe", None)
print("amplitude:", f.notes.get("amplitude"))
print("sie_sdk installed:", importlib.util.find_spec("sie_sdk") is not None)
