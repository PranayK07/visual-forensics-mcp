"""Lightweight stderr logging.

MCP servers communicate over stdout (stdio transport), so all diagnostic
logging MUST go to stderr to avoid corrupting the protocol stream.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger("visual_forensics")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that writes to stderr."""
    _configure_root()
    return logging.getLogger(f"visual_forensics.{name}")
