"""Git revision helpers shared across the crawler."""

from __future__ import annotations

import logging
import subprocess  # nosec: B404
from pathlib import Path

logger = logging.getLogger(__name__)


def _crawler_revision(*candidates: Path) -> str | None:
    """Return the HEAD Git revision for the first valid crawler directory candidate."""
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            result = subprocess.run(  # nosec: B607 B602 B603
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:
            logger.debug("Cannot read crawler revision from %s: %s", candidate, exc)
            continue
        rev = result.stdout.strip()
        if rev:
            return rev
    return None
