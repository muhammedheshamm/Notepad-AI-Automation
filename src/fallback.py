"""
fallback.py — BotCity template matching fallback for icon grounding.

Used automatically when Gemini AI grounding fails (GroundingError, API error,
or Stage 2 confidence below threshold).

BotCity uses OpenCV template matching (TM_CCOEFF_NORMED) to scan the live
desktop for a reference image and returns the best match coordinates.

Reference image setup:
    Save a clean PNG crop of the target icon to assets/notepad_icon.png.
    The image should be a tight crop of just the icon graphic (no label text).
"""

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Minimum template-match score for BotCity to accept a hit (0.0–1.0).
# Lower = more lenient; raise if you get false positives.
BOTCITY_THRESHOLD = 0.7


def find_with_botcity(
    reference_path: Path,
    matching: float = BOTCITY_THRESHOLD,
) -> Optional[tuple[int, int]]:
    """
    Scan the live desktop for `reference_path` using BotCity template matching.

    Args:
        reference_path: Path to the reference PNG of the icon to find.
        matching:       Minimum OpenCV match score (0.0–1.0). Default 0.7.

    Returns:
        (x, y) center pixel coordinates of the best match, or None if not found
        or if the reference image is missing / BotCity is unavailable.
    """
    try:
        from botcity.core import DesktopBot
    except ImportError:
        log.error(
            "BotCity not available — run: uv add botcity-framework-core"
        )
        return None

    reference_path = Path(reference_path)
    if not reference_path.exists():
        log.warning(
            "BotCity fallback: reference image not found at '%s'. "
            "Add a PNG crop of the icon to the assets/ folder.",
            reference_path,
        )
        return None

    log.info(
        "BotCity fallback: scanning desktop for '%s' (matching=%.2f)",
        reference_path.name,
        matching,
    )

    try:
        bot = DesktopBot()

        # Map the label directly to the reference file path so BotCity doesn't
        # need a 'resources/' folder — works from any working directory.
        label = reference_path.stem
        bot.state.map_images[label] = str(reference_path.resolve())

        result = bot.find(label, matching=matching, waiting_time=0)

        if result is None:
            log.warning(
                "BotCity: no match found (score < %.2f) — "
                "try lowering BOTCITY_THRESHOLD or re-capturing the reference.",
                matching,
            )
            return None

        # result is Box(left, top, width, height) — compute center
        cx = result.left + result.width // 2
        cy = result.top + result.height // 2
        log.info("BotCity match: center=(%d, %d)  box=%s", cx, cy, result)
        return (cx, cy)

    except Exception as exc:
        log.error("BotCity template matching error: %s", exc)
        return None
