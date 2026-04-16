"""
grounding.py — Cascaded visual grounding via Gemini.

Implements a two-stage ScreenSeekeR-style search inspired by ScreenSpot-Pro
(arXiv 2504.07981):

  Stage 1 (coarse) — full screenshot
    Gemini returns a large bounding box around the target region + confidence.

  Stage 2 (fine) — cropped & upscaled region (or full screen if Stage 1 failed)
    Gemini returns a precise bounding box + confidence within that crop.
    Stage 2 coordinates are then mapped back to original pixel space.

Fallback rule:
  - Stage 1 confidence < CONFIDENCE_THRESHOLD  →  Stage 2 runs on the full screen
  - Stage 2 confidence < CONFIDENCE_THRESHOLD  →  raises GroundingError

Works for ANY icon or UI element — just pass a plain-English description string.
"""

import json
import logging
import re
import time
from typing import Dict, Optional, Tuple

from google import genai
from google.genai import types
from PIL import Image

from screenshot import image_to_bytes

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3   # minimum acceptable confidence for each stage


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class GroundingError(Exception):
    """Raised when visual grounding cannot locate the target with sufficient confidence."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_COARSE_PROMPT = """\
You are a desktop UI automation assistant.

TASK: Find the approximate REGION on the desktop screenshot that contains the target element.
Return a generously-sized bounding box that covers the entire area where the element could be.
Also rate your confidence from 0.0 (no idea) to 1.0 (very sure).

Target element: {description}

NORMALIZED coordinates — 0 = top/left edge, 1000 = bottom/right edge.

Return ONLY a JSON object (no markdown, no explanation):

If found:
{{"found": true, "box": [y_min, x_min, y_max, x_max], "confidence": <0.0-1.0>}}

If NOT found:
{{"found": false, "confidence": 0.0}}
"""

_FINE_PROMPT = """\
You are a desktop UI automation assistant.

TASK: Find the PRECISE location of the target element in this image.
This image may be a zoomed-in crop of a larger screenshot — treat it as-is.
Also rate your confidence from 0.0 (no idea) to 1.0 (very sure).

Target element: {description}

NORMALIZED coordinates — 0 = top/left edge, 1000 = bottom/right edge.

Return ONLY a JSON object (no markdown, no explanation):

If found:
{{"found": true, "box": [y_min, x_min, y_max, x_max], "confidence": <0.0-1.0>}}

If NOT found:
{{"found": false, "confidence": 0.0}}
"""

_POPUP_DETECTION_PROMPT = """\
You are a desktop UI automation assistant. Analyze the screenshot below.

Your ONLY job is to detect small SYSTEM DIALOG BOXES that are blocking the workflow.
These are things like: Windows error dialogs, UAC prompts, "File already exists" confirmations,
"Are you sure?" prompts, or crash reports.

IMPORTANT rules:
- Do NOT flag Notepad or any text editor window — that is the intended application.
- Do NOT flag the Windows taskbar, desktop icons, or normal application windows.
- ONLY flag a small modal dialog box with OK/Cancel/Yes/No/Close buttons that is blocking the screen.

Return a bounding box around the dismiss button (OK, Close, X, Cancel, etc.) \
using NORMALIZED coordinates in the range 0 to 1000:
- 0 = top/left edge of the image
- 1000 = bottom/right edge of the image

Return ONLY a JSON object (no markdown, no explanation):

If a blocking system dialog exists:
{{"popup_exists": true, "description": "<what the dialog says>", "dismiss_box": [y_min, x_min, y_max, x_max], "action": "<click|escape|enter>"}}

If no blocking dialog (including if only Notepad is visible):
{{"popup_exists": false}}
"""


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def init_client(api_key: str) -> genai.Client:
    """Create and return a Gemini client."""
    client = genai.Client(api_key=api_key)
    log.info("Gemini client initialised")
    return client


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def find_element(
    client: genai.Client,
    model: str,
    screenshot: Image.Image,
    description: str,
) -> Tuple[int, int]:
    """
    Locate any UI element by plain-English description using cascaded grounding.

    Stage 1: coarse pass on the full screenshot.
    Stage 2: fine pass on the cropped region (or full screen if Stage 1 is uncertain).

    Args:
        client:      Gemini client from init_client().
        model:       Model name string.
        screenshot:  Full desktop screenshot as a PIL Image.
        description: Plain-English description of the target, e.g. "Notepad icon"
                     or "Chrome browser shortcut" — any string works.

    Returns:
        (x, y) center pixel coordinates in the original screenshot space.

    Raises:
        GroundingError: If Stage 2 returns no result or confidence < CONFIDENCE_THRESHOLD.
    """
    log.info("Cascaded grounding: '%s'", description)

    # ------------------------------------------------------------------
    # Stage 1 — coarse localisation on full screenshot
    # ------------------------------------------------------------------
    log.debug("Stage 1: coarse pass on full screenshot")
    coarse = _coarse_pass(client, model, screenshot, description)
    coarse_conf = coarse["confidence"] if coarse else 0.0

    if coarse and coarse["found"] and coarse_conf >= CONFIDENCE_THRESHOLD:
        log.info("Stage 1: box=%s conf=%.2f — cropping region for Stage 2",
                 coarse["box"], coarse_conf)
        fine_img, (rx1, ry1, rx2, ry2) = _crop_and_upscale(screenshot, coarse["box"])
    else:
        log.info("Stage 1: conf=%.2f below %.2f — Stage 2 will run on full screen",
                 coarse_conf, CONFIDENCE_THRESHOLD)
        fine_img = screenshot
        rx1, ry1, rx2, ry2 = 0, 0, screenshot.width, screenshot.height

    # ------------------------------------------------------------------
    # Stage 2 — fine localisation
    # ------------------------------------------------------------------
    log.debug("Stage 2: fine pass")
    fine = _fine_pass(client, model, fine_img, description)

    if fine is None or not fine.get("found"):
        raise GroundingError(
            f"'{description}': Stage 2 could not locate the element. "
            "Make sure the icon is visible on the desktop."
        )

    fine_conf = float(fine.get("confidence", 0.0))
    if fine_conf < CONFIDENCE_THRESHOLD:
        raise GroundingError(
            f"'{description}': Stage 2 confidence {fine_conf:.2f} is below "
            f"the threshold {CONFIDENCE_THRESHOLD}. The element may be partially "
            "hidden, too small, or not present."
        )

    # ------------------------------------------------------------------
    # Map Stage 2 coordinates back to original pixel space
    # ------------------------------------------------------------------
    y_min, x_min, y_max, x_max = fine["box"]
    cx_norm = (x_min + x_max) / 2   # center in 0-1000 space within fine_img
    cy_norm = (y_min + y_max) / 2

    region_w = rx2 - rx1
    region_h = ry2 - ry1

    full_x = rx1 + int(cx_norm / 1000 * region_w)
    full_y = ry1 + int(cy_norm / 1000 * region_h)
    full_x = max(0, min(full_x, screenshot.width - 1))
    full_y = max(0, min(full_y, screenshot.height - 1))

    log.info(
        "Grounding done: stage1_conf=%.2f  stage2_conf=%.2f → pixel (%d, %d)",
        coarse_conf, fine_conf, full_x, full_y,
    )
    return (full_x, full_y)


def detect_blocking_popup(
    client: genai.Client,
    model: str,
    screenshot: Image.Image,
) -> Optional[Dict]:
    """
    Detect any unexpected system popup/dialog on screen.
    Returns a dict with dismiss coordinates and action, or None if nothing found.
    """
    raw = _query_model(client, model, screenshot, _POPUP_DETECTION_PROMPT)
    if raw is None:
        return None

    parsed = _parse_json(raw)
    if parsed is None:
        log.warning("Could not parse popup response")
        return None

    if not parsed.get("popup_exists"):
        return None

    w, h = screenshot.width, screenshot.height
    dismiss_x, dismiss_y = None, None
    dismiss_box = parsed.get("dismiss_box")
    if dismiss_box and len(dismiss_box) == 4:
        y_min, x_min, y_max, x_max = dismiss_box
        dismiss_x = int(((x_min + x_max) / 2) / 1000 * w)
        dismiss_y = int(((y_min + y_max) / 2) / 1000 * h)

    result = {
        "description": parsed.get("description", "unknown"),
        "action": parsed.get("action", "click"),
        "dismiss_x": dismiss_x,
        "dismiss_y": dismiss_y,
    }
    log.info("Popup: '%s' — dismiss at (%s, %s) via %s",
             result["description"], dismiss_x, dismiss_y, result["action"])
    return result


# ---------------------------------------------------------------------------
# Private — cascaded grounding helpers
# ---------------------------------------------------------------------------

def _coarse_pass(
    client: genai.Client,
    model: str,
    screenshot: Image.Image,
    description: str,
) -> Optional[Dict]:
    """Run Stage 1: return {'found', 'box', 'confidence'} or None on error."""
    prompt = _COARSE_PROMPT.format(description=description)
    raw = _query_model(client, model, screenshot, prompt)
    if raw is None:
        return None
    parsed = _parse_json(raw)
    if parsed is None:
        return None
    confidence = float(parsed.get("confidence", 0.0))
    if not parsed.get("found") or not parsed.get("box"):
        return {"found": False, "box": None, "confidence": confidence}
    return {"found": True, "box": parsed["box"], "confidence": confidence}


def _fine_pass(
    client: genai.Client,
    model: str,
    image: Image.Image,
    description: str,
) -> Optional[Dict]:
    """Run Stage 2: return {'found', 'box', 'confidence'} or None on error."""
    prompt = _FINE_PROMPT.format(description=description)
    raw = _query_model(client, model, image, prompt)
    if raw is None:
        return None
    parsed = _parse_json(raw)
    if parsed is None:
        return None
    confidence = float(parsed.get("confidence", 0.0))
    if not parsed.get("found") or not parsed.get("box"):
        return {"found": False, "box": None, "confidence": confidence}
    return {"found": True, "box": parsed["box"], "confidence": confidence}


def _crop_and_upscale(
    screenshot: Image.Image,
    box: list,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop the coarse region from the screenshot (with 15% padding) and
    upscale it back to the original image dimensions.

    Upscaling ensures the model receives the same input size in Stage 2,
    and makes small elements appear much larger and easier to localise.

    Returns:
        (upscaled_crop, (px1, py1, px2, py2)) — pixel coords of the crop
        in the original screenshot (used to map Stage 2 coords back).
    """
    y_min, x_min, y_max, x_max = box
    W, H = screenshot.width, screenshot.height

    px1 = int(x_min / 1000 * W)
    py1 = int(y_min / 1000 * H)
    px2 = int(x_max / 1000 * W)
    py2 = int(y_max / 1000 * H)

    # 15% padding so we don't accidentally clip the target
    pad_x = max(30, int((px2 - px1) * 0.15))
    pad_y = max(30, int((py2 - py1) * 0.15))
    px1 = max(0, px1 - pad_x)
    py1 = max(0, py1 - pad_y)
    px2 = min(W, px2 + pad_x)
    py2 = min(H, py2 + pad_y)

    crop = screenshot.crop((px1, py1, px2, py2))
    upscaled = crop.resize((W, H), Image.LANCZOS)

    log.debug("Crop region: (%d,%d)–(%d,%d), upscaled to %dx%d", px1, py1, px2, py2, W, H)
    return upscaled, (px1, py1, px2, py2)


# ---------------------------------------------------------------------------
# Private — model I/O helpers
# ---------------------------------------------------------------------------

def _query_model(
    client: genai.Client,
    model: str,
    img: Image.Image,
    prompt: str,
    retries: int = 2,
) -> Optional[str]:
    """Send image + prompt to Gemini and return raw response text."""
    img_bytes = image_to_bytes(img, fmt="PNG")
    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    text_part = types.Part.from_text(text=prompt)

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[img_part, text_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    max_output_tokens=512,
                ),
            )
            return response.text
        except Exception as exc:
            if attempt < retries:
                wait = 2 ** attempt
                log.warning("Gemini error (attempt %d/%d): %s — retrying in %ds",
                            attempt + 1, retries + 1, exc, wait)
                time.sleep(wait)
            else:
                log.error("Gemini failed after %d attempts: %s", retries + 1, exc)
                return None
    return None


def _parse_json(text: str) -> Optional[Dict]:
    """Parse a JSON object from model output, stripping any markdown fences."""
    if not text:
        return None
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as exc:
        log.debug("JSON parse error: %s | text: %s", exc, clean[:200])
        return None
