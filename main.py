"""
main.py — Vision-based desktop automation orchestrator.

Workflow for each post:
  1. Fetch 10 posts from JSONPlaceholder.
  2. Launch Notepad (via fixed coords or Gemini visual grounding).
  3. Paste post content, save as post_{id}.txt, close Notepad.
  4. Repeat.

Usage:
    uv run python main.py

Prerequisites:
    GEMINI_API_KEY in .env  (only needed when FIXED_ICON is None)
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pyautogui
from dotenv import load_dotenv

from api_client import fetch_posts, format_post_content, post_filename
from automation import launch_notepad, type_text, save_file, close_notepad, is_notepad_running
from fallback import find_with_botcity
from grounding import init_client, find_element, detect_blocking_popup, GroundingError
from screenshot import capture_desktop

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ICON_FIND_ATTEMPTS  = 3    # retries when Gemini cannot locate the icon
ICON_RETRY_DELAY        = 1  # seconds between icon-find retries
MAX_POPUP_DISMISS_CYCLES = 2   # max Gemini popup-dismiss loops per step
NOTEPAD_TARGET = (
    "Find the Windows Notepad application shortcut icon on the desktop. "

    "VISUAL DESCRIPTION — what this icon looks like: "
    "A realistic 3D-rendered notepad/paper block viewed at a slight angle. "
    "The overall shape is a near-perfect SQUARE or slightly portrait rectangle — "
    "NOT a wide landscape rectangle like a folder. "

    "TOP EDGE: A dark grey or black SPIRAL COIL BINDING runs horizontally across "
    "the very top of the icon — this is the most distinctive feature. "
    "The coil appears as a series of small dark loops or bumps along the top border. "

    "PAGE BODY: The page surface is off-white, cream, or very light grey. "
    "Across the page body there are multiple thin HORIZONTAL BLUE RULED LINES, "
    "evenly spaced, running left to right — like a lined notebook page. "
    "These lines are light blue, similar to standard ruled notebook paper. "

    "CORNERS: One corner (typically bottom-right or top-right) has a subtle "
    "dog-ear fold, giving it a realistic paper look. "

    "SHORTCUT ARROW: A small blue-and-white Windows shortcut arrow icon "
    "may appear overlaid on the bottom-left corner of the icon. "

    "CRITICAL DIFFERENCES FROM FOLDERS: "
    "- A folder icon is wide/landscape shaped and yellow/golden coloured with a tab on top. "
    "- This Notepad icon is square shaped, off-white/cream coloured, and has a dark spiral coil on top. "
    "- A folder has NO lines on its body. This icon has clearly visible blue horizontal ruled lines. "
    "- A folder has NO spiral coil. This icon always has the coil along the top edge. "
    "- If you see a yellow or golden icon, it is a folder — do NOT return it. "

    "CONFIDENCE RULES: "
    "- Only return found: true if you can see the spiral coil AND the blue ruled lines on a square off-white icon. "
    "- If you see only a square icon without both of these features clearly visible, return found: false. "
    "- Do NOT rely on the text label beneath the icon to identify it — the label may say anything. "
    "- A wrong detection is far worse than returning found: false — prefer caution. "
)

# No-AI mode: set to (x, y) to skip all Gemini calls and use a fixed position.
# used for testing purposes
# Set to None to enable full AI grounding.
FIXED_ICON: Optional[tuple[int, int]] = None   # ← adjust or set to None

PROJECT_DIR = Path(__file__).parent

# Reference image used by the BotCity template-matching fallback.
# Place a clean PNG crop of the icon at assets/notepad_icon.png.
REFERENCE_IMAGE = PROJECT_DIR / "assets" / "notepad_icon.png"

# Set to True to try BotCity template matching BEFORE Gemini AI grounding.
# Useful when the reference image is reliable and you want to save API calls.
# If BotCity fails, the flow still falls through to Gemini grounding.
BOTCITY_FIRST: bool = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROJECT_DIR / "automation.log", encoding="utf-8"),
        ],
    )

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Popup handling
# ---------------------------------------------------------------------------

def handle_popups(client, model: str) -> None:
    """Detect and dismiss unexpected popups using Gemini. No-op in no-AI mode."""
    if FIXED_ICON is not None:
        return

    for cycle in range(MAX_POPUP_DISMISS_CYCLES):
        popup = detect_blocking_popup(client, model, capture_desktop())
        if popup is None:
            if cycle > 0:
                log.info("All popups cleared after %d cycle(s)", cycle)
            return

        action = popup.get("action", "click")
        dx, dy = popup.get("dismiss_x"), popup.get("dismiss_y")
        log.info("Popup: '%s' — dismissing via %s", popup.get("description"), action)

        if action == "escape":
            pyautogui.press("escape")
        elif action == "enter":
            pyautogui.press("enter")
        elif action == "click" and dx is not None and dy is not None:
            pyautogui.click(dx, dy)
        else:
            pyautogui.press("escape")

        time.sleep(0.5)

    log.warning("Reached max popup dismiss cycles; proceeding anyway")


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def find_and_launch(
    client,
    model: str,
    post_id: int,
    cached_coords: Optional[tuple[int, int]] = None,
) -> tuple[bool, Optional[tuple[int, int]]]:
    """
    Launch Notepad, returning (success, coords_to_cache).

    Order of attempts:
      1. Fixed coords (no-AI mode)
      2. Cached coords from previous post (no extra API call)
      3. BotCity template matching   ← only if BOTCITY_FIRST=True
      4. Gemini grounding (up to MAX_ICON_FIND_ATTEMPTS times)
      5. BotCity template matching   ← always as last-resort fallback
    """
    # No-AI mode
    if FIXED_ICON is not None:
        fx, fy = FIXED_ICON
        log.info("[Post %d] No-AI mode — fixed coords (%d, %d)", post_id, fx, fy)
        if launch_notepad(fx, fy):
            return True, FIXED_ICON
        log.error("[Post %d] Fixed coords did not open Notepad", post_id)
        return False, FIXED_ICON

    # Cached coords (fast path)
    if cached_coords is not None:
        cx, cy = cached_coords
        log.info("[Post %d] Using cached coords (%d, %d)", post_id, cx, cy)
        if launch_notepad(cx, cy):
            return True, cached_coords
        log.warning("[Post %d] Cached coords failed — re-grounding", post_id)

    # BotCity-first mode: try template matching before Gemini
    if BOTCITY_FIRST:
        log.info("[Post %d] BotCity-first mode — trying template matching", post_id)
        fb_coords = find_with_botcity(REFERENCE_IMAGE)
        if fb_coords is not None:
            fx, fy = fb_coords
            log.info("[Post %d] BotCity found icon at (%d, %d) — launching", post_id, fx, fy)
            if launch_notepad(fx, fy):
                return True, (fx, fy)
            log.warning("[Post %d] BotCity coords did not open Notepad — falling back to AI", post_id)
        else:
            log.warning("[Post %d] BotCity-first: no match — falling back to AI grounding", post_id)

    # Gemini grounding (slow path)
    coords: Optional[tuple[int, int]] = None
    for attempt in range(1, MAX_ICON_FIND_ATTEMPTS + 1):
        log.info("[Post %d] AI grounding attempt %d/%d", post_id, attempt, MAX_ICON_FIND_ATTEMPTS)
        try:
            coords = find_element(client, model, capture_desktop(), NOTEPAD_TARGET)
        except GroundingError as exc:
            log.warning("[Post %d] Grounding failed (attempt %d): %s", post_id, attempt, exc)
            coords = None

        if coords is None:
            if attempt < MAX_ICON_FIND_ATTEMPTS:
                time.sleep(ICON_RETRY_DELAY)
            continue

        x, y = coords
        log.info("[Post %d] AI found icon at (%d, %d) — launching", post_id, x, y)
        if launch_notepad(x, y):
            return True, (x, y)

        log.warning("[Post %d] Click did not open Notepad (attempt %d)", post_id, attempt)
        coords = None
        time.sleep(ICON_RETRY_DELAY)

    # ── BotCity template-matching fallback ─────────────────────────────────
    log.warning("[Post %d] AI grounding exhausted — trying BotCity template fallback", post_id)
    fb_coords = find_with_botcity(REFERENCE_IMAGE)
    if fb_coords is not None:
        fx, fy = fb_coords
        log.info("[Post %d] BotCity found icon at (%d, %d) — launching", post_id, fx, fy)
        if launch_notepad(fx, fy):
            return True, (fx, fy)
        log.error("[Post %d] BotCity coords did not open Notepad", post_id)
    else:
        log.error(
            "[Post %d] BotCity fallback also failed. "
            "Make sure assets/notepad_icon.png exists and is a clean crop of the icon.",
            post_id,
        )

    log.error("[Post %d] Could not launch Notepad", post_id)
    return False, cached_coords


# ---------------------------------------------------------------------------
# Per-post workflow
# ---------------------------------------------------------------------------

def process_post(
    post: Dict,
    client,
    model: str,
    cached_icon_coords: Optional[tuple[int, int]] = None,
) -> tuple[bool, Optional[tuple[int, int]]]:
    """
    Full cycle for one post: launch → type → save → close.
    Returns (success, coords_to_cache).
    """
    pid = post["id"]
    log.info("=" * 60)
    log.info("Processing post %d: '%s'", pid, post["title"][:50])
    log.info("=" * 60)

    # 1. Launch
    launched, new_coords = find_and_launch(client, model, pid, cached_icon_coords)
    if not launched:
        return False, None

    # 2. Type
    try:
        type_text(format_post_content(post))
    except Exception as exc:
        log.error("[Post %d] Failed to type content: %s", pid, exc)
        close_notepad()
        return False, new_coords

    # 3. Save
    save_path = PROJECT_DIR / post_filename(post)
    try:
        save_file(save_path)
    except Exception as exc:
        log.error("[Post %d] Failed to save: %s", pid, exc)
        close_notepad()
        return False, new_coords

    # 4. Close
    close_notepad()
    time.sleep(0.5)

    log.info("[Post %d] Done.", pid)
    return True, new_coords


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    load_dotenv()

    log.info("Vision-Based Desktop Automation — starting up")
    log.info("Project dir: %s", PROJECT_DIR)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if FIXED_ICON is not None:
        log.info("No-AI mode — fixed icon coords %s (zero API calls)", FIXED_ICON)
    elif BOTCITY_FIRST:
        log.info("BotCity-first mode — template matching before AI (model: %s)", model)
    else:
        log.info("AI mode — Gemini model: %s", model)

    client = init_client(api_key)

    try:
        posts = fetch_posts(limit=10)
    except Exception as exc:
        log.error("Failed to fetch posts: %s", exc)
        sys.exit(1)

    log.info("Will process %d posts", len(posts))
    log.info("Starting in 3 seconds — make sure the desktop is visible...")
    time.sleep(3)

    results = {"success": 0, "failed": 0}
    icon_coords: Optional[tuple[int, int]] = None

    for post in posts:
        success, icon_coords = process_post(post, client, model, icon_coords)
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
            log.warning("Skipping post %d due to errors", post["id"])
            icon_coords = None  # reset cache on failure

        time.sleep(1.0)

    log.info("")
    log.info("=" * 60)
    log.info("Automation complete.")
    log.info("  Succeeded: %d / %d", results["success"], len(posts))
    log.info("  Failed:    %d / %d", results["failed"], len(posts))
    log.info("  Files in:  %s", PROJECT_DIR)
    log.info("=" * 60)

    if results["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
