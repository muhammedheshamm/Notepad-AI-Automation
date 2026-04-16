"""
screenshot.py — Desktop screen capture and annotation utilities.

Provides fast screenshot capture via mss and PIL-based annotation
for drawing detection markers on captured images.
"""

import sys
import time
import logging
from pathlib import Path
from io import BytesIO

import mss
import mss.tools
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger(__name__)

# Directory where annotated screenshots are saved (deliverable)
ANNOTATED_DIR = Path(__file__).parent / "annotated_screenshots"
ANNOTATED_DIR.mkdir(exist_ok=True)


def capture_desktop() -> Image.Image:
    """Capture the full primary monitor desktop as a PIL Image."""
    with mss.mss() as sct:
        # Monitor 1 is the primary display
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        # mss returns BGRA; convert to RGB PIL Image
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    log.debug("Captured desktop: %dx%d", img.width, img.height)
    return img


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to raw bytes for API upload."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def annotate_detection(
    img: Image.Image,
    x: int,
    y: int,
    label: str = "Detected",
    radius: int = 30,
    color: str = "#FF3B3B",
) -> Image.Image:
    """
    Draw a circle and label at the detected coordinates on a copy of the image.

    Args:
        img:    Source PIL Image (not modified in place).
        x, y:  Center of detected element in pixels.
        label: Text label shown near the marker.
        radius: Radius of the circle in pixels.
        color:  Hex color for the annotation.

    Returns:
        Annotated PIL Image copy.
    """
    annotated = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Outer filled circle (semi-transparent)
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=(*_hex_to_rgb(color), 60),
        outline=(*_hex_to_rgb(color), 230),
        width=3,
    )

    # Crosshair lines
    line_len = radius + 15
    draw.line([(x - line_len, y), (x + line_len, y)], fill=(*_hex_to_rgb(color), 200), width=2)
    draw.line([(x, y - line_len), (x, y + line_len)], fill=(*_hex_to_rgb(color), 200), width=2)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    # Draw label text with a dark background for readability
    draw_final = ImageDraw.Draw(annotated)
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text = f" {label} ({x}, {y}) "
    bbox = draw_final.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position label below-right of circle, keep within image bounds
    tx = min(x + radius + 5, img.width - text_w - 5)
    ty = min(y + radius + 5, img.height - text_h - 5)

    draw_final.rectangle([tx - 2, ty - 2, tx + text_w + 2, ty + text_h + 2], fill=(0, 0, 0, 180))
    draw_final.text((tx, ty), text, fill=_hex_to_rgb(color), font=font)

    return annotated


def save_annotated(img: Image.Image, filename: str) -> Path:
    """Save an annotated image to the annotated_screenshots directory."""
    out_path = ANNOTATED_DIR / filename
    img.save(out_path)
    log.info("Saved annotated screenshot: %s", out_path)
    return out_path


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (R, G, B) tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Demo entry point  (uv run demo)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Demo tool for generating annotated screenshot deliverables.

    Runs the full cascaded two-stage grounding pipeline for each position:
      Stage 1 — coarse region on full screenshot
      Stage 2 — precise localisation on the cropped & upscaled region
    Both stages are annotated on the saved image so you can see exactly
    what the model did at each step.

    If AI grounding fails, falls back to BotCity template matching
    (same behaviour as main.py).

    Usage:
        uv run python screenshot.py

    Move the Notepad icon to the desired position (top-left, center,
    bottom-right) before pressing ENTER at each prompt.
    Requires GEMINI_API_KEY in .env.
    """
    import os
    from dotenv import load_dotenv
    from src.grounding import init_client, find_element, GroundingError, _coarse_pass, _crop_and_upscale, CONFIDENCE_THRESHOLD
    from src.fallback import find_with_botcity

    load_dotenv()

    # ── Config (mirrors main.py) ──────────────────────────────────────────
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
    REFERENCE_IMAGE = Path(__file__).parent / "assets" / "notepad_icon.png"
    BOTCITY_FIRST: bool = True   # set True to try template matching before AI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    client = init_client(api_key)
    positions = ["top_left", "center", "bottom_right"]

    for position in positions:
        input(
            f"\n[DEMO] Move the Notepad icon to the {position.replace('_', '-')} area, "
            f"then press ENTER to capture..."
        )

        # sleep for 2 second
        time.sleep(2)

        print("  Capturing desktop...")
        screenshot = capture_desktop()

        coords = None
        used_botcity = False
        coarse_conf = 0.0
        used_crop = False
        coarse = None

        # ── BotCity-first (optional) ──────────────────────────────────────
        if BOTCITY_FIRST:
            print("  BotCity-first: trying template matching...")
            coords = find_with_botcity(REFERENCE_IMAGE)
            if coords:
                print(f"  BotCity match: {coords}")
                used_botcity = True

        # ── Stage 1: coarse pass ──────────────────────────────────────────
        if coords is None:
            print("  Stage 1: coarse grounding on full screenshot...")
            coarse = _coarse_pass(client, model, screenshot, NOTEPAD_TARGET)
            coarse_conf = coarse["confidence"] if coarse else 0.0

            if coarse and coarse["found"] and coarse_conf >= CONFIDENCE_THRESHOLD:
                print(f"  Stage 1: box={coarse['box']}  conf={coarse_conf:.2f}  → cropping region")
                _fine_img, (rx1, ry1, rx2, ry2) = _crop_and_upscale(screenshot, coarse["box"])
                used_crop = True
            else:
                print(f"  Stage 1: conf={coarse_conf:.2f} below {CONFIDENCE_THRESHOLD} → Stage 2 on full screen")
                rx1, ry1, rx2, ry2 = 0, 0, screenshot.width, screenshot.height
                used_crop = False

            # ── Stage 2: fine pass ────────────────────────────────────────
            print("  Stage 2: fine grounding...")
            try:
                coords = find_element(client, model, screenshot, NOTEPAD_TARGET)
            except GroundingError as exc:
                print(f"  AI grounding failed for '{position}': {exc}")

                # ── BotCity fallback ──────────────────────────────────────
                print("  Trying BotCity template fallback...")
                coords = find_with_botcity(REFERENCE_IMAGE)
                if coords:
                    print(f"  BotCity fallback match: {coords}")
                    used_botcity = True
                else:
                    print(f"  BotCity also failed — skipping '{position}'")
                    continue

        x, y = coords

        # ── Annotate both stages on the saved image ───────────────────────
        annotated = screenshot.copy()

        # Stage 1 coarse region — blue rectangle (skip if BotCity was used)
        if not used_botcity and coarse and coarse["found"]:
            draw = ImageDraw.Draw(annotated)
            W, H = screenshot.width, screenshot.height
            y_min, x_min, y_max, x_max = coarse["box"]
            bx1 = int(x_min / 1000 * W)
            by1 = int(y_min / 1000 * H)
            bx2 = int(x_max / 1000 * W)
            by2 = int(y_max / 1000 * H)
            draw.rectangle([bx1, by1, bx2, by2], outline="#4A90D9", width=3)
            draw.text(
                (bx1 + 4, by1 + 4),
                f"Stage 1  conf={coarse_conf:.2f}",
                fill="#4A90D9",
            )

        # Stage 2 / BotCity precise detection — red crosshair
        source = "BotCity" if used_botcity else f"AI S1={coarse_conf:.2f} crop={'yes' if used_crop else 'no'}"
        label = f"Notepad [{position}]  {source}"
        annotated = annotate_detection(annotated, x, y, label=label)

        filename = f"notepad_detected_{position}.png"
        path = save_annotated(annotated, filename)
        print(f"  Final coords: ({x}, {y})")
        print(f"  Saved: {path}")

    print("\n[DEMO] Done. Check annotated_screenshots/ for results.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
