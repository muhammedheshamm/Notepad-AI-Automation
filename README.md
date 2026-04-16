# Vision-Based Desktop Automation

A Python application that uses **Gemini** as a visual grounding engine to dynamically locate desktop icons and automate Notepad — without hardcoded coordinates, template images, or OCR rules.

Built for Windows 10/11 at 1920×1080.

---

## How It Works

Instead of searching for a specific image or hardcoded position, a screenshot is sent to Gemini with a plain-English description of the target. Gemini reasons about the screen like a human and returns pixel coordinates.

### Cascaded Grounding (ScreenSeekeR-style)

Inspired by the **ScreenSeekeR** approach from [ScreenSpot-Pro (arXiv 2504.07981)](https://arxiv.org/abs/2504.07981), grounding runs in two stages:

```
Stage 1 — Coarse (full screenshot)
  Gemini returns a large bounding box around the target region + confidence score.
  If confidence >= 0.3 → the region is cropped and upscaled for Stage 2.
  If confidence < 0.3  → Stage 2 runs on the full screen instead.

Stage 2 — Fine (cropped & upscaled region, or full screen)
  Gemini returns a precise bounding box + confidence within the image it receives.
  Coordinates are mapped back to the original pixel space.
  If confidence < 0.3  → raises GroundingError with a clear message.
```

**Why upscale the crop?** A desktop icon is ~64×64 px on a 1920×1080 screen — only 0.2% of the image. Cropping the Stage 1 region and stretching it back to full resolution makes the target appear ~4× larger, giving the model far more detail to work with.

The same engine handles **unexpected popups**: a screenshot is sent to Gemini asking whether any system dialog is blocking the workflow. If one is found, Gemini returns coordinates to dismiss it — without needing to know what the popup looks like in advance.

### Icon Coordinate Caching

After the first successful grounding, the icon's (x, y) coordinates are cached and reused for all subsequent posts — no API call needed. The cache is cleared only if a launch fails, triggering a fresh grounding pass.

---

## Project Structure

```
tjm-project/
├── main.py                  # Orchestrator / entry point
├── screenshot.py            # mss capture + PIL annotation + demo tool
├── src/                     # Core library modules
│   ├── __init__.py
│   ├── grounding.py         # Cascaded Gemini visual grounding engine
│   ├── automation.py        # Notepad control (launch, type, save, close)
│   ├── api_client.py        # JSONPlaceholder API client with fallback
│   └── fallback.py          # BotCity template-matching fallback
├── assets/                  # Reference images for BotCity template matching
│   └── notepad_icon.png     # Clean PNG crop of the Notepad icon (add manually)
├── pyproject.toml           # uv project config + dependencies
├── requirements.txt         # Pinned dependency lockfile (pip-compatible)
├── .python-version          # Python 3.11
├── .env.example             # API key template
├── .gitignore
└── annotated_screenshots/   # Deliverable: annotated detection screenshots
```

---

## Prerequisites

- Windows 10 or 11 at 1920×1080 resolution
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A **Notepad shortcut icon on the desktop** (right-click desktop → New → Shortcut → `notepad.exe`)
- Notepad configured to **always open a new session** (Settings → On startup → Open a new window)
- A Gemini API key (free tier works)

---

## Setup

### 1. Get a Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with a Google account
3. Click **Create API key**
4. Copy the key

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
GEMINI_API_KEY=AIza...your_key_here
```

Optionally override the model:

```
GEMINI_MODEL=gemini-2.5-flash-lite    # default — highest free tier quota
```

### 3. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Running the Automation

Make sure the Notepad shortcut icon is visible on the desktop, then:

```bash
uv run python main.py
```

The script will:
1. Fetch 10 blog posts from JSONPlaceholder (falls back to generated posts if offline)
2. Pause 3 seconds (time to switch focus to the desktop)
3. For each post: ground icon → launch Notepad → paste content → save → close
4. Save files as `post_1.txt` through `post_10.txt` in this project folder
5. Print a success/failure summary

A full run log is written to `automation.log`.

---

## BotCity Template Matching Fallback

When Gemini AI grounding fails — due to a `GroundingError`, low confidence, or an API problem — the automation automatically falls back to **BotCity template matching**, which uses OpenCV to scan the live desktop for a reference image of the icon.

### Setup

Place a clean PNG crop of the Notepad icon (just the icon graphic, no label text) at:

```
assets/notepad_icon.png
```

The image should be a tight crop around the icon at its natural desktop size (~64–80 px square).

### BotCity-First Mode

To try template matching **before** AI grounding (saves API calls when the reference image is reliable), set the flag at the top of `main.py` or `screenshot.py`:

```python
BOTCITY_FIRST: bool = True
```

If BotCity finds the icon, Gemini is never called. If BotCity fails, the flow falls through to full AI grounding automatically.

### Full Priority Order

```
1. FIXED_ICON          — hardcoded coords, zero API calls (testing only)
2. Cached coords        — reused from the previous post, zero API calls
3. BotCity template     — only if BOTCITY_FIRST = True
4. Gemini AI grounding  — cascaded Stage 1 + Stage 2, up to 3 retries
5. BotCity template     — always tried as last resort if AI fails
```

Tune the match sensitivity in `fallback.py`:

```python
BOTCITY_THRESHOLD: float = 0.7   # lower = more lenient, higher = stricter
```

---

## No-AI Testing Mode

To test the automation without any API calls, set `FIXED_ICON` in `main.py`:

```python
FIXED_ICON: Optional[tuple[int, int]] = (45, 45)   # pixel coords of the Notepad icon
```

All grounding and popup-detection calls are skipped entirely. To find the exact coordinates of your icon, run:

```bash
uv run python -c "import time, pyautogui; time.sleep(3); print(pyautogui.position())"
```

Hover over the icon within 3 seconds. Set `FIXED_ICON = None` to re-enable full AI grounding.

---

## Generating Annotated Screenshots (Deliverable)

The demo tool captures the desktop, runs cascaded grounding, annotates the detected location, and saves the result to `annotated_screenshots/`:

```bash
uv run python screenshot.py
```

You will be prompted three times (top-left, center, bottom-right). Move the Notepad icon to each position before pressing Enter.

---

## Module Reference

### `grounding.py`

| Function | Description |
|----------|-------------|
| `init_client(api_key)` | Create a Gemini client |
| `find_element(client, model, screenshot, description)` | Cascaded two-stage grounding — returns `(x, y)` or raises `GroundingError` |
| `detect_blocking_popup(client, model, screenshot)` | Detect any system dialog and return dismiss coordinates, or `None` |

`find_element` is **element-agnostic** — pass any plain-English description:  
`"Notepad icon"`, `"Chrome shortcut"`, `"VS Code taskbar button"`, etc.

`GroundingError` is raised (not returned) when Stage 2 confidence is below `0.3`, carrying a clear human-readable message.

### `automation.py`

| Function | Description |
|----------|-------------|
| `launch_notepad(x, y)` | Double-click icon, wait up to 8s for window |
| `type_text(text)` | Clear Notepad and paste content via clipboard |
| `save_file(filepath)` | Ctrl+S → Save As dialog → paste path → Enter |
| `close_notepad()` | WM_CLOSE → Tab+Enter (Don't Save) → taskkill if needed |
| `is_notepad_running()` | Returns True if any Notepad window is open |

### `api_client.py`

| Function | Description |
|----------|-------------|
| `fetch_posts(limit)` | Fetch from JSONPlaceholder; falls back to generated posts if offline |
| `format_post_content(post)` | Format `{"title", "body"}` dict as file content |
| `post_filename(post)` | Returns `"post_{id}.txt"` |
| `validate_post(post)` | Returns True if post has `id`, `title`, `body` fields |

### `fallback.py`

| Function | Description |
|----------|-------------|
| `find_with_botcity(reference_path, matching)` | Scan the live desktop for the reference PNG using BotCity OpenCV template matching; returns `(x, y)` or `None` |

`BOTCITY_THRESHOLD = 0.7` — default minimum match score. Adjust in `fallback.py` if you get false positives (raise it) or misses (lower it).

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Stage 1 confidence < 0.3 | Stage 2 runs on the full screen instead of a crop |
| Stage 2 confidence < 0.3 | `GroundingError` raised with descriptive message; BotCity fallback triggered |
| Icon not found / grounding error | Retry up to 3× with 1.5s delay; BotCity fallback tried; skip post on all failures |
| BotCity reference image missing | Warning logged; BotCity step skipped, AI grounding proceeds normally |
| Notepad doesn't open | 8s timeout on window title check; retry or skip |
| Unexpected system popup | Gemini detects and dismisses automatically (up to 2 cycles) |
| "Save changes?" dialog on close | Tab → Enter ("Don't save"); `taskkill` as last resort |
| JSONPlaceholder API unavailable | Falls back to 10 generated numbered posts automatically |
| Gemini API error | Exponential backoff (2 retries); returns `None` / raises on final failure |
| Cached icon coords stale | Cleared on failure; fresh grounding triggered on next attempt |

---

## Available Gemini Models

Models confirmed available on this API key (vision-capable, relevant to this project):

| Model | Notes |
|-------|-------|
| `gemini-2.5-flash-lite` | **Default** — highest free tier quota |
| `gemini-2.5-flash` | Higher quality, lower free quota |
| `gemini-2.5-pro` | Most capable |
| `gemini-3-flash-preview` | Gemini 3 |
| `gemini-3.1-flash-lite-preview` | Gemini 3.1 lite |
| `gemini-3.1-pro-preview` | Gemini 3.1 Pro |
| `gemini-2.5-computer-use-preview-10-2025` | Purpose-built for screen automation |

Override via `.env`:
```
GEMINI_MODEL=gemini-2.5-computer-use-preview-10-2025
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Google AI Studio API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model to use |

`FIXED_ICON`, `BOTCITY_FIRST`, `REFERENCE_IMAGE`, and all retry/timeout constants are configured at the top of `main.py`.

---

## Discussion Notes

**Why VLM-based grounding instead of template matching?**  
Template matching requires a reference image and breaks when the icon changes size, theme, or background. A VLM reasons semantically — it finds "the Notepad icon" the same way a human would scan the screen. The description can be changed to target any element without touching the automation code.

**Why keep BotCity template matching as a fallback?**  
Template matching is deterministic, zero-latency, and works offline. When the reference image is a reliable match for the icon, it provides a fast safety net for the cases where the VLM has low confidence, hits a rate limit, or encounters an API error. The two approaches are complementary: VLM for flexibility, template matching for reliability.

**Why two stages?**  
A 64×64 icon on a 1920×1080 screenshot occupies ~0.2% of the image. Stage 1 narrows the search area; Stage 2 receives a zoomed view where the target is ~4× larger and easier to precisely localise. This mirrors how humans scan: first locate the region, then focus on the detail.

**Known failure cases:**
- Icons smaller than ~20px may still challenge Stage 2 at high confidence
- Heavily occluded icons (< 30% visible) — retry logic helps but may ultimately fail
- Non-English Windows locales — extend `NOTEPAD_TARGET` with localised terms

**API efficiency:**  
Grounding runs twice on post 1 (Stage 1 + Stage 2 = 2 calls). Posts 2–10 use cached coordinates (0 calls). Total grounding API usage per full run: 2 calls.
