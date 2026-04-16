"""
automation.py — Notepad desktop automation via pyautogui and pygetwindow.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import pyautogui
import pygetwindow as gw
import pyperclip

log = logging.getLogger(__name__)

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

LAUNCH_TIMEOUT    = 8.0   # seconds to wait for Notepad to open
SAVE_DIALOG_SETTLE = 1.2  # seconds for the Save As dialog to appear


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def _get_notepad() -> Optional[object]:
    """Return the first Notepad window found, or None."""
    windows = gw.getWindowsWithTitle("Notepad")
    if windows:
        return windows[0]
    for w in gw.getAllWindows():
        if "notepad" in (w.title or "").lower():
            return w
    return None


def _activate(win) -> None:
    """Bring a window to the foreground and wait for focus."""
    try:
        win.activate()
        time.sleep(0.4)
    except Exception as exc:
        log.debug("activate() error: %s", exc)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def is_notepad_running() -> bool:
    """Return True if a Notepad window is currently open."""
    return _get_notepad() is not None


def launch_notepad(x: int, y: int, timeout: float = LAUNCH_TIMEOUT) -> bool:
    """Double-click the Notepad icon at (x, y) and wait for the window to open."""
    log.info("Double-clicking Notepad icon at (%d, %d)", x, y)
    pyautogui.moveTo(x, y, duration=0.3)
    time.sleep(0.2)
    pyautogui.doubleClick(x, y, interval=0.1)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _get_notepad():
            log.info("Notepad window confirmed open")
            return True
        time.sleep(0.3)

    log.warning("Notepad did not open within %.1fs", timeout)
    return False


def type_text(text: str) -> None:
    """Clear the Notepad window and paste text via clipboard."""
    win = _get_notepad()
    if win:
        _activate(win)

    log.info("Pasting content (%d chars) into Notepad", len(text))
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.2)


def save_file(filepath: Path) -> bool:
    """
    Save via Ctrl+S → Save As dialog → paste full path → Enter.

    Notepad is configured to always start a new (untitled) session so Ctrl+S
    always opens the Save As dialog.  The file is deleted beforehand to
    prevent the overwrite sub-dialog (an extra Enter would land in the text
    area and dirty the document after the save).

    Returns True if the file exists on disk after saving.
    """
    win = _get_notepad()
    if win:
        _activate(win)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        filepath.unlink()

    log.info("Saving to: %s", filepath)
    pyperclip.copy(str(filepath))

    pyautogui.hotkey("ctrl", "s")
    time.sleep(SAVE_DIALOG_SETTLE)

    pyautogui.hotkey("ctrl", "v")   # paste full path into the filename field
    time.sleep(0.3)
    pyautogui.press("enter")        # confirm save
    time.sleep(0.6)

    saved = filepath.exists()
    if saved:
        log.info("File saved: %s (%d bytes)", filepath.name, filepath.stat().st_size)
    else:
        log.warning("File not found after save: %s", filepath)
    return saved


def close_notepad() -> None:
    """
    Close Notepad via win.close() (WM_CLOSE).

    After a clean save the document has no unsaved changes so it closes
    immediately.  If a "save changes?" dialog appears, Tab → Enter dismisses
    it ("Don't save").  taskkill is the last resort.
    """
    win = _get_notepad()
    if not win:
        return

    log.info("Closing Notepad")
    try:
        win.close()
    except Exception as exc:
        log.debug("win.close() failed: %s — trying Alt+F4", exc)
        _activate(win)
        pyautogui.hotkey("alt", "f4")

    time.sleep(1.0)

    if _get_notepad():
        log.debug("Save dialog detected — pressing Don't Save")
        pyautogui.press("tab")
        time.sleep(0.15)
        pyautogui.press("enter")
        time.sleep(0.5)

    if _get_notepad():
        log.warning("Force-closing Notepad")
        try:
            subprocess.run(
                ["taskkill", "/f", "/im", "notepad.exe"],
                capture_output=True, timeout=5,
            )
            time.sleep(0.5)
        except Exception as exc:
            log.debug("taskkill failed: %s", exc)


def wait_before_next(delay: float = 1.0) -> None:
    time.sleep(delay)
