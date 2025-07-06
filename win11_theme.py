# ──────────────────────────────────────────────────────────────────────────
# Complete Windows 11–style theme helper for PySide6 / PyQt6
#
# Usage:
#     from win11_theme import apply_win11_theme
#     ...
#     apply_win11_theme(self, palette="auto", acrylic=True)
#
# Requirements: Windows 11 (build ≥ 22000), PySide6 ≥ 6.4
# --------------------------------------------------------------------------

from __future__ import annotations
import ctypes
import platform
import sys

from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


# ──────────────────────────────────────────────────────────────────────────
# ❶  Complete QPalette definitions (light & dark)
# --------------------------------------------------------------------------
_LIGHT = {
    "WINDOW":        "#ffffff",
    "WINDOW_TEXT":   "#000000",
    "BASE":          "#f6f6f6",
    "TEXT":          "#000000",
    "BUTTON":        "#f0f0f0",
    "BUTTON_TEXT":   "#000000",
    "HIGHLIGHT":     "#0078d4",
    "HIGHLIGHT_TEXT":"#ffffff",
    "BORDER":        "#d0d0d0"
}
_DARK = {
    "WINDOW":        "#202020",
    "WINDOW_TEXT":   "#ffffff",
    "BASE":          "#2a2a2a",
    "TEXT":          "#ffffff",
    "BUTTON":        "#2d2d2d",
    "BUTTON_TEXT":   "#ffffff",
    "HIGHLIGHT":     "#3b78ff",
    "HIGHLIGHT_TEXT":"#ffffff",
    "BORDER":        "#3a3a3a"
}


def _make_palette(colors: dict[str, str]) -> QPalette:
    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor(colors["WINDOW"]))
    pal.setColor(QPalette.WindowText,    QColor(colors["WINDOW_TEXT"]))
    pal.setColor(QPalette.Base,          QColor(colors["BASE"]))
    pal.setColor(QPalette.Text,          QColor(colors["TEXT"]))
    pal.setColor(QPalette.Button,        QColor(colors["BUTTON"]))
    pal.setColor(QPalette.ButtonText,    QColor(colors["BUTTON_TEXT"]))
    pal.setColor(QPalette.Highlight,     QColor(colors["HIGHLIGHT"]))
    pal.setColor(QPalette.HighlightedText, QColor(colors["HIGHLIGHT_TEXT"]))
    return pal


# ──────────────────────────────────────────────────────────────────────────
# ❷  Win32 Acrylic / Mica helpers
# --------------------------------------------------------------------------
class ACCENTPOLICY(ctypes.Structure):
    _fields_ = [("AccentState", ctypes.c_int),
                ("AccentFlags", ctypes.c_int),
                ("GradientColor", ctypes.c_uint32),
                ("AnimationId", ctypes.c_int)]

class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
    _fields_ = [("Attrib", ctypes.c_int),
                ("pvData", ctypes.c_void_p),
                ("cbData", ctypes.c_size_t)]

# constants
ACCENT_ENABLE_ACRYLIC_BLUR = 6     # Acrylic backdrop (build 22000+)
ACCENT_ENABLE_BLURBEHIND   = 3     # fallback for older builds
WCA_ACCENT_POLICY          = 19

_user32 = ctypes.windll.user32
_SetWindowCompositionAttribute = _user32.SetWindowCompositionAttribute
_SetWindowCompositionAttribute.restype = ctypes.c_int


# ──────────────────────────────────────────────────────────────────────────
# ❸  Public function
# --------------------------------------------------------------------------
def apply_win11_theme(
        widget: QWidget,
        *,
        palette: str = "auto",     # "light" | "dark" | "auto"
        acrylic: bool = True,
) -> None:
    """
    Apply Windows-11-like palette, rounded-corner stylesheet and optional
    Acrylic/Mica blur behind *widget* (usually the QMainWindow).
    """

    # Determine light/dark choice
    if palette == "auto":
        is_dark = _system_prefers_dark()
    else:
        is_dark = (palette.lower() == "dark")

    # ---- 1. Palette ------------------------------------------------------
    pal = _make_palette(_DARK if is_dark else _LIGHT)
    widget.setPalette(pal)

    # ---- 2. Global stylesheet (for the whole app) -----------------------
    # Rounded corners + neutral spacing/fonts everywhere
    STYLE = f"""
        /* make the entire window see‐through */
        * {{
            background: transparent;
            border: none;
            border-radius: 7px;
            padding: 6px;
            font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
            font-size: 16px;
        }}
        QLineEdit {{
            /* a slightly opaque input box */
            background-color: rgba(255,255,255,0.03);
            padding: 6px 10px;
        }}
        QListWidget, QListView {{
            background: transparent;
        }}
        QPushButton {{
            background-color: rgba(255,255,255,0.04);
        }}
        QPushButton:hover {{
            background-color: rgba(255,255,255,0.08);
        }}
    """
    widget.setStyleSheet(STYLE)

    # ---- 3. Acrylic / Mica blur behind the window -----------------------
    if acrylic and platform.system() == "Windows":
        hwnd = int(widget.winId())        # PySide returns WId → int
        try:
            from BlurWindow.blurWindow import blur
            # apply blur to your main window:
            blur(hwnd)
            print("[DEBUG] BlurWindow: acrylic blur applied")
        except ImportError:
            print("[DEBUG] BlurWindow not installed—falling back to solid background")


# ──────────────────────────────────────────────────────────────────────────
# ❹  Detect Windows dark / light preference
# --------------------------------------------------------------------------
def _system_prefers_dark() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                 r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
        return v == 0
    except OSError:
        return False
