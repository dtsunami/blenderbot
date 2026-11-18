"""Cross-platform clipboard copy, browser launch, and Windows conhost QuickEdit hardening."""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import shutil
import subprocess
import sys
import webbrowser


def prepare_windows_console() -> None:
    """Disable Win32 QuickEdit Mode in legacy `conhost` so accidental console clicks cannot freeze stdout."""
    if platform.system() != "Windows":
        return
    if os.environ.get("WT_SESSION") or not sys.stdout.isatty() or not sys.stdin.isatty():
        return
    with contextlib.suppress(Exception):
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleOutputCP(65001)
        std_input_handle = kernel32.GetStdHandle(-10)
        mode = ctypes.c_uint32(0)
        if kernel32.GetConsoleMode(std_input_handle, ctypes.byref(mode)):
            # Clear ENABLE_QUICK_EDIT_MODE (0x0040), set ENABLE_EXTENDED_FLAGS (0x0080)
            new_mode = (mode.value & ~0x0040) | 0x0080
            kernel32.SetConsoleMode(std_input_handle, new_mode)


def copy_to_clipboard(text: str) -> bool:
    """Copy `text` to the OS clipboard without third-party dependencies."""
    sys_name = platform.system()
    try:
        if sys_name == "Windows":
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
                input=text,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=3,
                check=False,
            )
            return res.returncode == 0
        if sys_name == "Darwin" and shutil.which("pbcopy"):
            res = subprocess.run(
                ["pbcopy"], input=text, text=True, encoding="utf-8", timeout=3, check=False
            )
            return res.returncode == 0
        for linux_cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"]):
            if shutil.which(linux_cmd[0]):
                res = subprocess.run(
                    linux_cmd, input=text, text=True, encoding="utf-8", timeout=3, check=False
                )
                return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    return False


def open_in_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url, new=2))
    except Exception:  # noqa: BLE001
        return False