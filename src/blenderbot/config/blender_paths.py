"""Cross-platform discovery for Blender 5.x executables, user extensions, and FFmpeg."""

from __future__ import annotations

import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION_DIR_RE = re.compile(r"^Blender\s+(\d+)\.(\d+)(?:\.(\d+))?$", re.IGNORECASE)
BLENDER_VER_OUT_RE = re.compile(r"Blender\s+(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)


def resolve_home(env: dict[str, str] | None = None) -> Path:
    e = env if env is not None else os.environ
    raw = e.get("BLENDERBOT_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".blenderbot").resolve()


@dataclass(frozen=True)
class BlenderInstallation:
    platform: str
    python_version: str
    python_arch_bits: int
    blender_exe_path: str | None
    blender_version: str | None
    blender_major_minor: str | None
    user_extensions_dir: str
    user_scripts_startup_dir: str
    blender_running: bool
    ffmpeg_path: str | None
    ffprobe_path: str | None

    @property
    def ready(self) -> bool:
        return bool(self.blender_exe_path and Path(self.blender_exe_path).is_file())

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["ready"] = self.ready
        return d


def _is_blender_running(sys_name: str) -> bool:
    try:
        if sys_name == "Windows":
            res = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq blender.exe", "/NH"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            return "blender.exe" in res.stdout.lower()
        res = subprocess.run(
            ["pgrep", "-f", "[bB]lender"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _probe_blender_version(exe: Path) -> tuple[str | None, str | None]:
    try:
        res = subprocess.run(
            [str(exe), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        m = BLENDER_VER_OUT_RE.search(res.stdout)
        if m:
            full = m.group(1)
            parts = full.split(".")
            mm = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else full
            return full, mm
    except (OSError, subprocess.SubprocessError):
        pass

    # Fallback: infer from parent directory name ("Blender 5.3")
    m_dir = VERSION_DIR_RE.match(exe.parent.name)
    if m_dir:
        mm = f"{m_dir.group(1)}.{m_dir.group(2)}"
        full = f"{mm}.{m_dir.group(3) or '0'}"
        return full, mm
    return None, None


def _find_windows_blender_candidates(prog_files: Path) -> list[Path]:
    bf_root = prog_files / "Blender Foundation"
    scored: list[tuple[tuple[int, int, int], Path]] = []
    try:
        if bf_root.is_dir():
            for child in bf_root.iterdir():
                if not child.is_dir():
                    continue
                exe = child / "blender.exe"
                if not exe.is_file():
                    continue
                m = VERSION_DIR_RE.match(child.name)
                ver_key = (
                    (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
                    if m
                    else (0, 0, 0)
                )
                scored.append((ver_key, exe))
    except OSError:
        pass
    scored.sort(key=lambda item: item[0], reverse=True)
    return [exe for _, exe in scored]


def discover_blender(env: dict[str, str] | None = None) -> BlenderInstallation:
    """Locate the newest installed Blender 5.x binary, extension paths, and FFmpeg."""
    e = env if env is not None else dict(os.environ)
    sys_name = platform.system()
    arch_bits = struct.calcsize("P") * 8

    candidates: list[Path] = []
    env_exe = e.get("BLENDERBOT_BLENDER_EXE") or e.get("BLENDER_EXE")
    if env_exe:
        candidates.append(Path(env_exe).expanduser())

    which_blender = shutil.which("blender")
    if which_blender:
        candidates.append(Path(which_blender))

    if sys_name == "Windows":
        prog_files = Path(e.get("PROGRAMFILES", r"C:\Program Files"))
        candidates.extend(_find_windows_blender_candidates(prog_files))
    elif sys_name == "Darwin":
        candidates.extend(
            [
                Path("/Applications/Blender.app/Contents/MacOS/Blender"),
                Path.home() / "Applications/Blender.app/Contents/MacOS/Blender",
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/bin/blender"),
                Path("/snap/bin/blender"),
                Path("/usr/local/bin/blender"),
            ]
        )

    exe_path: Path | None = None
    for cand in candidates:
        try:
            if cand.is_file():
                exe_path = cand.resolve()
                break
        except OSError:
            continue

    full_ver, mm_ver = _probe_blender_version(exe_path) if exe_path else (None, None)
    ver_slot = mm_ver or "5.3"

    if sys_name == "Windows":
        appdata = Path(e.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        bl_user_base = appdata / "Blender Foundation" / "Blender" / ver_slot
    elif sys_name == "Darwin":
        bl_user_base = Path.home() / "Library" / "Application Support" / "Blender" / ver_slot
    else:
        bl_user_base = Path.home() / ".config" / "blender" / ver_slot

    user_ext = bl_user_base / "extensions" / "user_default"
    user_startup = bl_user_base / "scripts" / "startup"

    return BlenderInstallation(
        platform=f"{sys_name} ({platform.machine()})",
        python_version=sys.version.split()[0],
        python_arch_bits=arch_bits,
        blender_exe_path=str(exe_path) if exe_path else None,
        blender_version=full_ver,
        blender_major_minor=mm_ver,
        user_extensions_dir=str(user_ext),
        user_scripts_startup_dir=str(user_startup),
        blender_running=_is_blender_running(sys_name),
        ffmpeg_path=shutil.which("ffmpeg"),
        ffprobe_path=shutil.which("ffprobe"),
    )