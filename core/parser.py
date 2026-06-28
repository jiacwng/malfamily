"""Ghidra parser wrapper for PE, ELF, and Mach-O files.

Converts a binary into a list of functions and their assembly mnemonics.
We run Ghidra in headless mode (`analyzeHeadless`) as a subprocess. The Java export
script (`ghidra_scripts/MalfamilyExport.java`) dumps the mnemonics to a temporary
JSON file, which this script reads back.

Requirements (NOT pip-installable):
  * Ghidra 11+  -> set the ``GHIDRA_INSTALL_DIR`` environment variable.
  * A JDK 21+   -> set ``JAVA_HOME`` (Ghidra needs it; a system default ``java``
    older than 21 will not work).

Public API:
    parse(path) -> ParsedBinary
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

_SCRIPT_DIR = Path(__file__).resolve().parent / "ghidra_scripts"
_SCRIPT_NAME = "MalfamilyExport.java"

# For Windows Defender exclusion: Ghidra holds a copy of the sample while
# processing, which Defender will flag if this directory isn't excluded.
_WORK_DIR = Path(__file__).resolve().parent.parent / "ghidra_work"


class GhidraError(RuntimeError):
    """Raised when the Ghidra headless run fails to produce output,
    but also on timeout and corrupt json."""


@dataclass
class Function:
    """just the function class representing a singular function from the binaries
    ghidra decoded and extracted the mnemonics from"""

    name: str  # Debugging information

    va: int
    mnemonics: list[str]

    @property
    def num_instructions(self) -> int:
        return len(self.mnemonics)


@dataclass
class ParsedBinary:
    """Format-normalized result of parsing one binary."""

    format: str
    arch: str
    functions: list[Function] = field(default_factory=list)  # mutability issues fix

    @property
    def num_functions(self) -> int:
        return len(self.functions)


# Anti corruption layer
def _arch_of(processor: str, ptr_bytes: int) -> str:
    """('x86', 8) -> 'x86-64'; ('x86', 4) -> 'x86'; ('AARCH64', 8) -> 'arm64'."""
    bits = ptr_bytes * 8
    proc = processor.lower()
    if proc == "x86":
        return "x86-64" if bits == 64 else "x86"
    if proc == "aarch64":
        return "arm64"
    return f"unsupported:{processor}"


def _format_of(fmt: str) -> str:
    """Ghidra's verbose format string -> a short tag."""
    f = fmt.lower()
    if "mach-o" in f:
        return "macho"
    if "elf" in f:
        return "elf"

    if "portable executable" in f or "(pe)" in f:
        return "pe"
    return f


def _ghidra_dir() -> Path:
    """just a function that handles path retrival for ghidra installation"""
    raw = os.environ.get("GHIDRA_INSTALL_DIR")
    if not raw:
        raise GhidraError(
            "GHIDRA_INSTALL_DIR is not set. Point it at your Ghidra install, e.g.\n"
            '  Windows:     setx GHIDRA_INSTALL_DIR "C:\\ghidra_11.3_PUBLIC"\n'
            "  Linux/macOS: export GHIDRA_INSTALL_DIR=/opt/ghidra"
        )
    p = Path(raw)
    if not (p / "support").is_dir():
        raise GhidraError(f"GHIDRA_INSTALL_DIR={p!s} has no 'support/' dir; not a Ghidra install?")
    return p


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the Ghidra subprocess *and* its child JVM.

    On Windows, regular proc.kill() only stops the CMD window, leaving Java running.
    This Java process keeps holding onto a lock file in our temporary folder,
    which crashes Python when it tries to clean up. To fix this, 
    we kill the whole process tree (taskkill /T) to unlock the file.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def parse(path: str | Path, analysis_timeout: int = 180) -> ParsedBinary:
    """Parse a PE / ELF / Mach-O binary into a :class:`ParsedBinary`.

    Time limit for Ghidra to analyze a file (seconds).
    Might need to increase this if the disk/database is slow.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    ghidra = _ghidra_dir()
    headless = (
        ghidra / "support" / ("analyzeHeadless.bat" if os.name == "nt" else "analyzeHeadless")
    )  # Linux and Windows support

    # Temporary Ghidra files handler, avoid keeping unecessary files
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    # ignore_cleanup_errors: If Ghidra times out, it might still lock the project files 
    # for a moment. We ignore cleanup errors so this timing issue doesn't crash the whole 
    # batch. The leftovers will get cleaned up on the next run anyway.
    with TemporaryDirectory(
        prefix="malfamily_ghidra_", dir=_WORK_DIR, ignore_cleanup_errors=True
    ) as tmp:
        # Import a copy under a fixed, safe name: the original
        # filename never reaches the command line, so it can't inject shell metacharacters.
        sample = Path(tmp) / "sample.bin"
        shutil.copyfile(path, sample)

        # as soon as this with block ends, the temp folder is delete after we've
        # extracted everything we need
        out_json = Path(tmp) / "functions.json"
        headless_args = [
            str(headless),
            tmp,
            "mfproj",
            "-import",
            str(sample),  # loading a safe-named copy of the malware binary
            "-scriptPath",
            str(_SCRIPT_DIR),
            "-postScript",
            _SCRIPT_NAME,
            str(out_json),
            "-deleteProject",
            "-analysisTimeoutPerFile",
            str(analysis_timeout),
        ]

        cmd = ["cmd", "/c", *headless_args] if os.name == "nt" else headless_args

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,  # so ghidra's .bat "pause" on error gets EOF
            # Read results only from out_json. Piping stdout/stderr crashed on Windows because 
            # Python tried to decode Ghidra's output using the default cp1252 encoding and failed. 
            # Sending output to DEVNULL completely avoids this decoding error.  
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # the +300 is a buffer: ghidra burns time booting the JVM before analysis starts
        try:
            proc.communicate(timeout=analysis_timeout + 300)
        except subprocess.TimeoutExpired as e:
            _kill_tree(proc)  # kill the child JVM too, or it locks the temp dir
            raise GhidraError(f"Ghidra timed out on {path.name}") from e

        if not out_json.exists():
            raise GhidraError(f"Ghidra produced no output for {path.name} (exit {proc.returncode})")
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise GhidraError(f"Ghidra wrote corrupt JSON for {path.name}: {e}") from e




    arch = _arch_of(data["processor"], int(data["ptr_bytes"]))
    functions = [
        Function(
            name=fn["name"],
            va=int(fn["va"]),
            mnemonics=[m.lower() for m in fn["mnemonics"]],
        )
        for fn in data["functions"]
    ]
    return ParsedBinary(format=_format_of(data["format"]), arch=arch, functions=functions)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        pb = parse(arg)
        print(f"{Path(arg).name}: {pb.format}/{pb.arch}  functions={pb.num_functions}")
        for fn in pb.functions[:5]:
            print(f"    {fn.name:28s} va=0x{fn.va:x} ninstr={fn.num_instructions}")
