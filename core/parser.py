"""Binary front-end for malfamily (Ghidra-backed).

Turns a PE / ELF / Mach-O file into a uniform list of code *functions*, each
carrying the mnemonics Ghidra decoded for it. We drive Ghidra in **headless**
mode: a subprocess runs Ghidra's batch analyzer (``analyzeHeadless``) on the
binary, our small Java export script (``ghidra_scripts/MalfamilyExport.java``)
dumps every function's mnemonics to a temporary JSON file, and we read it back.


Requirements (NOT pip-installable):
  * Ghidra 11+  -> set the ``GHIDRA_INSTALL_DIR`` environment variable.
  * A JDK 21+   -> set ``JAVA_HOME`` (Ghidra needs it; a system default ``java``
    older than 21 will not work).

Public API:
    parse(path) -> ParsedBinary
"""

from __future__ import annotations  # for deferred annotation evaluation

import json
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

_SCRIPT_DIR = Path(__file__).resolve().parent / "ghidra_scripts"
_SCRIPT_NAME = "MalfamilyExport.java"

# FOr windows defender exclusion, since ghidra holds a sample of the data 
# while processing, 
# windows defender will activate if this line is not here
_WORK_DIR = Path(__file__).resolve().parent.parent / "ghidra_work"


class GhidraError(RuntimeError):
    """Raised when the Ghidra headless run fails to produce output,
    but also on timeout and corrupt json."""


@dataclass
class Function:
    """just the function class representing a singular function from the binaries
    ghidra decoded and extracted the mnemonics from"""

    name: str
    # mostly for debugging

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


def build_parsed_binary(data: dict) -> ParsedBinary:
    """Turn the export script's JSON into a :class:`ParsedBinary`.

    Kept separate from the parsing function below so that we can test on ghidra
    exports without needing ghidra 12 or java (Ive had dependencies issues on
    another laptop).
    """
    arch = _arch_of(data["processor"], int(data["ptr_bytes"]))
    functions = [
        Function(
            name=fn["name"],
            va=int(fn["va"]),  # mostly for debugging
            mnemonics=[m.lower() for m in fn["mnemonics"]],
        )
        for fn in data["functions"]
    ]
    return ParsedBinary(format=_format_of(data["format"]), arch=arch, functions=functions)


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


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    """kill headless launcher AND its child JVM on timeout (which would leak if not done)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # type: ignore[attr-defined]
    except Exception:
        proc.kill()


def parse(path: str | Path, analysis_timeout: int = 180) -> ParsedBinary:
    """Parse a PE / ELF / Mach-O binary into a :class:`ParsedBinary`.

    ``analysis_timeout`` bounds Ghidra's auto-analysis per file (seconds).
    this timout time can change depending on the database behavior
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    ghidra = _ghidra_dir()
    headless = (
        ghidra / "support" / ("analyzeHeadless.bat" if os.name == "nt" else "analyzeHeadless")
    )  # I have a windows laptop but my schools pcs run on linux

    # this was a clever way to delete the remaining files I didnt need from ghidra
    # (logs) as they piled up
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="malfamily_ghidra_", dir=_WORK_DIR) as tmp:
        # Import a copy under a fixed, safe name: the original (attacker-controlled)
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
            str(_SCRIPT_DIR),  # run our java script
            "-postScript",
            _SCRIPT_NAME,
            str(out_json),
            "-deleteProject",  # delete the temp ghidra database
            "-analysisTimeoutPerFile",  # to avoid getting stuck
            str(analysis_timeout),
        ]
        # ive found that sometimes running batch commands in a python file can be unreliable
        cmd = ["cmd", "/c", *headless_args] if os.name == "nt" else headless_args

        # Own process group so a timeout can take down the child JVM too, not just the launcher.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,  # so ghidra's .bat "pause" on error gets EOF
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),  # windows only
        )
        # the +300 is a buffer, ghidra burns time booting the jvm before analysis even starts
        try:
            stdout, stderr = proc.communicate(timeout=analysis_timeout + 300)
        except subprocess.TimeoutExpired as e:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()  # drain whatever ghidra logged so far
            tail = ((stdout or "") + (stderr or ""))[-1500:]  # shows where it got stuck
            raise GhidraError(
                f"Ghidra timed out on {path.name} after {analysis_timeout + 300}s\n"
                f"--- last Ghidra output before kill ---\n{tail}"
            ) from e

        if not out_json.exists():
            tail = ((stdout or "") + (stderr or ""))[-2000:]  # to make sure we REALLY get the error
            raise GhidraError(
                f"Ghidra produced no output for {path.name} (exit {proc.returncode}).\n"
                f"--- last lines of Ghidra output ---\n{tail}"
            )
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:  # a crash mid-write leaves truncated JSON
            raise GhidraError(f"Ghidra wrote corrupt JSON for {path.name}: {e}") from e

    return build_parsed_binary(data)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        pb = parse(arg)
        print(f"{Path(arg).name}: {pb.format}/{pb.arch}  functions={pb.num_functions}")
        for fn in pb.functions[:5]:
            print(f"    {fn.name:28s} va=0x{fn.va:x} ninstr={fn.num_instructions}")
