"""Helper utilities for backend code generation."""
from __future__ import annotations

from typing import Final

# Mapping from backend identifier to the token used to begin a comment.  For
# block comments we produce the balanced closing token inside :func:`comment`.
COMMENT_SYNTAX: Final[dict[str, str]] = {
    "nim": "#",
    "d": "//",
    "hs": "--",
    "ocaml": "(*",
    "crystal": "#",
    "ts": "//",
    "dart": "//",
    "rb": "#",
    "lua": "--",
    "r": "#",
    "c": "/*",
    "cpp": "//",
    "rust": "//",
    "zig": "//",
    "go": "//",
    "swift": "//",
    "java": "//",
    "kotlin": "//",
    "cs": "//",
    "ada": "--",
    "julia": "#",
    "m": "%",
    "slx": "%",
    "js": "//",
    "wasm": ";;",
    "cuda": "//",
    "opencl": "//",
    "omp_f": "!",
    "vitis": "//",
    "vhdl": "--",
    "verilog": "//",
    "sv": "//",
    "st": "(*",
    "chisel": "//",
    "bsv": "//",
    "spinal": "//",
    "intel_hls": "//",
    "sycl": "//",
    "metal": "//",
    "misrac": "//",
    "mikroc": "//",
    "armasm": "@",
    "isabelle": "(*",
    "coq": "(*",
    "agda": "--",
    "chapel": "//",
    "futhark": "--",
    "fortran": "!",
    "lean": "--",
    "spirv": ";",
    "bqn": "#",
}


def comment(backend: str, text: str) -> str:
    """Render ``text`` as a backend-specific comment string.

    Parameters
    ----------
    backend:
        Identifier of the backend/language target.
    text:
        Comment body.
    """

    token = COMMENT_SYNTAX.get(backend, "//")
    if token == "/*":
        return f"/* {text} */"
    if token == "(*":
        return f"(* {text} *)"
    return f"{token} {text}"


__all__ = ["COMMENT_SYNTAX", "comment"]
