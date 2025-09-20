import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_ROOT = PROJECT_ROOT / "upel4" / "src"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from upeel.codegen import COMMENT_SYNTAX, get_backend, iter_backends

CLAUSE = "\u2207\u03d5"

STABLE_BACKENDS = [
    "c",
    "cpp",
    "go",
    "java",
    "js",
    "julia",
    "lua",
    "r",
    "rb",
    "rust",
    "swift",
]


def test_backend_registry_matches_comment_syntax():
    registry_names = {descriptor.name for descriptor in iter_backends()}
    assert registry_names == set(COMMENT_SYNTAX.keys())


@pytest.mark.parametrize("backend_name", STABLE_BACKENDS)
def test_stable_backends_render_and_requirements(backend_name: str):
    descriptor = get_backend(backend_name)
    snippet = descriptor.render_invocation("grad", ["phi"], CLAUSE)

    assert descriptor.maturity == "stable"
    assert COMMENT_SYNTAX[backend_name] in snippet
    assert "grad(phi)" in snippet
    assert CLAUSE in snippet
    assert descriptor.compile_cmd or descriptor.run_cmd

    binaries = descriptor.required_binaries()
    assert all("{" not in binary and "}" not in binary for binary in binaries)

    if descriptor.statement_terminator:
        assert snippet.strip().endswith(descriptor.statement_terminator)

def test_backends_provide_basic_metadata():
    descriptors = list(iter_backends())
    assert descriptors, "expected at least one backend descriptor"

    for descriptor in descriptors:
        assert descriptor.name
        assert descriptor.language
        assert descriptor.file_extension is not None
        assert descriptor.invocation_kind