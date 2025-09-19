import math
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_ROOT = PROJECT_ROOT / "upel4" / "src"
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import upeel  # type: ignore  # noqa: E402
from upeel import symbolic_operators as ops  # type: ignore  # noqa: E402


def test_mean_operator():
    assert upeel.mean([1, 2, 3]) == pytest.approx(2.0)


def test_gradient_from_dict():
    gradient = upeel.grad({"values": (2.0, 4.0, 6.0), "spacing": (2.0, 2.0, 2.0)})
    assert gradient == [1.0, 2.0, 3.0]


def test_scalar_field_clause_and_eval():
    params = {"phi0": 2.0, "k": 0.5, "x": 1.0}
    assert upeel.scalar_field(params) == pytest.approx(3.0)
    assert ops.REGISTRY["scalar_field"].clause == "ϕ(x) = ϕ₀(1 + kx)"


def test_uncertainty_relation():
    params = {"dx": 1.0, "dp": 1.0, "hbar": 1.0}
    assert upeel.uncertainty_relation(params) is True


def test_lowering_includes_clause_comment():
    lowered = ops.REGISTRY["grad"].lower_cpp("phi")
    assert "grad(phi)" in lowered
    assert "∇ϕ" in lowered


def test_all_registered_ops_have_clauses():
    assert all(op.clause for op in ops.REGISTRY.values())


def test_plasma_frequency_positive():
    params = {"charge_density": 4.0, "mass": 2.0, "permittivity": 1.0}
    assert upeel.plasma_frequency(params) == pytest.approx(math.sqrt(2.0))
