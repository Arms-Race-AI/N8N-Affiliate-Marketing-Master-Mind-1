# -*- coding: utf-8 -*-
"""U P E L⁴ – Symbolic Operator Library (Tier‑7).

This module provides the authoritative registry for symbolic physics operators
and mathematical utilities.  Each operator is paired with a *clause* — a short
statement of the governing law or invariant — so downstream components can
trace generated code back to its physical provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import cmath
import math
import statistics

from .codegen.backends import BackendDescriptor, get_backend, iter_backends


@dataclass()
class Op:
    name: str
    arity: int
    arg_types: Sequence[str]
    ret_type: str
    eval_py: Callable[..., Any]
    clause: str
    description: str


REGISTRY: Dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.name in REGISTRY:
        raise ValueError(f"Operator '{op.name}' already registered")
    REGISTRY[op.name] = op
    return op


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _is_matrix(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and value
        and all(isinstance(row, Sequence) for row in value)
    )


def _matrix_transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def _matrix_multiply(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    if not _is_matrix(a) or not _is_matrix(b):
        raise TypeError("matrix multiplication requires two matrices")
    if len(a[0]) != len(b):
        raise ValueError("incompatible shapes for matrix multiplication")
    result: list[list[float]] = []
    for row in a:
        result_row = []
        for col in zip(*b):
            result_row.append(sum(float(x) * float(y) for x, y in zip(row, col)))
        result.append(result_row)
    return result


def _matrix_minor(matrix: Sequence[Sequence[float]], i: int, j: int) -> list[list[float]]:
    return [
        [value for cj, value in enumerate(row) if cj != j]
        for ri, row in enumerate(matrix)
        if ri != i
    ]


def _determinant(matrix: Sequence[Sequence[float]]) -> float:
    if not _is_matrix(matrix):
        raise TypeError("determinant requires a matrix input")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square to compute determinant")
    if n == 1:
        return float(matrix[0][0])
    if n == 2:
        return float(matrix[0][0]) * float(matrix[1][1]) - float(matrix[0][1]) * float(matrix[1][0])
    det = 0.0
    for j, value in enumerate(matrix[0]):
        det += ((-1) ** j) * float(value) * _determinant(_matrix_minor(matrix, 0, j))
    return det


def _matrix_inverse(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not _is_matrix(matrix):
        raise TypeError("inverse requires a matrix input")
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square to compute inverse")
    det = _determinant(matrix)
    if math.isclose(det, 0.0):
        raise ValueError("matrix is singular")
    if n == 1:
        return [[1.0 / float(matrix[0][0])]]
    cofactors = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor = _matrix_minor(matrix, i, j)
            cofactor_row.append(((-1) ** (i + j)) * _determinant(minor))
        cofactors.append(cofactor_row)
    adjugate = _matrix_transpose(cofactors)
    return [[value / det for value in row] for row in adjugate]


def _scale(value: Any, factor: float) -> Any:
    if _is_matrix(value):
        return [[factor * float(entry) for entry in row] for row in value]  # type: ignore[arg-type]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [factor * float(entry) for entry in value]  # type: ignore[arg-type]
    return factor * float(value)


def _matrix_subtract(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    if not (_is_matrix(a) and _is_matrix(b)):
        raise TypeError("matrix subtraction requires two matrices")
    if len(a) != len(b) or any(len(row_a) != len(row_b) for row_a, row_b in zip(a, b)):
        raise ValueError("matrices must have identical shapes for subtraction")
    return [
        [float(x) - float(y) for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)
    ]


def _frobenius_norm(value: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(float(entry) ** 2 for row in value for entry in row))


def _quat_mul(q1: Sequence[float], q2: Sequence[float]) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _vector_from_mapping(mapping: Mapping[str, Iterable[float]], key: str, default: Sequence[float]) -> list[float]:
    values = mapping.get(key)
    if values is None:
        return list(default)
    return [float(v) for v in values]


def _get_spacing(mapping: Mapping[str, Iterable[float]], default_length: int) -> list[float]:
    spacing = mapping.get("spacing")
    if spacing is None:
        return [1.0] * default_length
    spacing_list = [float(v) for v in spacing]
    if len(spacing_list) != default_length:
        raise ValueError("spacing length does not match vector dimensions")
    return spacing_list


def _gradient(field: Any) -> list[float]:
    if hasattr(field, "grad") and callable(field.grad):  # type: ignore[attr-defined]
        return list(field.grad())  # type: ignore[call-arg]
    if isinstance(field, Mapping):
        if "gradient" in field:
            return [float(v) for v in field["gradient"]]
        values = _vector_from_mapping(field, "values", (0.0, 0.0, 0.0))
        spacing = _get_spacing(field, len(values))
        return [val / step for val, step in zip(values, spacing)]
    if isinstance(field, Sequence) and not isinstance(field, (str, bytes)):
        return [float(v) for v in field]
    raise TypeError("unsupported field representation for gradient")


def _divergence(vec_field: Any) -> float:
    if hasattr(vec_field, "div") and callable(vec_field.div):  # type: ignore[attr-defined]
        return float(vec_field.div())  # type: ignore[call-arg]
    if isinstance(vec_field, Mapping):
        if "divergence" in vec_field:
            return float(vec_field["divergence"])
        components = _vector_from_mapping(vec_field, "components", (0.0, 0.0, 0.0))
        spacing = _get_spacing(vec_field, len(components))
        return sum(comp / step for comp, step in zip(components, spacing))
    if isinstance(vec_field, Sequence) and not isinstance(vec_field, (str, bytes)):
        return sum(float(v) for v in vec_field)
    raise TypeError("unsupported vector field representation for divergence")


def _curl(vec_field: Any) -> list[float]:
    if hasattr(vec_field, "curl") and callable(vec_field.curl):  # type: ignore[attr-defined]
        return list(vec_field.curl())  # type: ignore[call-arg]
    if isinstance(vec_field, Mapping):
        if "curl" in vec_field:
            return [float(v) for v in vec_field["curl"]]
        partials = {
            "dAz_dy": 0.0,
            "dAy_dz": 0.0,
            "dAx_dz": 0.0,
            "dAz_dx": 0.0,
            "dAy_dx": 0.0,
            "dAx_dy": 0.0,
        }
        partials.update({k: float(v) for k, v in vec_field.items() if k in partials})
        return [
            partials["dAz_dy"] - partials["dAy_dz"],
            partials["dAx_dz"] - partials["dAz_dx"],
            partials["dAy_dx"] - partials["dAx_dy"],
        ]
    raise TypeError("unsupported vector field representation for curl")


def _laplacian(field: Any) -> float:
    if hasattr(field, "laplacian") and callable(field.laplacian):  # type: ignore[attr-defined]
        return float(field.laplacian())  # type: ignore[call-arg]
    if isinstance(field, Mapping):
        if "laplacian" in field:
            return float(field["laplacian"])
        second_derivatives = _vector_from_mapping(field, "second_derivatives", (0.0, 0.0, 0.0))
        return sum(second_derivatives)
    if isinstance(field, Sequence) and not isinstance(field, (str, bytes)):
        return sum(float(v) for v in field)
    raise TypeError("unsupported field representation for laplacian")


def _dalembertian(params: Mapping[str, Any]) -> float:
    d2phi_dt2 = float(params.get("d2phi_dt2", 0.0))
    laplacian_phi = float(params.get("laplacian_phi", 0.0))
    c_sq = float(params.get("c_sq", 1.0))
    return d2phi_dt2 - c_sq * laplacian_phi


def _covariant_derivative(params: Mapping[str, Any]) -> list[float]:
    field = _vector_from_mapping(params, "field", (0.0, 0.0, 0.0))
    connection = params.get("connection", ((0.0, 0.0, 0.0),) * len(field))
    if not _is_matrix(connection):
        raise TypeError("connection must be a matrix")
    if len(connection) != len(field):
        raise ValueError("connection and field dimensions must match")
    return [
        float(component) - sum(float(gamma) for gamma in row)
        for component, row in zip(field, connection)
    ]


def _stress_energy_tensor(params: Mapping[str, Any]) -> list[list[float]]:
    energy = float(params.get("energy", 0.0))
    momentum = _vector_from_mapping(params, "momentum", (0.0, 0.0, 0.0))
    density = float(params.get("density", 0.0))
    return [
        [energy, momentum[0], momentum[1], momentum[2]],
        [momentum[0], density, 0.0, 0.0],
        [momentum[1], 0.0, density, 0.0],
        [momentum[2], 0.0, 0.0, density],
    ]


def _metric_tensor(params: Mapping[str, Any]) -> list[list[float]]:
    metric = params.get(
        "metric",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
        ),
    )
    return [list(map(float, row)) for row in metric]


def _kronecker_delta(i: int, j: int) -> int:
    return 1 if i == j else 0


def _levi_civita(i: int, j: int, k: int) -> int:
    indices = (i, j, k)
    if len(set(indices)) < 3:
        return 0
    if indices in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        return 1
    return -1


def _fourier_transform(symbol: str = "phi") -> str:
    return f"Fourier[{symbol}](omega)"


def _inverse_fourier_transform(symbol: str = "phi") -> str:
    return f"Fourier^-1[{symbol}](t)"


def _hamiltonian(params: Mapping[str, Any]) -> float:
    return float(params.get("kinetic", 0.0)) + float(params.get("potential", 0.0))


def _lagrangian(params: Mapping[str, Any]) -> float:
    return float(params.get("kinetic", 0.0)) - float(params.get("potential", 0.0))


def _action_integral(params: Mapping[str, Any]) -> float:
    lagrangian = float(params.get("lagrangian", 0.0))
    t0 = float(params.get("t0", 0.0))
    t1 = float(params.get("t1", 1.0))
    return lagrangian * (t1 - t0)


def _commutator(params: Mapping[str, Any]) -> Any:
    A = params.get("A", 0.0)
    B = params.get("B", 0.0)
    if _is_matrix(A) and _is_matrix(B):
        return _matrix_subtract(_matrix_multiply(A, B), _matrix_multiply(B, A))
    return A * B - B * A  # type: ignore[operator]


def _poisson_bracket(params: Mapping[str, Any]) -> float:
    dF_dx = float(params.get("dF_dx", 0.0))
    dG_dp = float(params.get("dG_dp", 0.0))
    dF_dp = float(params.get("dF_dp", 0.0))
    dG_dx = float(params.get("dG_dx", 0.0))
    return dF_dx * dG_dp - dF_dp * dG_dx


def _time_derivative(value: float, dt: float) -> float:
    if math.isclose(dt, 0.0):
        raise ValueError("dt must be non-zero")
    return float(value) / float(dt)


def _space_derivative(value: float, dx: float) -> float:
    if math.isclose(dx, 0.0):
        raise ValueError("dx must be non-zero")
    return float(value) / float(dx)


def _euler_lagrange(params: Mapping[str, Any]) -> float:
    dL_dq = float(params.get("dL_dq", 0.0))
    d_dt_dL_dqdot = float(params.get("d_dt_dL_dqdot", 0.0))
    return d_dt_dL_dqdot - dL_dq


def _boltzmann_entropy(params: Mapping[str, Any]) -> float:
    k_B = float(params.get("k_B", 1.380649e-23))
    omega = float(params.get("omega", 1.0))
    if omega <= 0:
        raise ValueError("omega must be positive for entropy calculation")
    return k_B * math.log(omega)


def _partition_function(params: Mapping[str, Any]) -> float:
    energies = [float(e) for e in params.get("energies", (0.0,))]
    beta = float(params.get("beta", 1.0))
    return sum(math.exp(-beta * e) for e in energies)


def _expectation_value(params: Mapping[str, Any]) -> float:
    observables = [float(o) for o in params.get("observables", (0.0,))]
    probabilities = [float(p) for p in params.get("probabilities", (1.0,))]
    if len(probabilities) != len(observables):
        raise ValueError("observables and probabilities must have the same length")
    return sum(o * p for o, p in zip(observables, probabilities))


def _quantum_operator(params: Mapping[str, Any]) -> Any:
    operator = params.get("operator", lambda x: x)
    state = params.get("state", 1.0)
    return operator(state)


def _schrodinger(params: Mapping[str, Any]) -> complex:
    H = params.get("H", 0.0)
    psi = params.get("psi", 0.0)
    dt = float(params.get("dt", 1.0))
    hbar = float(params.get("hbar", 1.0))
    return (H * psi * dt) / (1j * hbar)  # type: ignore[operator]


def _heisenberg(params: Mapping[str, Any]) -> complex:
    commutator_value = params.get("commutator", 0.0)
    hbar = float(params.get("hbar", 1.0))
    return (1j / hbar) * commutator_value  # type: ignore[operator]


def _uncertainty(params: Mapping[str, Any]) -> bool:
    dx = float(params.get("dx", 1.0))
    dp = float(params.get("dp", 1.0))
    hbar = float(params.get("hbar", 1.0))
    return dx * dp >= hbar / 2.0


def _path_integral(params: Mapping[str, Any]) -> complex:
    action = float(params.get("action", 0.0))
    hbar = float(params.get("hbar", 1.0))
    return cmath.exp(1j * action / hbar)


def _green_function() -> str:
    return "G(x, t; x', t')"


def _propagator(params: Mapping[str, Any]) -> Any:
    evolution = params.get("evolution", lambda state: state)
    state = params.get("state", 1.0)
    return evolution(state)


def _operator_norm(value: Any) -> float:
    if _is_matrix(value):
        return _frobenius_norm(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return math.sqrt(sum(float(v) ** 2 for v in value))
    return abs(float(value))


def _spectral_decomposition(params: Mapping[str, Any]) -> list[Any]:
    eigvals = [float(v) for v in params.get("eigenvalues", (1.0,))]
    projectors = params.get("projectors", [1.0] * len(eigvals))
    if len(projectors) != len(eigvals):
        raise ValueError("eigenvalues and projectors must have the same length")
    return [_scale(projector, lam) for lam, projector in zip(eigvals, projectors)]


def _lie_derivative(params: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "vector_field": params.get("vector_field", "X"),
        "tensor": params.get("tensor", "T"),
        "result": "Lie_X(T)",
    }


def _em_field_tensor(params: Mapping[str, Any]) -> list[list[float]]:
    E = _vector_from_mapping(params, "E", (0.0, 0.0, 0.0))
    B = _vector_from_mapping(params, "B", (0.0, 0.0, 0.0))
    return [
        [0.0, -E[0], -E[1], -E[2]],
        [E[0], 0.0, -B[2], B[1]],
        [E[1], B[2], 0.0, -B[0]],
        [E[2], -B[1], B[0], 0.0],
    ]


def _maxwell_clause(params: Mapping[str, Any]) -> Mapping[str, Any]:
    electric_flux = float(params.get("electric_flux", 0.0))
    magnetic_flux = float(params.get("magnetic_flux", 0.0))
    emf = float(params.get("emf", 0.0))
    displacement_current = float(params.get("displacement_current", 0.0))
    return {
        "gauss_electric": electric_flux,
        "gauss_magnetic": magnetic_flux,
        "faraday_law": -emf,
        "ampere_maxwell": displacement_current,
    }


def _scalar_field(params: Mapping[str, Any]) -> float:
    phi0 = float(params.get("phi0", 1.0))
    k = float(params.get("k", 0.0))
    displacement = float(params.get("x", params.get("displacement", 0.0)))
    return phi0 * (1.0 + k * displacement)


def _tensor_invariant(params: Mapping[str, Any]) -> float:
    tensor = params.get("tensor", ((1.0, 0.0), (0.0, 1.0)))
    return _determinant(tensor)


def _time_evolution(params: Mapping[str, Any]) -> float:
    rate = float(params.get("rate", 0.0))
    time = float(params.get("time", 0.0))
    return math.exp(rate * time)


def _time_entropy(params: Mapping[str, Any]) -> float:
    entropy_rate = float(params.get("entropy_rate", 0.0))
    time = float(params.get("time", 0.0))
    return entropy_rate * time


def _time_uncertainty(params: Mapping[str, Any]) -> float:
    delta_energy = float(params.get("delta_energy", 0.0))
    hbar = float(params.get("hbar", 1.0))
    if math.isclose(delta_energy, 0.0):
        raise ValueError("delta_energy must be non-zero")
    return hbar / (2.0 * delta_energy)


def _spacetime_interval(params: Mapping[str, Any]) -> float:
    c = float(params.get("c", 1.0))
    dt = float(params.get("dt", 0.0))
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    dz = float(params.get("dz", 0.0))
    return (c * dt) ** 2 - (dx ** 2 + dy ** 2 + dz ** 2)


def _gravitational_time_dilation(params: Mapping[str, Any]) -> float:
    g_potential = float(params.get("gravitational_potential", 0.0))
    c = float(params.get("c", 1.0))
    return math.sqrt(1.0 + 2.0 * g_potential / (c ** 2))


def _relativistic_time_dilation(params: Mapping[str, Any]) -> float:
    velocity = float(params.get("velocity", 0.0))
    c = float(params.get("c", 1.0))
    beta_sq = (velocity / c) ** 2
    if beta_sq >= 1.0:
        raise ValueError("velocity must be less than the speed of light")
    return 1.0 / math.sqrt(1.0 - beta_sq)


def _quantum_tunneling_time(params: Mapping[str, Any]) -> float:
    barrier_width = float(params.get("barrier_width", 1.0))
    barrier_height = float(params.get("barrier_height", 1.0))
    particle_energy = float(params.get("particle_energy", 0.5))
    return barrier_width * math.sqrt(max(barrier_height - particle_energy, 0.0))


def _wormhole(params: Mapping[str, Any]) -> Mapping[str, Any]:
    throat_radius = float(params.get("throat_radius", 1.0))
    redshift = float(params.get("redshift", 0.0))
    return {
        "throat_radius": throat_radius,
        "redshift": redshift,
        "metric": "Morris-Thorne",
    }


def _plasma_frequency(params: Mapping[str, Any]) -> float:
    charge_density = float(params.get("charge_density", 1.0))
    mass = float(params.get("mass", 1.0))
    permittivity = float(params.get("permittivity", 1.0))
    return math.sqrt(charge_density / (mass * permittivity))


def _fractal_dimension(params: Mapping[str, Any]) -> float:
    scaling_factor = float(params.get("scaling_factor", 2.0))
    copies = float(params.get("copies", 2.0))
    if scaling_factor <= 0:
        raise ValueError("scaling_factor must be positive")
    return math.log(copies, scaling_factor)


def _cosmological_expansion(params: Mapping[str, Any]) -> float:
    hubble_parameter = float(params.get("hubble_parameter", 70.0))
    scale_factor = float(params.get("scale_factor", 1.0))
    return hubble_parameter * scale_factor


def _memory_decay(params: Mapping[str, Any]) -> float:
    initial_state = float(params.get("initial_state", 1.0))
    decay_constant = float(params.get("decay_constant", 1.0))
    time = float(params.get("time", 0.0))
    return initial_state * math.exp(-decay_constant * time)


def _recursive_feedback(params: Mapping[str, Any]) -> float:
    value = float(params.get("value", 0.0))
    feedback = float(params.get("feedback", 0.0))
    return value / (1.0 - feedback) if feedback < 1.0 else math.inf


def _observer_effect(params: Mapping[str, Any]) -> Mapping[str, Any]:
    measurement = params.get("measurement", "observable")
    disturbance = params.get("disturbance", "delta")
    return {"measurement": measurement, "disturbance": disturbance}


def _consciousness_operator(params: Mapping[str, Any]) -> Mapping[str, Any]:
    awareness = float(params.get("awareness", 1.0))
    integration = float(params.get("integration", 1.0))
    return {"phi": awareness * integration}


def _adaptive_feedback(params: Mapping[str, Any]) -> float:
    signal = float(params.get("signal", 0.0))
    response = float(params.get("response", 0.0))
    adaptation = float(params.get("adaptation", 1.0))
    return adaptation * (signal - response)


def _time_perception(params: Mapping[str, Any]) -> float:
    stimulus = float(params.get("stimulus", 1.0))
    attention = float(params.get("attention", 1.0))
    return stimulus / max(attention, 1e-12)


def _available_backends() -> tuple[str, ...]:
    return tuple(desc.name for desc in iter_backends())


BACKENDS = _available_backends()


# ---------------------------------------------------------------------------
# Operator registration
# ---------------------------------------------------------------------------

register(
    Op(
        name="grad",
        arity=1,
        arg_types=["field3"],
        ret_type="vec3",
        eval_py=_gradient,
        clause="∇ϕ",
        description="Spatial gradient of a scalar field",
    )
)

register(
    Op(
        name="div",
        arity=1,
        arg_types=["vecfield3"],
        ret_type="scalar",
        eval_py=_divergence,
        clause="∇·A",
        description="Divergence measuring net flux from a vector field",
    )
)

register(
    Op(
        name="curl",
        arity=1,
        arg_types=["vecfield3"],
        ret_type="vec3",
        eval_py=_curl,
        clause="∇×A",
        description="Curl capturing rotation of a vector field",
    )
)

register(
    Op(
        name="laplacian",
        arity=1,
        arg_types=["field3"],
        ret_type="scalar",
        eval_py=_laplacian,
        clause="∇²ϕ",
        description="Laplacian diffusion operator",
    )
)

register(
    Op(
        name="ddt",
        arity=2,
        arg_types=["scalar", "dt"],
        ret_type="scalar",
        eval_py=_time_derivative,
        clause="d/dt",
        description="Time derivative of an observable",
    )
)

register(
    Op(
        name="ddx",
        arity=2,
        arg_types=["scalar", "dx"],
        ret_type="scalar",
        eval_py=_space_derivative,
        clause="d/dx",
        description="Spatial derivative along x",
    )
)

register(
    Op(
        name="mat_mul",
        arity=2,
        arg_types=["mat", "mat"],
        ret_type="mat",
        eval_py=_matrix_multiply,
        clause="(AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ",
        description="Matrix multiplication",
    )
)

register(
    Op(
        name="transpose",
        arity=1,
        arg_types=["mat"],
        ret_type="mat",
        eval_py=_matrix_transpose,
        clause="(Aᵀ)ᵢⱼ = Aⱼᵢ",
        description="Matrix transpose",
    )
)

register(
    Op(
        name="det",
        arity=1,
        arg_types=["mat"],
        ret_type="scalar",
        eval_py=_determinant,
        clause="det(A)",
        description="Determinant giving volume scaling",
    )
)

register(
    Op(
        name="inv",
        arity=1,
        arg_types=["mat"],
        ret_type="mat",
        eval_py=_matrix_inverse,
        clause="A⁻¹",
        description="Matrix inverse",
    )
)

register(
    Op(
        name="eig",
        arity=1,
        arg_types=["mat"],
        ret_type="pair",
        eval_py=lambda matrix: (_determinant(matrix), matrix),
        clause="det(A - λI) = 0",
        description="Eigenstructure placeholder returning determinant and matrix",
    )
)

register(
    Op(
        name="complex",
        arity=2,
        arg_types=["scalar", "scalar"],
        ret_type="complex",
        eval_py=lambda re, im: complex(re, im),
        clause="z = x + iy",
        description="Construct a complex number",
    )
)

register(
    Op(
        name="conj",
        arity=1,
        arg_types=["complex"],
        ret_type="complex",
        eval_py=lambda z: z.conjugate(),
        clause="z*",
        description="Complex conjugation",
    )
)

register(
    Op(
        name="abs",
        arity=1,
        arg_types=["complex"],
        ret_type="scalar",
        eval_py=abs,
        clause="|z|",
        description="Complex magnitude",
    )
)

register(
    Op(
        name="arg",
        arity=1,
        arg_types=["complex"],
        ret_type="scalar",
        eval_py=cmath.phase,
        clause="arg(z)",
        description="Complex argument",
    )
)

register(
    Op(
        name="quat_mul",
        arity=2,
        arg_types=["quat", "quat"],
        ret_type="quat",
        eval_py=_quat_mul,
        clause="q = q₁ ⊗ q₂",
        description="Quaternion product",
    )
)

register(
    Op(
        name="quat_norm",
        arity=1,
        arg_types=["quat"],
        ret_type="scalar",
        eval_py=lambda q: math.sqrt(sum(component * component for component in q)),
        clause="||q||",
        description="Quaternion norm",
    )
)

register(
    Op(
        name="mean",
        arity=1,
        arg_types=["vec"],
        ret_type="scalar",
        eval_py=statistics.fmean,
        clause="⟨x⟩",
        description="Arithmetic mean",
    )
)

register(
    Op(
        name="variance",
        arity=1,
        arg_types=["vec"],
        ret_type="scalar",
        eval_py=lambda data: statistics.pvariance(data),
        clause="Var(x)",
        description="Population variance",
    )
)

register(
    Op(
        name="stddev",
        arity=1,
        arg_types=["vec"],
        ret_type="scalar",
        eval_py=lambda data: statistics.pstdev(data),
        clause="σ",
        description="Population standard deviation",
    )
)


def _gaussian(x: float, mu: float, sigma: float) -> float:
    if math.isclose(sigma, 0.0):
        raise ValueError("sigma must be non-zero")
    coeff = 1.0 / math.sqrt(2.0 * math.pi * sigma * sigma)
    return coeff * math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


register(
    Op(
        name="gaussian",
        arity=3,
        arg_types=["scalar", "scalar", "scalar"],
        ret_type="scalar",
        eval_py=_gaussian,
        clause="𝒩(x; μ, σ²)",
        description="Gaussian probability density",
    )
)

register(
    Op(
        name="erf",
        arity=1,
        arg_types=["scalar"],
        ret_type="scalar",
        eval_py=math.erf,
        clause="erf(x)",
        description="Error function",
    )
)


@dataclass(frozen=True)
class QFormat:
    sign: int
    intw: int
    fracw: int


register(
    Op(
        name="q_format",
        arity=3,
        arg_types=["scalar", "scalar", "scalar"],
        ret_type="qfmt",
        eval_py=lambda s, i, f: QFormat(int(s), int(i), int(f)),
        clause="Q(sign, int, frac)",
        description="Fixed-point format descriptor",
    )
)

register(
    Op(
        name="to_fixed",
        arity=2,
        arg_types=["scalar", "qfmt"],
        ret_type="int",
        eval_py=lambda x, q: int(round(float(x) * (1 << q.fracw))),
        clause="⌊x · 2^{frac}⌉",
        description="Convert floating point to fixed representation",
    )
)

register(
    Op(
        name="from_fixed",
        arity=2,
        arg_types=["int", "qfmt"],
        ret_type="scalar",
        eval_py=lambda z, q: float(z) / (1 << q.fracw),
        clause="z / 2^{frac}",
        description="Convert fixed representation back to float",
    )
)

# --- Physics domain operators ------------------------------------------------

register(
    Op(
        name="scalar_field",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_scalar_field,
        clause="ϕ(x) = ϕ₀(1 + kx)",
        description="Scalar field propagation",
    )
)

register(
    Op(
        name="tensor_invariant",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_tensor_invariant,
        clause="det(T)",
        description="Tensor invariant via determinant",
    )
)

register(
    Op(
        name="em_field_tensor",
        arity=1,
        arg_types=["params"],
        ret_type="mat",
        eval_py=_em_field_tensor,
        clause="F_{μν}",
        description="Electromagnetic field tensor",
    )
)

register(
    Op(
        name="dalembertian",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_dalembertian,
        clause="□ϕ",
        description="Wave operator in spacetime",
    )
)

register(
    Op(
        name="covariant_derivative",
        arity=1,
        arg_types=["params"],
        ret_type="vec",
        eval_py=_covariant_derivative,
        clause="D_μ",
        description="Covariant derivative under a connection",
    )
)

register(
    Op(
        name="stress_energy_tensor",
        arity=1,
        arg_types=["params"],
        ret_type="mat",
        eval_py=_stress_energy_tensor,
        clause="T_{μν}",
        description="Stress-energy tensor for energy-momentum conservation",
    )
)

register(
    Op(
        name="metric_tensor",
        arity=1,
        arg_types=["params"],
        ret_type="mat",
        eval_py=_metric_tensor,
        clause="g_{μν}",
        description="Metric tensor describing spacetime interval",
    )
)

register(
    Op(
        name="kronecker_delta",
        arity=2,
        arg_types=["index", "index"],
        ret_type="scalar",
        eval_py=_kronecker_delta,
        clause="δᵢⱼ",
        description="Kronecker delta identity",
    )
)

register(
    Op(
        name="levi_civita",
        arity=3,
        arg_types=["index", "index", "index"],
        ret_type="scalar",
        eval_py=_levi_civita,
        clause="εᵢⱼₖ",
        description="Levi-Civita antisymmetric symbol",
    )
)

register(
    Op(
        name="fourier_transform",
        arity=1,
        arg_types=["symbol"],
        ret_type="expr",
        eval_py=_fourier_transform,
        clause="F[ϕ](ω)",
        description="Symbolic Fourier transform",
    )
)

register(
    Op(
        name="inverse_fourier_transform",
        arity=1,
        arg_types=["symbol"],
        ret_type="expr",
        eval_py=_inverse_fourier_transform,
        clause="F⁻¹[ϕ](t)",
        description="Inverse Fourier transform",
    )
)

register(
    Op(
        name="hamiltonian",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_hamiltonian,
        clause="H = T + V",
        description="Hamiltonian total energy",
    )
)

register(
    Op(
        name="lagrangian",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_lagrangian,
        clause="L = T - V",
        description="Lagrangian energy-action",
    )
)

register(
    Op(
        name="action_integral",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_action_integral,
        clause="S = ∫ L dt",
        description="Action integral over time",
    )
)

register(
    Op(
        name="commutator",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_commutator,
        clause="[A, B]",
        description="Commutator capturing non-commutativity",
    )
)

register(
    Op(
        name="poisson_bracket",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_poisson_bracket,
        clause="{F, G}",
        description="Poisson bracket in phase space",
    )
)

register(
    Op(
        name="euler_lagrange",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_euler_lagrange,
        clause="∂L/∂q - d/dt (∂L/∂q̇) = 0",
        description="Euler-Lagrange equation",
    )
)

register(
    Op(
        name="boltzmann_entropy",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_boltzmann_entropy,
        clause="S = k_B ln Ω",
        description="Boltzmann entropy",
    )
)

register(
    Op(
        name="partition_function",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_partition_function,
        clause="Z = Σ e^{-βE}",
        description="Canonical partition function",
    )
)

register(
    Op(
        name="expectation_value",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_expectation_value,
        clause="⟨O⟩ = Σ pₙ Oₙ",
        description="Expectation value",
    )
)

register(
    Op(
        name="density_matrix",
        arity=1,
        arg_types=["params"],
        ret_type="mat",
        eval_py=lambda params: params.get("rho", ((1.0,),)),
        clause="ρ",
        description="Density matrix for mixed states",
    )
)

register(
    Op(
        name="quantum_operator",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_quantum_operator,
        clause="Ô|ψ⟩",
        description="Apply an operator to a quantum state",
    )
)

register(
    Op(
        name="schrodinger_eq",
        arity=1,
        arg_types=["params"],
        ret_type="complex",
        eval_py=_schrodinger,
        clause="iħ dψ/dt = Hψ",
        description="Schrödinger equation of motion",
    )
)

register(
    Op(
        name="heisenberg_eq",
        arity=1,
        arg_types=["params"],
        ret_type="complex",
        eval_py=_heisenberg,
        clause="dÔ/dt = (i/ħ)[H, Ô]",
        description="Heisenberg equation of motion",
    )
)

register(
    Op(
        name="uncertainty_relation",
        arity=1,
        arg_types=["params"],
        ret_type="bool",
        eval_py=_uncertainty,
        clause="Δx Δp ≥ ħ/2",
        description="Heisenberg uncertainty principle",
    )
)

register(
    Op(
        name="path_integral",
        arity=1,
        arg_types=["params"],
        ret_type="complex",
        eval_py=_path_integral,
        clause="∫ D[ϕ] e^{iS/ħ}",
        description="Quantum path integral",
    )
)

register(
    Op(
        name="green_function",
        arity=0,
        arg_types=[],
        ret_type="expr",
        eval_py=_green_function,
        clause="G(x, t; x', t')",
        description="Green's function",
    )
)

register(
    Op(
        name="propagator",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_propagator,
        clause="K = ⟨x|e^{-iHt/ħ}|x'⟩",
        description="Quantum propagator",
    )
)

register(
    Op(
        name="operator_norm",
        arity=1,
        arg_types=["tensor"],
        ret_type="scalar",
        eval_py=_operator_norm,
        clause="||A||",
        description="Operator norm via Frobenius metric",
    )
)

register(
    Op(
        name="spectral_decomposition",
        arity=1,
        arg_types=["params"],
        ret_type="list",
        eval_py=_spectral_decomposition,
        clause="A = Σ λ P",
        description="Spectral decomposition of an operator",
    )
)

register(
    Op(
        name="commutation_relation",
        arity=1,
        arg_types=["params"],
        ret_type="complex",
        eval_py=lambda params: 1j * float(params.get("hbar", 1.0)),
        clause="[x, p] = iħ",
        description="Canonical commutation relation",
    )
)

register(
    Op(
        name="lie_derivative",
        arity=1,
        arg_types=["params"],
        ret_type="map",
        eval_py=_lie_derivative,
        clause="ℒ_X T",
        description="Lie derivative capturing flow of tensors",
    )
)

register(
    Op(
        name="maxwell_clause",
        arity=1,
        arg_types=["params"],
        ret_type="map",
        eval_py=_maxwell_clause,
        clause="∮ E·dA, ∮ B·dA, ...",
        description="Bundle of Maxwell equation clauses",
    )
)

register(
    Op(
        name="time_evolution",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_time_evolution,
        clause="e^{Γt}",
        description="Time evolution under exponential growth/decay",
    )
)

register(
    Op(
        name="time_entropy",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_time_entropy,
        clause="S_t = k t",
        description="Entropy accumulation over time",
    )
)

register(
    Op(
        name="time_uncertainty",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_time_uncertainty,
        clause="Δt ≈ ħ/(2ΔE)",
        description="Energy-time uncertainty relation",
    )
)

register(
    Op(
        name="spacetime_interval",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_spacetime_interval,
        clause="s² = c²t² - x² - y² - z²",
        description="Invariant spacetime interval",
    )
)

register(
    Op(
        name="gravitational_time_dilation",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_gravitational_time_dilation,
        clause="Δt' = Δt √(1 + 2Φ/c²)",
        description="Time dilation from gravitational potential",
    )
)

register(
    Op(
        name="relativistic_time_dilation",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_relativistic_time_dilation,
        clause="γ = 1/√(1 - v²/c²)",
        description="Velocity-induced time dilation",
    )
)

register(
    Op(
        name="quantum_tunneling_time",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_quantum_tunneling_time,
        clause="τ ∝ a √(V₀ - E)",
        description="Estimated tunneling time",
    )
)

register(
    Op(
        name="wormhole",
        arity=1,
        arg_types=["params"],
        ret_type="map",
        eval_py=_wormhole,
        clause="Traversable wormhole metric",
        description="Simplified wormhole descriptor",
    )
)

register(
    Op(
        name="plasma_frequency",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_plasma_frequency,
        clause="ω_p = √(ρ/(mε))",
        description="Plasma oscillation frequency",
    )
)

register(
    Op(
        name="fractal_dimension",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_fractal_dimension,
        clause="D = log N / log(1/r)",
        description="Fractal dimension",
    )
)

register(
    Op(
        name="cosmological_expansion",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_cosmological_expansion,
        clause="H(t) a(t)",
        description="Hubble expansion rate",
    )
)

register(
    Op(
        name="memory_decay",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_memory_decay,
        clause="M(t) = M₀ e^{-λt}",
        description="Exponential memory decay",
    )
)

register(
    Op(
        name="recursive_feedback",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_recursive_feedback,
        clause="x/(1 - f)",
        description="Recursive feedback stabilizer",
    )
)

register(
    Op(
        name="observer_effect",
        arity=1,
        arg_types=["params"],
        ret_type="map",
        eval_py=_observer_effect,
        clause="Measurement disturbs system",
        description="Observer effect metadata",
    )
)

register(
    Op(
        name="consciousness_operator",
        arity=1,
        arg_types=["params"],
        ret_type="map",
        eval_py=_consciousness_operator,
        clause="Φ = awareness × integration",
        description="Integrated information proxy",
    )
)

register(
    Op(
        name="adaptive_feedback",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_adaptive_feedback,
        clause="Δ = α(s - r)",
        description="Adaptive control feedback",
    )
)

register(
    Op(
        name="time_perception",
        arity=1,
        arg_types=["params"],
        ret_type="scalar",
        eval_py=_time_perception,
        clause="τ ∝ stimulus / attention",
        description="Perceived time scaling",
    )
)


# Attach lowering stubs for every backend.
for op in REGISTRY.values():
    for backend_name in BACKENDS:
        backend_descriptor = get_backend(backend_name)

        def _make_lower(op_ref: Op, descriptor: BackendDescriptor) -> Callable[..., str]:
            def _lower(*args: Any) -> str:
                arg_strings = [str(arg) for arg in args]
                return descriptor.render_invocation(op_ref.name, arg_strings, op_ref.clause)

            return _lower

        setattr(op, f"lower_{backend_name}", _make_lower(op, backend_descriptor))


__all__ = ["REGISTRY", "Op", "BACKENDS", "QFormat"]
