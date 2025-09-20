"""Minimal telemetry helpers with graceful fallbacks for offline environments."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace  # type: ignore
    from opentelemetry.trace import Status, StatusCode  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    trace = None
    Status = None
    StatusCode = None

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter as PromCounter  # type: ignore
    from prometheus_client import Gauge as PromGauge  # type: ignore
    from prometheus_client import Histogram as PromHistogram  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    PromCounter = None
    PromGauge = None
    PromHistogram = None


_FALLBACK_LOCK = Lock()
_FALLBACK_SPANS: list[Dict[str, Any]] = []
_FALLBACK_HISTOGRAMS: list[Dict[str, Any]] = []
_FALLBACK_COUNTERS: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
_FALLBACK_GAUGES: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}

_PROM_HISTOGRAMS: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
_PROM_COUNTERS: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
_PROM_GAUGES: Dict[Tuple[str, Tuple[str, ...]], Any] = {}


def _coerce_attributes(attributes: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not attributes:
        return {}
    coerced = {}
    for key, value in attributes.items():
        if value is None:
            continue
        coerced[str(key)] = str(value)
    return coerced


def _attribute_tuple(attributes: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(attributes.items()))


def _record_span_snapshot(name: str, duration: float, attributes: Dict[str, str], error: bool) -> None:
    snapshot = {
        "name": name,
        "duration_seconds": duration,
        "attributes": attributes,
        "error": error,
        "timestamp": time.time(),
    }
    with _FALLBACK_LOCK:
        _FALLBACK_SPANS.append(snapshot)


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Optional[Any]]:
    """Start a span using OpenTelemetry when available, otherwise fallback to local records."""

    coerced = _coerce_attributes(attributes)
    start_time = time.perf_counter()

    if trace is not None:  # pragma: no branch - keep fallback simple
        tracer = trace.get_tracer("upeel.telemetry")

        @contextmanager
        def _otel_context() -> Iterator[Any]:
            with tracer.start_as_current_span(name) as span:
                for key, value in coerced.items():
                    span.set_attribute(key, value)
                try:
                    yield span
                except Exception as exc:  # pragma: no cover - side effects only when OTEL present
                    if Status is not None and StatusCode is not None:
                        span.record_exception(exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise

        ctx = _otel_context()
    else:
        ctx = nullcontext()

    error = False
    try:
        with ctx as span:
            yield span
    except Exception:
        error = True
        raise
    finally:
        duration = time.perf_counter() - start_time
        _record_span_snapshot(name, duration, coerced, error)


def instrument_span(
    name: str,
    attribute_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to wrap a callable in a telemetry span."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            attributes = attribute_fn(*args, **kwargs) if attribute_fn else None
            with start_span(name, attributes):
                return func(*args, **kwargs)

        return _wrapper

    return _decorator


def _store_histogram(name: str, value: float, attributes: Dict[str, str], unit: str) -> None:
    record = {
        "name": name,
        "value": float(value),
        "attributes": attributes,
        "unit": unit,
        "timestamp": time.time(),
    }
    with _FALLBACK_LOCK:
        _FALLBACK_HISTOGRAMS.append(record)


def _get_prom_metric(collection: Dict[Tuple[str, Tuple[str, ...]], Any], factory: Callable[..., Any], name: str, attributes: Dict[str, str]) -> Optional[Any]:
    if factory is None:
        return None
    label_names = tuple(sorted(attributes))
    key = (name, label_names)
    metric = collection.get(key)
    if metric is None:
        documentation = f"Auto-generated metric for {name}"
        metric = factory(name, documentation, labelnames=list(label_names)) if label_names else factory(name, documentation)
        collection[key] = metric
    return metric


def record_histogram(name: str, value: float, attributes: Optional[Dict[str, Any]] = None, unit: str = "seconds") -> None:
    """Record a histogram sample with optional Prometheus export."""

    coerced = _coerce_attributes(attributes)
    _store_histogram(name, value, coerced, unit)

    histogram = _get_prom_metric(_PROM_HISTOGRAMS, PromHistogram, name, coerced)
    if histogram is None:
        return

    if coerced:
        histogram.labels(**coerced).observe(float(value))
    else:
        histogram.observe(float(value))


def increment_counter(name: str, value: float = 1.0, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Increment a counter with optional Prometheus export."""

    coerced = _coerce_attributes(attributes)
    key = (name, _attribute_tuple(coerced))
    with _FALLBACK_LOCK:
        _FALLBACK_COUNTERS[key] = _FALLBACK_COUNTERS.get(key, 0.0) + float(value)

    counter = _get_prom_metric(_PROM_COUNTERS, PromCounter, name, coerced)
    if counter is None:
        return

    if coerced:
        counter.labels(**coerced).inc(float(value))
    else:
        counter.inc(float(value))


def set_gauge(name: str, value: float, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Set a gauge value with optional Prometheus export."""

    coerced = _coerce_attributes(attributes)
    key = (name, _attribute_tuple(coerced))
    with _FALLBACK_LOCK:
        _FALLBACK_GAUGES[key] = float(value)

    gauge = _get_prom_metric(_PROM_GAUGES, PromGauge, name, coerced)
    if gauge is None:
        return

    if coerced:
        gauge.labels(**coerced).set(float(value))
    else:
        gauge.set(float(value))


def get_fallback_metrics() -> Dict[str, Any]:
    """Return a snapshot of locally recorded telemetry for testing or offline analysis."""

    with _FALLBACK_LOCK:
        spans = list(_FALLBACK_SPANS)
        histograms = list(_FALLBACK_HISTOGRAMS)
        counters = {key: value for key, value in _FALLBACK_COUNTERS.items()}
        gauges = {key: value for key, value in _FALLBACK_GAUGES.items()}
    return {
        "spans": spans,
        "histograms": histograms,
        "counters": counters,
        "gauges": gauges,
    }


__all__ = [
    "start_span",
    "instrument_span",
    "record_histogram",
    "increment_counter",
    "set_gauge",
    "get_fallback_metrics",
]
