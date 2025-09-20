"""Code generation toolkit exports."""

from .helpers import COMMENT_SYNTAX, comment
from .backends import (
    BACKEND_REGISTRY,
    BackendDescriptor,
    backend_summary,
    get_backend,
    iter_backends,
    ready_backends,
)

__all__ = [
    "COMMENT_SYNTAX",
    "comment",
    "BACKEND_REGISTRY",
    "BackendDescriptor",
    "backend_summary",
    "get_backend",
    "iter_backends",
    "ready_backends",
]
