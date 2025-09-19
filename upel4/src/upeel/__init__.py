from .symbolic_operators import REGISTRY
import types, sys

def __getattr__(name):
    op = REGISTRY.get(name)
    if op is None:
        raise AttributeError(name)
    def _fn(*args):
        return op.eval_py(*args)
    return _fn
