
from importlib import import_module as _imp
from pathlib import Path as _P
_root = _P(__file__).parent / "upeel"
for f in _root.glob("symbolic_operators.py"):
    mod = _imp("upel4.src.upeel.symbolic_operators")
globals().update(mod.REGISTRY)

from pathlib import Path as _P
cli_path=_P(__file__).parent.parent.parent/"upeelctl.py"
