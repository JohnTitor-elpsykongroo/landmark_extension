"""Put the task-local compatibility shims in front of the broken packages.

Call ``activate()`` before importing ``teethland``:

    from landmark_extension.shims import activate
    activate()
    import teethland

It installs ``landmark_extension/shims`` at the front of ``sys.path`` so that
``import pymeshlab`` resolves to the shim when the real DLL is blocked by
Windows Smart App Control. If the real pymeshlab imports successfully, the
shim is not used at all.
"""

from __future__ import annotations

from pathlib import Path
import sys


SHIM_DIR = Path(__file__).resolve().parent


def real_pymeshlab_available() -> bool:
    import importlib.util

    spec = importlib.util.find_spec('pymeshlab')
    if spec is None or spec.origin is None:
        return False
    if str(SHIM_DIR) in str(Path(spec.origin).resolve()):
        return False
    try:
        import pymeshlab  # noqa: F401
    except Exception:
        return False
    return True


def activate(verbose: bool = True) -> str:
    """Return the name of the module that will serve ``import pymeshlab``."""
    if str(SHIM_DIR) not in sys.path:
        sys.path.insert(0, str(SHIM_DIR))
    # drop a half-imported real module so the shim can take over
    sys.modules.pop('pymeshlab', None)
    try:
        import pymeshlab
        origin = getattr(pymeshlab, '__file__', '?')
        if verbose:
            print(f'[shims] pymeshlab -> {origin}')
        return origin
    except Exception as exc:  # pragma: no cover - only on broken installs
        if verbose:
            print(f'[shims] pymeshlab shim unavailable: {exc}')
        raise
