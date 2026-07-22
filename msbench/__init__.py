"""MS blood RNA ML benchmark (GSE17048).

Public API:
    from msbench import Config, run
"""

from .config import Config, WEIGHTS

__version__ = "1.0.0"
__all__ = ["Config", "WEIGHTS", "run"]


def run(cfg: Config) -> None:
    from .benchmark import run as _run
    _run(cfg)
