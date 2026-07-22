"""TrueColor blue -> purple -> pink ANSI banner for the CLI."""

from __future__ import annotations

import os
import sys

_RAW = r"""
╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                         ║
║    ███    ███ ██         ██████  ███████ ███    ██  ██████ ██   ██ ███    ███  █████  ██████  ██   ██   ║
║   ░████  ████░██        ░██░░░██░██░░░░ ░████   ██░██░░░░ ░██  ░██░████ ░████░██░░░██░██░░░██░██  ██    ║
║   ░██░████░██░██        ░██████ ░█████  ░██░██  ██░██     ░███████░██░████░██░███████░██████ ░█████     ║
║   ░██░░██░░██░██        ░██░░░██░██░░   ░██░░██ ██░██     ░██░░░██░██░░██░░██░██░░░██░██░░░██░██░░██    ║
║   ░██ ░░  ░██░███████   ░██████ ░███████░██ ░░████░░██████░██  ░██░██ ░░   ██░██  ░██░██  ██ ░██ ░░██   ║
║   ░░      ░░ ░░░░░░░    ░░░░░░  ░░░░░░░ ░░   ░░░░  ░░░░░░ ░░   ░░ ░░      ░░ ░░   ░░ ░░  ░░  ░░   ░░    ║
║                                                                                                         ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  ML BENCHMARK                                                                                           ║
║  Blood RNA Multiple Sclerosis Classification                                                            ║
║  External model benchmarking & validation                                                               ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""


def _build_color() -> str:
    chars = list(_RAW)
    total = max(sum(1 for c in chars if c != "\n") - 1, 1)
    idx = 0
    out = []
    for ch in chars:
        if ch == "\n":
            out.append(ch)
            continue
        t = idx / total
        if t < 0.5:
            u = t * 2
            a, b = (78, 172, 255), (170, 135, 255)
        else:
            u = (t - 0.5) * 2
            a, b = (170, 135, 255), (255, 112, 166)
        r = round(a[0] + (b[0] - a[0]) * u)
        g = round(a[1] + (b[1] - a[1]) * u)
        bb = round(a[2] + (b[2] - a[2]) * u)
        out.append(f"\033[38;2;{r};{g};{bb}m{ch}")
        idx += 1
    out.append("\033[0m")
    return "".join(out)


def render() -> str:
    """Colored banner on a capable terminal, plain text otherwise."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return _RAW
    return _build_color()
