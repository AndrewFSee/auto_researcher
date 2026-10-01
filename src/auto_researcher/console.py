"""Console helpers for command-line entry points."""

from __future__ import annotations

import sys


def use_utf8_output() -> None:
    """
    Switch stdout/stderr to UTF-8 (unencodable characters are replaced).

    On Windows, output that is piped or redirected uses the ANSI code page
    (cp1252), so the box-drawing characters and symbols several CLIs print raise
    ``UnicodeEncodeError``, sometimes after the real work is already done.
    Call this at the start of a CLI's ``main()``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
