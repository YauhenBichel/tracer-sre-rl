"""Tests for the main entry point."""

from __future__ import annotations

import contextlib
import sys
from io import StringIO
from unittest.mock import patch

from app.main import main


def test_main_shows_help_with_no_args():
    captured = StringIO()
    with patch.object(sys, "argv", ["app.main"]), patch("sys.stdout", captured):
        main()

    assert "Commands:" in captured.getvalue()


def test_main_unknown_command():
    captured = StringIO()
    with (
        patch.object(sys, "argv", ["app.main", "nonexistent"]),
        patch("sys.stdout", captured),
        contextlib.suppress(SystemExit),
    ):
        main()

    assert "Unknown command" in captured.getvalue()
