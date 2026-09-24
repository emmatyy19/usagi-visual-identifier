"""Tests for package metadata."""

import usagi_visual_identifier


def test_package_exposes_version() -> None:
    """The installed package exposes its semantic version."""
    assert usagi_visual_identifier.__version__ == "0.1.0"
