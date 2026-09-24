"""Tests for package metadata."""

import usagi_visual_catalog


def test_package_exposes_version() -> None:
    """The installed package exposes its semantic version."""
    assert usagi_visual_catalog.__version__ == "0.1.0"
