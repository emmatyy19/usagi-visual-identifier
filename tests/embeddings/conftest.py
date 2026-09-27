"""Shared invented fixtures for embedding tests."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image


@dataclass(frozen=True)
class PreparedFixture:
    """Paths for a tiny complete prepared dataset."""

    private_root: Path
    processed_root: Path


@pytest.fixture
def prepared_fixture(tmp_path: Path) -> PreparedFixture:
    """Create two identities with two prepared images each."""
    private_root = tmp_path / "private"
    processed_root = tmp_path / "processed"
    private_root.mkdir()
    processed_root.mkdir()

    (private_root / "catalog.csv").write_text(
        "catalog_id,display_name,label_status,distinguishing_cues,"
        "product_type,included_in_model\n"
        "usagi-0001,Red,verified,red,plush,true\n"
        "usagi-0002,Green,verified,green,plush,true\n",
        encoding="utf-8",
    )
    (private_root / "images.csv").write_text(
        "image_id,relative_path,catalog_id,unknown_identity_id,"
        "capture_session_id,source_image_id,split,source\n"
        "img-000001,images/usagi-0001/session-001/001.png,usagi-0001,,"
        "session-001,src-000001,train,owner_capture\n"
        "img-000002,images/usagi-0001/session-001/002.png,usagi-0001,,"
        "session-001,src-000002,train,owner_capture\n"
        "img-000003,images/usagi-0002/session-001/001.png,usagi-0002,,"
        "session-001,src-000003,train,owner_capture\n"
        "img-000004,images/usagi-0002/session-001/002.png,usagi-0002,,"
        "session-001,src-000004,train,owner_capture\n",
        encoding="utf-8",
    )

    prepared_rows = [
        "image_id,source_relative_path,processed_relative_path,source_sha256,"
        "processed_sha256,width,height"
    ]
    colors = [(255, 0, 0), (220, 10, 10), (0, 255, 0), (10, 220, 10)]
    for number, color in enumerate(colors, start=1):
        identity_number = 1 if number <= 2 else 2
        photo_number = number if number <= 2 else number - 2
        source_path = (
            f"images/usagi-{identity_number:04d}/session-001/{photo_number:03d}.png"
        )
        processed_path = (
            f"images/usagi-{identity_number:04d}/session-001/{photo_number:03d}.jpg"
        )
        image_path = processed_root / processed_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color).save(image_path)
        prepared_rows.append(
            f"img-{number:06d},{source_path},{processed_path},"
            f"source{number},processed{number},16,12"
        )
    (processed_root / "images.csv").write_text(
        "\n".join(prepared_rows) + "\n", encoding="utf-8"
    )
    return PreparedFixture(private_root=private_root, processed_root=processed_root)
