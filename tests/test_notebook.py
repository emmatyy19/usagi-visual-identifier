"""Publication and structure checks for committed notebooks."""

import json
from pathlib import Path
from typing import cast


def test_embedding_walkthrough_is_an_executed_showcase() -> None:
    """The committed notebook is executed, portable, and ready to share."""
    path = (
        Path(__file__).parents[1] / "notebooks" / "embedding_baseline_walkthrough.ipynb"
    )
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    notebook = cast(dict[str, object], raw)
    assert notebook["nbformat"] == 4

    cells = cast(list[dict[str, object]], notebook["cells"])
    assert len(cells) >= 20
    output_count = 0
    image_count = 0
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None
            outputs = cast(list[dict[str, object]], cell["outputs"])
            output_count += len(outputs)
            for output in outputs:
                data = output.get("data")
                if isinstance(data, dict) and "image/png" in data:
                    image_count += 1

    serialized = path.read_text(encoding="utf-8")
    assert output_count > 0
    assert image_count >= 10
    assert "/Users/emmatong" not in serialized
    assert "file:///" not in serialized
    assert "© 2026 Emma Tong. All rights reserved." in serialized
    assert path.stat().st_size < 20_000_000
