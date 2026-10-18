"""Validate the small, explicitly published collection snapshot before deployment."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "site"


def main() -> None:
    """Reject missing photos, extra files, and unreviewed collection fields."""
    data = json.loads((ROOT / "collection.json").read_text())
    assert set(data) == {"owner", "title", "items"}
    expected = {
        "index.html",
        "app.js",
        "styles.css",
        "favicon.svg",
        "collection.json",
        ".nojekyll",
    }
    identities: set[str] = set()
    for item in data["items"]:
        assert set(item) == {"id", "name", "details", "cues", "photo"}
        assert item["id"] not in identities
        identities.add(item["id"])
        path = f"photos/{item['id']}.jpg"
        assert item["photo"] == path
        photo = (ROOT / path).read_bytes()
        assert photo.startswith(b"\xff\xd8") and b"Exif\x00\x00" not in photo
        assert len(photo) < 200_000
        expected.add(path)
    assert len(identities) == 7, "Review the published collection before adding dolls"
    actual = {
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()
    }
    assert actual == expected, (
        f"Unexpected or missing public files: {actual ^ expected}"
    )
    print(f"Public shelf checked: {len(identities)} dolls, {len(actual)} files")


if __name__ == "__main__":
    main()
