"""Browser workflow tests for the personal catalog interface."""

from pathlib import Path
from typing import cast

import httpx2
from fastapi.testclient import TestClient

from usagi_visual_identifier.api.app import create_app


def _form(**changes: str) -> dict[str, str]:
    """Return a complete valid browser form with optional changes."""
    value = {
        "preferred_name": "Magical Usagi",
        "product_type": "plush",
        "distinguishing_details": "Yellow bow and star accessory",
        "notes": "One of my favorites",
        "verification_status": "manually_verified",
        "known_to_exist": "on",
        "owned": "on",
        "interest_level": "nice_to_have",
        "linked_identity_id": "usagi-0004",
    }
    value.update(changes)
    return value


def test_browser_catalog_create_edit_and_filter(tmp_path: Path) -> None:
    """The primary browser workflow persists records through server forms."""
    app = create_app(database_path=tmp_path / "catalog.sqlite3")
    with TestClient(app) as test_client:
        client = cast(httpx2.Client, test_client)
        home = client.get("/")
        assert home.status_code == 200
        assert "No dolls here yet" in home.text
        assert client.get("/static/catalog.css").status_code == 200
        assert "Add a doll" in client.get("/catalog/new").text

        created = client.post("/catalog/new", data=_form(), follow_redirects=False)
        assert created.status_code == 303
        location = created.headers["location"]
        assert location.startswith("/catalog/item-")

        detail = client.get(location)
        assert detail.status_code == 200
        assert "Magical Usagi" in detail.text
        assert "usagi-0004" in detail.text
        owned_shelf = client.get("/catalog")
        assert 'class="interest interest-nice_to_have"' not in owned_shelf.text
        assert "No records match" in client.get("/catalog?interest=nice_to_have").text
        assert (
            "Magical Usagi"
            in client.get("/catalog?ownership=owned&interest=nice_to_have").text
        )

        record_id = location.rsplit("/", 1)[1]
        updated_form = _form(
            preferred_name="Magical Usagi Mascot",
            interest_level="actively_looking",
            notes="Updated from the product tag",
        )
        updated_form.pop("owned")
        updated = client.post(
            f"/catalog/{record_id}",
            data=updated_form,
            follow_redirects=False,
        )
        assert updated.status_code == 303

        catalog = client.get("/catalog")
        assert "Magical Usagi Mascot" in catalog.text
        assert "Actively looking" in catalog.text
        assert "1" in catalog.text

        owned_only = client.get("/catalog", params={"ownership": "owned"})
        assert "No records match these filters" in owned_only.text
        looking = client.get("/catalog", params={"interest": "actively_looking"})
        assert "Magical Usagi Mascot" in looking.text
        assert client.get("/catalog?ownership=all&interest=").status_code == 200
        assert client.get("/catalog?interest=invalid").status_code == 422


def test_browser_forms_render_domain_errors_and_missing_page(tmp_path: Path) -> None:
    """Expected form conflicts and missing records remain readable HTML pages."""
    app = create_app(database_path=tmp_path / "catalog.sqlite3")
    missing = "item-0123456789abcdef0123456789abcdef"
    with TestClient(app) as test_client:
        client = cast(httpx2.Client, test_client)
        first = client.post("/catalog/new", data=_form(), follow_redirects=False)
        assert first.status_code == 303

        conflict = client.post(
            "/catalog/new",
            data=_form(preferred_name="Duplicate identity"),
            follow_redirects=False,
        )
        assert conflict.status_code == 422
        assert "already uses that visual identity" in conflict.text
        assert "Duplicate identity" in conflict.text

        assert client.get(f"/catalog/{missing}").status_code == 404
        assert client.post(f"/catalog/{missing}", data=_form()).status_code == 404

        created_location = first.headers["location"]
        invalid = _form()
        invalid.pop("known_to_exist")
        invalid_update = client.post(created_location, data=invalid)
        assert invalid_update.status_code == 422
        assert "owned catalog record must be known to exist" in invalid_update.text


def test_private_cover_requires_a_linked_record_and_existing_image(
    tmp_path: Path,
) -> None:
    """Covers are served by record identity without exposing the private directory."""
    app = create_app(database_path=tmp_path / "catalog.sqlite3")
    with TestClient(app) as test_client:
        client = cast(httpx2.Client, test_client)
        created = client.post("/catalog/new", data=_form(), follow_redirects=False)
        location = created.headers["location"]
        assert client.get(f"{location}/cover").status_code == 404
        covers = tmp_path / "catalog-covers"
        covers.mkdir()
        image_bytes = b"test cover bytes"
        (covers / "usagi-0004.jpg").write_bytes(image_bytes)
        response = client.get(f"{location}/cover")
        assert response.status_code == 200
        assert response.content == image_bytes
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "private, no-cache"
        assert f"{location}/cover" in client.get("/catalog").text
        assert client.get("/catalog/missing/cover").status_code == 404
        client.post(location, data=_form(linked_identity_id=""))
        assert client.get(f"{location}/cover").status_code == 404
