"""Source editing and manual availability provenance."""

from pathlib import Path
from typing import cast

import httpx2
from fastapi.testclient import TestClient

from usagi_visual_identifier.api.app import create_app


def test_source_workflow(tmp_path: Path) -> None:
    """Validate links, isolate records, preserve drafts, and track explicit checks."""
    with TestClient(create_app(database_path=tmp_path / "db.sqlite3")) as raw:
        client = cast(httpx2.Client, raw)
        doll = client.post("/api/catalog", json={"preferred_name": "Doll"}).json()
        other = client.post("/api/catalog", json={"preferred_name": "Other"}).json()
        url = f"/api/catalog/{doll['record_id']}/sources"
        page = f"/catalog/{doll['record_id']}/sources"
        assert "No sources saved yet" in client.get(page).text
        value = {
            "url": "https://example.com/doll",
            "availability": "available",
            "checked_now": True,
        }
        response = client.post(url, json=value)
        assert response.status_code == 201
        source = response.json()
        assert source["reviewed_at"]
        assert client.post(url, json=value).status_code == 409
        edit = f"{url}/{source['source_id']}"
        changed = client.put(
            edit, json={**value, "checked_now": False, "label": "Shop"}
        ).json()
        assert changed["reviewed_at"] == source["reviewed_at"]
        changed = client.put(
            edit, json={**value, "checked_now": False, "availability": "sold_out"}
        ).json()
        assert changed["reviewed_at"] is None
        assert (
            client.put(
                f"/api/catalog/{other['record_id']}/sources/{source['source_id']}",
                json=value,
            ).status_code
            == 404
        )
        for invalid in (
            "javascript:alert(1)",
            "file:///tmp/image",
            "https://user:secret@example.com",
        ):
            assert client.post(url, json={"url": invalid}).status_code == 422
        assert client.get("/api/catalog/missing/sources").status_code == 404
        invalid_form = client.post(
            page, data={"url": "bad", "label": "Keep draft", "availability": "unknown"}
        )
        assert invalid_form.status_code == 422
        assert "Keep draft" in invalid_form.text
        saved = client.post(
            page,
            data={
                "url": "https://example.com/other",
                "label": "<script>bad</script>",
                "availability": "preorder",
                "checked_now": "on",
            },
        )
        assert saved.status_code == 200
        assert "Source saved" in saved.text
        assert "Last checked" in saved.text
        assert "&lt;script&gt;bad&lt;/script&gt;" in saved.text
        assert "<script>bad</script>" not in saved.text
        assert client.get(f"/api/catalog/{doll['record_id']}").json() == doll
