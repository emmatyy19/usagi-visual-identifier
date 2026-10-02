"""HTTP contract tests for the private personal catalog service."""

from pathlib import Path
from typing import cast

import httpx2
from fastapi.testclient import TestClient

from usagi_visual_identifier.api.app import create_app


def test_catalog_api_lifecycle(tmp_path: Path) -> None:
    """The API creates, lists, reads, and updates one catalog record."""
    app = create_app(database_path=tmp_path / "catalog.sqlite3")
    with TestClient(app) as test_client:
        client = cast(httpx2.Client, test_client)
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/catalog").json() == []

        created_response = client.post(
            "/api/catalog",
            json={
                "preferred_name": "Magical Usagi",
                "owned": True,
                "linked_identity_id": "usagi-0004",
            },
        )
        assert created_response.status_code == 201
        created = created_response.json()
        record_id = created["record_id"]
        assert created["owned"] is True
        assert created["verification_status"] == "unverified"

        assert client.get(f"/api/catalog/{record_id}").json() == created
        assert client.get("/api/catalog").json() == [created]

        updated_response = client.patch(
            f"/api/catalog/{record_id}",
            json={"owned": False, "interest_level": "actively_looking"},
        )
        assert updated_response.status_code == 200
        assert updated_response.json()["interest_level"] == "actively_looking"
        assert updated_response.json()["owned"] is False


def test_catalog_api_reports_validation_missing_and_conflict(tmp_path: Path) -> None:
    """The HTTP boundary keeps validation, missing, and conflict errors distinct."""
    app = create_app(database_path=tmp_path / "catalog.sqlite3")
    missing = "item-0123456789abcdef0123456789abcdef"
    with TestClient(app) as test_client:
        client = cast(httpx2.Client, test_client)
        invalid = client.post(
            "/api/catalog",
            json={"preferred_name": "Impossible", "interest_level": "must_have"},
        )
        assert invalid.status_code == 422
        assert client.get(f"/api/catalog/{missing}").status_code == 404
        assert (
            client.patch(f"/api/catalog/{missing}", json={"notes": "x"}).status_code
            == 404
        )

        first = client.post(
            "/api/catalog",
            json={"preferred_name": "First", "linked_identity_id": "usagi-0001"},
        )
        assert first.status_code == 201
        conflict = client.post(
            "/api/catalog",
            json={"preferred_name": "Second", "linked_identity_id": "usagi-0001"},
        )
        assert conflict.status_code == 409
