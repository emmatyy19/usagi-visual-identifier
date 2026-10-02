"""Run the private personal catalog API for local development."""

import uvicorn


def main() -> None:
    """Start the local server in one debugger friendly process."""
    uvicorn.run(
        "usagi_visual_identifier.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
