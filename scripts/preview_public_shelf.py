"""Preview only the published collection files on localhost."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    """Serve the static collection without exposing the repository root."""
    root = Path(__file__).resolve().parents[1] / "site"
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", 8766), handler) as server:
        print("Public collection preview: http://127.0.0.1:8766")
        server.serve_forever()


if __name__ == "__main__":
    main()
