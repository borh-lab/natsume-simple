import tempfile
from pathlib import Path

import uvicorn

from natsume_simple import server
from tests.database_fixture import build_search_database


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="natsume-fixture-") as directory:
        server.DATABASE_PATH = build_search_database(Path(directory) / "corpus.db")
        uvicorn.run(server.app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
