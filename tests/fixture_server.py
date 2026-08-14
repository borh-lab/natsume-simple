import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from natsume_simple.api import create_app
from tests.database_fixture import build_search_artifact


def main() -> None:
    api_port = int(os.environ.get("NATSUME_TEST_API_PORT", "8000"))
    frontend_port = int(os.environ.get("NATSUME_TEST_FRONTEND_PORT", "4173"))
    with tempfile.TemporaryDirectory(prefix="natsume-fixture-") as directory:
        artifact = build_search_artifact(Path(directory) / "artifact")
        app = create_app(artifact)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[f"http://127.0.0.1:{frontend_port}"],
            allow_methods=["GET"],
            allow_headers=["*"],
        )
        uvicorn.run(app, host="127.0.0.1", port=api_port, log_level="warning")


if __name__ == "__main__":
    main()
