import json
from pathlib import Path

import pytest

from natsume_simple.api import ArtifactValidationError
from natsume_simple.artifact_registry import current_artifact, publish_artifact
from tests.test_artifact_builder import build_fixture


def test_publish_and_rollback_replace_one_current_pointer(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    first = build_fixture(artifacts / "first", "first")
    second = build_fixture(artifacts / "second", "second")
    deploy = tmp_path / "deploy"

    publish_artifact(first, deploy)
    assert current_artifact(deploy) == first.resolve()

    publish_artifact(second, deploy)
    assert current_artifact(deploy) == second.resolve()

    publish_artifact(first, deploy)
    assert current_artifact(deploy) == first.resolve()
    assert not (deploy / "current.next").exists()


def test_invalid_artifact_cannot_change_current_pointer(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    good = build_fixture(artifacts / "good", "good")
    broken = build_fixture(artifacts / "broken", "broken")
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["databaseSha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    deploy = tmp_path / "deploy"
    publish_artifact(good, deploy)

    with pytest.raises(ArtifactValidationError) as error:
        publish_artifact(broken, deploy)

    assert error.value.reason == "checksum_mismatch"
    assert current_artifact(deploy) == good.resolve()
