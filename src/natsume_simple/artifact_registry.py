import os
from pathlib import Path

from natsume_simple.api import validate_artifact


def publish_artifact(artifact_directory: Path, deploy_directory: Path) -> Path:
    """Validate an artifact, then atomically select it as current."""
    artifact = artifact_directory.resolve(strict=True)
    validate_artifact(artifact)
    deploy_directory.mkdir(parents=True, exist_ok=True)
    next_pointer = deploy_directory / "current.next"
    current_pointer = deploy_directory / "current"
    try:
        next_pointer.symlink_to(artifact, target_is_directory=True)
        os.replace(next_pointer, current_pointer)
    except Exception:
        next_pointer.unlink(missing_ok=True)
        raise
    return artifact


def current_artifact(deploy_directory: Path) -> Path:
    """Resolve the artifact selected by the deployment pointer."""
    return (deploy_directory / "current").resolve(strict=True)
