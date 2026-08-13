import hashlib
import re
from pathlib import Path

import pytest

from natsume_simple import builder_cli
from natsume_simple import release_inputs


def test_artifact_instance_id_is_utc_timestamp_plus_128_bits():
    instance_id = builder_cli.new_artifact_instance_id(
        timestamp="20260812T153045Z", random_hex="ab" * 16
    )

    assert instance_id == "20260812T153045Z-" + "ab" * 16
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", instance_id)


def test_path_sha256_covers_relative_names_and_contents(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "a").write_bytes(b"first")
    (model / "b").write_bytes(b"second")

    expected = hashlib.sha256()
    for name, content in [("a", b"first"), ("b", b"second")]:
        expected.update(name.encode())
        expected.update(b"\0")
        expected.update(content)

    assert builder_cli.path_sha256(model) == expected.hexdigest()


def test_build_requires_at_least_one_explicit_local_corpus(tmp_path: Path, capsys):
    model = tmp_path / "model"
    model.mkdir()
    license_file = tmp_path / "license.txt"
    license_file.write_text("content license")
    attribution = tmp_path / "attribution.md"
    attribution.write_text("attribution")

    with pytest.raises(SystemExit, match="2"):
        builder_cli.main(
            [
                "build",
                "--artifacts-directory",
                str(tmp_path / "artifacts"),
                "--splitter-model",
                str(model),
                "--content-license",
                str(license_file),
                "--attribution",
                str(attribution),
            ]
        )
    assert "at least one local corpus input is required" in capsys.readouterr().err


def test_build_parser_accepts_locked_wikipedia_metadata(tmp_path: Path):
    args = builder_cli._parser().parse_args(
        [
            "build",
            "--artifacts-directory",
            str(tmp_path / "artifacts"),
            "--wikipedia-parquet",
            str(tmp_path / "train-00000-of-00015.parquet"),
            "--source-lock",
            str(tmp_path / "sources.json"),
            "--wikipedia-subset",
            str(tmp_path / "subset.json"),
            "--splitter-model",
            str(tmp_path / "model"),
            "--content-license",
            str(tmp_path / "license.txt"),
            "--attribution",
            str(tmp_path / "attribution.md"),
        ]
    )

    assert args.source_lock == tmp_path / "sources.json"
    assert args.wikipedia_subset == tmp_path / "subset.json"


def test_publish_command_delegates_to_atomic_registry(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "artifact"
    deploy = tmp_path / "deploy"
    calls = []

    monkeypatch.setattr(
        builder_cli,
        "publish_artifact",
        lambda artifact_directory, deploy_directory: (
            calls.append((artifact_directory, deploy_directory)) or artifact_directory
        ),
    )

    assert builder_cli.main(["publish", str(artifact), str(deploy)]) == 0
    assert calls == [(artifact, deploy)]


def test_acquire_release_inputs_command_uses_the_source_lock(
    monkeypatch, tmp_path: Path, capsys
):
    source_lock = tmp_path / "sources.json"
    output_directory = tmp_path / "inputs"
    sources = object()
    acquired = (output_directory / "jnlp.zip", output_directory / "wiki.parquet")
    monkeypatch.setattr(release_inputs, "load_release_sources", lambda path: sources)
    monkeypatch.setattr(
        release_inputs,
        "acquire_release_inputs",
        lambda selected, output: (
            acquired
            if (selected, output) == (sources, output_directory)
            else pytest.fail("unexpected acquisition arguments")
        ),
    )

    assert (
        builder_cli.main(
            [
                "acquire-release-inputs",
                "--source-lock",
                str(source_lock),
                "--output-directory",
                str(output_directory),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [str(path) for path in acquired]
