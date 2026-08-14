import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from natsume_simple import builder_cli
from natsume_simple import release_check
from natsume_simple import release_inputs
from natsume_simple.corpus_pipeline import AdaptationResult
from natsume_simple.release_inputs import LockedFile, ReleaseSources, WikipediaSubset


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


def test_build_parser_accepts_locked_release_metadata(tmp_path: Path):
    args = builder_cli._parser().parse_args(
        [
            "build",
            "--artifacts-directory",
            str(tmp_path / "artifacts"),
            "--wikipedia-parquet",
            str(tmp_path / "train-00000-of-00015.parquet"),
            "--ted-iwslt-archive",
            str(tmp_path / "ja-en.zip"),
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
    assert args.ted_iwslt_archive == tmp_path / "ja-en.zip"


def test_build_requires_source_lock_for_ted(tmp_path: Path, capsys):
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
                "--ted-iwslt-archive",
                str(tmp_path / "ja-en.zip"),
                "--splitter-model",
                str(model),
                "--content-license",
                str(license_file),
                "--attribution",
                str(attribution),
            ]
        )
    assert "TED requires --source-lock" in capsys.readouterr().err


def test_split_japanese_sentences_keeps_language_policy_outside_segmentation():
    class Splitter:
        def split(self, paragraphs: list[str]):
            assert paragraphs == ["日本語の段落です。", "English paragraph."]
            return [["日本語の文章です。"], ["English sentence."]]

    observations = Counter()
    assert list(
        builder_cli.split_japanese_sentences(
            ("日本語の段落です。\nEnglish paragraph.",),
            splitter=Splitter(),
            observations=observations,
        )
    ) == ["日本語の文章です。"]
    assert observations == {"candidate": 2, "retained": 1, "dropped": 1}


def test_inspect_inputs_prints_adapter_counts(monkeypatch, tmp_path: Path, capsys):
    summary = {
        "jnlp": {"acceptedSources": 5, "rejections": {"missing_source_path": 1}},
        "ted": {
            "acceptedSources": 1,
            "textUnits": 2,
            "rejections": {"empty_subtitle_text": 1},
        },
        "wiki": {"acceptedSources": 971, "rejections": {}},
    }
    monkeypatch.setattr(builder_cli, "_inspect_inputs", lambda args: summary)

    assert (
        builder_cli.main(
            [
                "inspect-inputs",
                "--source-lock",
                str(tmp_path / "sources.json"),
                "--wikipedia-subset",
                str(tmp_path / "subset.json"),
                "--jnlp-root",
                str(tmp_path / "jnlp"),
                "--wikipedia-parquet",
                str(tmp_path / "train-00000-of-00015.parquet"),
                "--ted-iwslt-archive",
                str(tmp_path / "ja-en.zip"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == summary


def test_build_rejects_wrong_wikipedia_bytes_before_model_loading(
    monkeypatch, tmp_path: Path, capsys
):
    model = tmp_path / "model"
    model.mkdir()
    wikipedia = tmp_path / "train-00000-of-00015.parquet"
    wikipedia.write_bytes(b"wrong")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    attribution = tmp_path / "attribution.md"
    attribution.write_text("attribution")
    monkeypatch.setattr(
        "wtpsplit.SaT", lambda *_args, **_kwargs: pytest.fail("model loaded")
    )

    with pytest.raises(SystemExit, match="2"):
        builder_cli.main(
            [
                "build",
                "--artifacts-directory",
                str(tmp_path / "artifacts"),
                "--wikipedia-parquet",
                str(wikipedia),
                "--source-lock",
                "docs/corpus-sources.lock.json",
                "--wikipedia-subset",
                "docs/wikipedia-ja-20231101-subset.json",
                "--splitter-model",
                str(model),
                "--content-license",
                str(license_file),
                "--attribution",
                str(attribution),
            ]
        )
    assert "source_size_mismatch" in capsys.readouterr().err


def test_build_records_release_sources_and_sentence_policy(monkeypatch, tmp_path: Path):
    import spacy
    import torch
    import wtpsplit
    from natsume_simple import corpus_pipeline

    model = tmp_path / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"model")
    wikipedia = tmp_path / "train-00000-of-00015.parquet"
    wikipedia.write_bytes(b"fixture")
    ted_archive = tmp_path / "ja-en.zip"
    ted_archive.write_bytes(b"ted")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    attribution = tmp_path / "attribution.md"
    attribution.write_text("attribution")
    jnlp = LockedFile("jnlp", "NLP_LATEX_CORPUS.zip", "jnlp.zip", "jnlp", 1, "a" * 64)
    wiki = LockedFile(
        "wikipedia-ja-20231101",
        wikipedia.name,
        wikipedia.name,
        "wiki",
        len(b"fixture"),
        hashlib.sha256(b"fixture").hexdigest(),
    )
    ted = LockedFile(
        "ted-iwslt-2017-ja-en",
        "ja-en.zip",
        "ja-en.zip",
        "ted",
        3,
        hashlib.sha256(b"ted").hexdigest(),
    )
    sources = ReleaseSources(jnlp, wiki, ted, "c" * 64)
    captured = {}
    torch_thread_counts: list[int] = []

    class Splitter:
        def eval(self):
            return self

        def to(self, _device: str):
            return self

        def split(self, paragraphs: list[str]):
            return [[paragraph] for paragraph in paragraphs]

    monkeypatch.setattr(release_inputs, "load_release_sources", lambda _path: sources)
    monkeypatch.setattr(
        release_inputs,
        "load_wikipedia_subset",
        lambda _path, *, sources: WikipediaSubset(
            "wikipedia-ja-20231101", tuple(str(index) for index in range(971))
        ),
    )
    monkeypatch.setattr(
        release_inputs, "validate_wikipedia_paths", lambda paths, *, sources: paths[0]
    )
    monkeypatch.setattr(release_inputs, "verify_file", lambda _path, _locked: None)
    monkeypatch.setattr(
        corpus_pipeline,
        "adapt_wikipedia_parquet",
        lambda _path, *, article_ids: AdaptationResult("wiki", (), {}),
    )
    monkeypatch.setattr(
        corpus_pipeline,
        "adapt_ted_iwslt_archive",
        lambda _path: AdaptationResult("ted", (), {}),
    )
    monkeypatch.setattr(
        corpus_pipeline,
        "build_corpus_artifact",
        lambda output, **kwargs: captured.update(kwargs) or output,
    )
    monkeypatch.setattr(wtpsplit, "SaT", lambda _path: Splitter())
    monkeypatch.setattr(spacy, "load", lambda _name: object())
    monkeypatch.setattr(torch, "set_num_threads", torch_thread_counts.append)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _count: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda _enabled: None)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(torch, "__version__", "fixture-torch")
    monkeypatch.setattr(
        builder_cli.importlib.metadata,
        "version",
        lambda name: {
            "natsume-simple": "0.3.0",
            "spacy": "3.8.11",
            "ja-ginza": "5.2.0",
            "wtpsplit": "2.2.1",
        }[name],
    )
    args = SimpleNamespace(
        jnlp_root=None,
        wikipedia_parquet=[wikipedia],
        ted_iwslt_archive=ted_archive,
        source_lock=tmp_path / "sources.json",
        wikipedia_subset=tmp_path / "subset.json",
        splitter_model=model,
        artifacts_directory=tmp_path / "artifacts",
        artifact_instance_id="fixture-build",
        content_license=license_file,
        attribution=attribution,
        max_rejections=2,
        max_rejection_fraction=0.5,
    )

    builder_cli._build(args)

    identity = captured["metadata"].identity_inputs
    assert torch_thread_counts == [8]
    assert identity["executionProfile"]["torchThreads"] == 8
    assert identity["sentenceFilter"] == {"name": "is_japanese", "minLength": 5}
    assert identity["sourceContentHash"] == "natsume-source-content-v1"
    assert [corpus.id for corpus in captured["corpora"]] == ["ted", "wiki"]
    assert identity["sourceAdapters"] == ["ted", "wiki"]
    assert identity["tedSelectionPolicy"] == "iwslt2017-ja-en-training-v1"
    assert identity["sentenceSplitter"]["modelSha256"] == builder_cli.path_sha256(model)
    assert identity["sourceFiles"] == [
        {
            "corpusId": "ted-iwslt-2017-ja-en",
            "name": "ja-en.zip",
            "sha256": hashlib.sha256(b"ted").hexdigest(),
            "size": 3,
        },
        {
            "corpusId": "wikipedia-ja-20231101",
            "name": wikipedia.name,
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
            "size": len(b"fixture"),
        },
    ]


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


def test_release_check_command_prints_the_structural_summary(
    monkeypatch, tmp_path: Path, capsys
):
    artifact = tmp_path / "artifact"
    source_lock_path = tmp_path / "sources.json"
    subset = tmp_path / "subset.json"
    summary = {"artifactInstanceId": "release", "corpusIds": ["jnlp", "wiki"]}
    monkeypatch.setattr(
        release_check,
        "check_release_artifact",
        lambda selected, *, source_lock, wikipedia_subset: (
            summary
            if (selected, source_lock, wikipedia_subset)
            == (artifact, source_lock_path, subset)
            else pytest.fail("unexpected release-check arguments")
        ),
    )

    assert (
        builder_cli.main(
            [
                "release-check",
                str(artifact),
                "--source-lock",
                str(source_lock_path),
                "--wikipedia-subset",
                str(subset),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == summary
