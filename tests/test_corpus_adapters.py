import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from natsume_simple.artifact_builder import BuildMetadata, CorpusRecord, SourceDocument
from natsume_simple.corpus_pipeline import (
    PipelineRejected,
    RejectionLimits,
    adapt_jnlp_directory,
    adapt_ted_iwslt_archive,
    adapt_wikipedia_parquet,
    build_corpus_artifact,
    enforce_rejection_limits,
    extract_collocations,
    prepare_jnlp_archive,
    segment_documents,
    source_content_sha256,
)
from tests.token_observations import EXTRACTION_OBSERVATIONS, observed_doc


def test_source_content_hash_frames_ordered_utf8_units():
    assert source_content_sha256(("一行目。", "二行目。")) == (
        "2ed1b83c19688e8f8a24142f973ff992face1c71641953875b40cb54b3583a5f"
    )


def write_iwslt_archive(path: Path, japanese_training: str) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ja-en/train.tags.ja-en.ja", japanese_training)
    return path


def test_ted_adapter_groups_rows_by_talk_and_counts_empty_talks(tmp_path: Path):
    archive = write_iwslt_archive(
        tmp_path / "ja-en.zip",
        """<doc docid="1" genre="lectures">
<url>https://www.ted.com/talks/example</url>
<speaker>話者</speaker>
<talkid>42</talkid>
<title>例の講演</title>
一行目。
二行目。
</doc>
<doc docid="2" genre="lectures">
<talkid>43</talkid>
<title>空の講演</title>
</doc>
""",
    )

    result = adapt_ted_iwslt_archive(archive)

    assert result.corpus_id == "ted"
    assert result.rejections == {"empty_subtitle_text": 1}
    assert len(result.documents) == 1
    talk = result.documents[0]
    assert talk.external_id == "42"
    assert talk.title == "例の講演"
    assert talk.year is None
    assert talk.author == "話者"
    assert talk.publisher == "TED Conference LLC"
    assert talk.url == "https://www.ted.com/talks/example"
    assert talk.text_units == ("一行目。", "二行目。")
    assert talk.content_sha256 == source_content_sha256(talk.text_units)


@pytest.mark.parametrize(
    ("training", "reason"),
    [
        ("<doc>\n<title>missing</title>\n本文。\n</doc>", "ted_identity_missing"),
        (
            "<doc>\n<talkid>1</talkid>\n本文。\n</doc>\n"
            "<doc>\n<talkid>1</talkid>\n別文。\n</doc>",
            "ted_identity_duplicate",
        ),
        (
            "<doc>\n<talkid>1</talkid>\n<title>A</title>\n"
            "<title>B</title>\n本文。\n</doc>",
            "ted_metadata_conflict",
        ),
    ],
)
def test_ted_adapter_aborts_on_identity_errors(
    tmp_path: Path, training: str, reason: str
):
    archive = write_iwslt_archive(tmp_path / "ja-en.zip", training)

    with pytest.raises(ValueError, match=f"^{reason}$"):
        adapt_ted_iwslt_archive(archive)


def test_ted_adapter_requires_exactly_one_named_member(tmp_path: Path):
    import zipfile

    missing = tmp_path / "missing.zip"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr("other.txt", "text")
    with pytest.raises(ValueError, match="^ted_archive_member_invalid$"):
        adapt_ted_iwslt_archive(missing)

    duplicate = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("ja-en/train.tags.ja-en.ja", "<doc></doc>")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("ja-en/train.tags.ja-en.ja", "<doc></doc>")
    with pytest.raises(ValueError, match="^ted_archive_member_invalid$"):
        adapt_ted_iwslt_archive(duplicate)
    assert source_content_sha256(("一行目。二行目。",)) != source_content_sha256(
        ("一行目。", "二行目。")
    )


def test_jnlp_archive_preparation_uses_declared_converters(tmp_path: Path, monkeypatch):
    import subprocess
    import zipfile

    archive = tmp_path / "jnlp.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("NLP_LATEX_CORPUS/file_DB.xls", "metadata")
        bundle.writestr("NLP_LATEX_CORPUS/V01/lesson.tex", "教材です。")

    commands: list[list[str]] = []

    def run(command: list[str], *, check: bool):
        assert check
        commands.append(command)
        if command[0] == "pandoc":
            destination = Path(command[command.index("-o") + 1])
            destination.write_text("教材です。", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("natsume_simple.corpus_pipeline.subprocess.run", run)

    root = prepare_jnlp_archive(archive, tmp_path / "prepared")

    assert root == tmp_path / "prepared" / "NLP_LATEX_CORPUS"
    assert (root / "V01" / "lesson.txt").read_text() == "教材です。"
    assert [command[0] for command in commands] == ["nkf", "pandoc"]


def test_jnlp_archive_preparation_rejects_parent_paths(tmp_path: Path):
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.tex", "unsafe")

    with pytest.raises(ValueError, match="unsafe archive member"):
        prepare_jnlp_archive(archive, tmp_path / "prepared")


def test_jnlp_adapter_reads_metadata_and_counts_missing_plaintext(tmp_path: Path):
    corpus_root = tmp_path / "NLP_LATEX_CORPUS"
    (corpus_root / "V01").mkdir(parents=True)
    (corpus_root / "V01" / "V01N01-01.txt").write_text("教材を読む。", encoding="utf-8")
    metadata_path = corpus_root / "file_DB.xlsx"
    pl.DataFrame(
        {
            "状態": [1, 1],
            "ファイル名": ["V01/V01N01-01.tex", "V01N01-02.tex"],
            "Vol": [1, 1],
            "No": [1, 1],
            "タイトル": ["教材", "欠落"],
            "著者": ["著者", None],
            "J-Stageにおける論文URL": ["https://example.test/lesson", None],
        }
    ).write_excel(metadata_path)

    result = adapt_jnlp_directory(corpus_root, metadata_path=metadata_path)

    assert result.rejections == {"missing_plain_text": 1}
    assert len(result.documents) == 1
    document = result.documents[0]
    assert document.corpus_id == "jnlp"
    assert document.external_id == "V01/V01N01-01.tex"
    assert document.title == "教材"
    assert document.year == 1994
    assert document.author == "著者"
    assert document.publisher == "自然言語処理"
    assert document.url == "https://example.test/lesson"
    assert document.text_units == ("教材を読む。",)
    assert (
        document.content_sha256
        == "f0510a814ea358642b12f3490a1504b8b7583198a6d03c749aaba0e54d070131"
    )


def test_wikipedia_adapter_reads_only_selected_articles_in_identity_order(
    tmp_path: Path,
):
    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "id": ["2", "1", "3"],
            "url": [
                "https://ja.wikipedia.org/wiki/第二",
                "https://ja.wikipedia.org/wiki/第一",
                "https://ja.wikipedia.org/wiki/除外",
            ],
            "title": ["第二", "第一", "除外"],
            "text": ["第三文。", "第一文。第二文。", "除外文。"],
        }
    ).write_parquet(source)

    result = adapt_wikipedia_parquet(source, article_ids={"1", "2"})

    assert result.rejections == {}
    assert [document.external_id for document in result.documents] == ["1", "2"]
    assert result.documents[0].text_units == ("第一文。第二文。",)
    assert (
        result.documents[0].content_sha256
        == "8d6f8d6e7fbfc4a27e54b52f0c1707214ca66d343eb62ce739617b5af6589e42"
    )


@pytest.mark.parametrize(
    ("ids", "reason"),
    [
        (["1"], "wikipedia_identity_missing"),
        (["1", "1", "2"], "wikipedia_identity_duplicate"),
    ],
)
def test_wikipedia_adapter_requires_exact_selected_membership(
    tmp_path: Path, ids: list[str], reason: str
):
    source = tmp_path / "source.parquet"
    pl.DataFrame(
        {
            "id": ids,
            "url": [None] * len(ids),
            "title": [f"title-{index}" for index in range(len(ids))],
            "text": ["本文です。"] * len(ids),
        }
    ).write_parquet(source)

    with pytest.raises(ValueError, match=reason):
        adapt_wikipedia_parquet(source, article_ids={"1", "2"})


def test_segmentation_assigns_stable_ordinals_and_reports_progress(caplog):
    documents = (
        SourceDocument(
            "wiki",
            "2",
            "第二",
            2023,
            None,
            "Wikimedia Foundation",
            None,
            ("第三文。",),
            "a" * 64,
        ),
        SourceDocument(
            "wiki",
            "1",
            "第一",
            2023,
            None,
            "Wikimedia Foundation",
            None,
            ("第一文。第二文。",),
            "b" * 64,
        ),
    )

    def split(text_units: tuple[str, ...]) -> list[str]:
        if text_units == ("第一文。第二文。",):
            return ["第一文。", "", "第二文。"]
        return ["第三文。"]

    with caplog.at_level(logging.INFO, logger="natsume_simple.corpus_pipeline"):
        sentences = segment_documents(documents, split)

    assert "segmented documents=2/2 sentences=3" in caplog.text
    assert [
        (sentence.source_identity, sentence.ordinal, sentence.text)
        for sentence in sentences
    ] == [
        (("wiki", "1"), 0, "第一文。"),
        (("wiki", "1"), 1, "第二文。"),
        (("wiki", "2"), 0, "第三文。"),
    ]


def test_extraction_preserves_sentence_identity_and_offsets(monkeypatch):
    documents = (
        SourceDocument(
            "wiki",
            "1",
            "第一",
            2023,
            None,
            "Wikimedia Foundation",
            None,
            ("ことを説明するならば",),
            "a" * 64,
        ),
    )
    sentences = segment_documents(documents, lambda units: units)
    monkeypatch.setattr(
        "natsume_simple.pattern_extraction.ginza.bunsetu_span",
        lambda token: token.doc[token.i :],
    )

    result = extract_collocations(
        sentences,
        lambda text: observed_doc(EXTRACTION_OBSERVATIONS[text]),
        extractor_id="fixture-extractor",
    )

    assert result.rejections == {}
    assert len(result.occurrences) == 1
    occurrence = result.occurrences[0]
    assert occurrence.source_identity == ("wiki", "1")
    assert occurrence.sentence_ordinal == 0
    assert (occurrence.noun, occurrence.particle, occurrence.verb) == (
        "こと",
        "を",
        "説明する",
    )
    assert occurrence.noun_span == (0, 2)
    assert occurrence.particle_span == (2, 3)
    assert occurrence.verb_span == (3, 10)


def test_rejection_limits_fail_on_absolute_or_fraction_threshold():
    limits = RejectionLimits(max_count=2, max_fraction=0.25)

    enforce_rejection_limits({"missing_plain_text": 1}, total=4, limits=limits)

    for rejections, total in [
        ({"missing_plain_text": 3}, 100),
        ({"missing_plain_text": 2}, 4),
    ]:
        with pytest.raises(PipelineRejected, match="missing_plain_text"):
            enforce_rejection_limits(rejections, total=total, limits=limits)


def test_pipeline_builds_api_artifact_and_records_real_rejections(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "wiki.parquet"
    pl.DataFrame(
        {
            "id": ["1", "2"],
            "url": [None, None],
            "title": ["説明", "空"],
            "text": ["ことを説明するならば", None],
        }
    ).write_parquet(source)
    adaptation = adapt_wikipedia_parquet(source, article_ids={"1", "2"})
    monkeypatch.setattr(
        "natsume_simple.pattern_extraction.ginza.bunsetu_span",
        lambda token: token.doc[token.i :],
    )

    artifact = build_corpus_artifact(
        tmp_path / "artifact",
        corpora=(CorpusRecord("wiki", "Wikipedia"),),
        adaptations=(adaptation,),
        split=lambda units: units,
        parse=lambda text: observed_doc(EXTRACTION_OBSERVATIONS[text]),
        metadata=BuildMetadata(
            artifact_instance_id="pipeline-fixture",
            identity_inputs={"builderRevision": "fixture"},
            built_at=datetime(2026, 8, 12, tzinfo=UTC),
            content_license="Synthetic fixture data.",
            attribution="Generated by the test suite.",
        ),
        rejection_limits=RejectionLimits(max_count=2, max_fraction=0.75),
        extractor_id="fixture-extractor",
    )

    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["rejectionCounts"] == {
        "extraction": {},
        "wiki": {"empty_text": 1},
    }
    assert manifest["rejectionLimits"] == {"maxCount": 2, "maxFraction": 0.75}
    assert manifest["rejectionTotals"] == {"extraction": 1, "wiki": 2}
