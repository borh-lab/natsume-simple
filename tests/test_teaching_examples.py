import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi.testclient import TestClient

from natsume_simple.api import create_app
from natsume_simple.artifact_builder import (
    CollocationOccurrence,
    SentenceRecord,
    SourceDocument,
)
from natsume_simple.corpus_pipeline import (
    adapt_ted_iwslt_archive,
    extract_collocations,
    source_content_sha256,
)
from natsume_simple.data import is_japanese, split_japanese_sentences
from natsume_simple.pattern_extraction import normalize_verb_span
from tests.database_fixture import build_search_artifact
from tests.token_observations import (
    EXTRACTION_OBSERVATIONS,
    NORMALIZATION_OBSERVATIONS,
    observed_doc,
)

TEACHING_BOUNDARIES = {
    "source_adaptation",
    "japanese_filtering",
    "segmentation",
    "normalization",
    "occurrence_construction",
    "aggregation",
    "query_semantics",
}


def test_boundary_inventory_targets_executable_examples():
    inventory_path = Path(__file__).with_name("teaching_boundaries.toml")
    inventory = tomllib.loads(inventory_path.read_text(encoding="utf-8"))

    assert set(inventory) == TEACHING_BOUNDARIES
    for row in inventory.values():
        assert callable(globals().get(row["target"]))
        assert row["behavior"]


def test_source_adaptation_example(tmp_path: Path):
    archive = tmp_path / "ja-en.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        output.writestr(
            "ja-en/train.tags.ja-en.ja",
            "\n".join(
                [
                    "<doc>",
                    "<talkid>lesson-1</talkid>",
                    "<title>教材</title>",
                    "教材を読む。",
                    "</doc>",
                ]
            ),
        )

    adaptation = adapt_ted_iwslt_archive(archive)

    assert adaptation.corpus_id == "ted"
    assert adaptation.rejections == {}
    assert adaptation.documents == (
        SourceDocument(
            corpus_id="ted",
            external_id="lesson-1",
            title="教材",
            year=None,
            author=None,
            publisher="TED Conference LLC",
            url=None,
            text_units=("教材を読む。",),
            content_sha256=source_content_sha256(("教材を読む。",)),
        ),
    )


def test_japanese_filtering_example():
    assert is_japanese("教材です", min_length=4)
    assert not is_japanese("lesson", min_length=4)


def test_segmentation_example():
    class FixtureSplitter:
        def split(self, paragraphs: list[str]) -> list[list[str]]:
            assert paragraphs == ["第一文。第二文。", "第三文。"]
            return [["第一文。", "第二文。"], ["第三文。", ""]]

    assert list(
        split_japanese_sentences(
            ("第一文。第二文。\n\n第三文。",), splitter=FixtureSplitter()
        )
    ) == ["第一文。", "第二文。", "第三文。"]


def test_normalization_example():
    doc = observed_doc(NORMALIZATION_OBSERVATIONS["語ります"])

    assert normalize_verb_span(doc) == ("語る", 0, 2)


def test_occurrence_construction_example(monkeypatch):
    doc = observed_doc(EXTRACTION_OBSERVATIONS["ことを説明するならば"])
    monkeypatch.setattr(
        "natsume_simple.pattern_extraction.ginza.bunsetu_span",
        lambda token: token.doc[token.i :],
    )

    extraction = extract_collocations(
        (SentenceRecord(("lesson", "source-1"), 0, "ことを説明するならば"),),
        lambda _text: doc,
        extractor_id="teaching-extractor",
    )

    assert extraction.rejections == {}
    assert extraction.occurrences == (
        CollocationOccurrence(
            source_identity=("lesson", "source-1"),
            sentence_ordinal=0,
            noun="こと",
            particle="を",
            verb="説明する",
            noun_span=(0, 2),
            particle_span=(2, 3),
            verb_span=(3, 10),
            extractor_id="teaching-extractor",
        ),
    )


def test_aggregation_example(tmp_path: Path):
    artifact = build_search_artifact(tmp_path / "artifact")
    with TestClient(create_app(artifact)) as client:
        response = client.get(
            "/api/collocations",
            params={"term": "情報", "pos": "noun"},
        )

    collocate = next(
        item
        for group in response.json()["particleGroups"]
        if group["particle"] == "を"
        for item in group["items"]
        if item["verb"] == "集める"
    )
    assert collocate["totalRawFrequency"] == 4
    assert {row["corpusId"] for row in collocate["contributions"]} == {
        "alpha",
        "beta",
    }


def test_query_semantics_example(tmp_path: Path):
    artifact = build_search_artifact(tmp_path / "artifact")
    with TestClient(create_app(artifact)) as client:
        noun = client.get(
            "/api/collocations",
            params={"term": "情報", "pos": "noun"},
        ).json()
        verb = client.get(
            "/api/collocations",
            params={"term": "集める", "pos": "verb"},
        ).json()

    assert {
        item["verb"] for group in noun["particleGroups"] for item in group["items"]
    } >= {"集める", "分析する"}
    assert {
        item["noun"] for group in verb["particleGroups"] for item in group["items"]
    } == {"情報"}
