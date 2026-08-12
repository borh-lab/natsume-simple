from pathlib import Path
import tomllib

import polars as pl

from natsume_simple.data import BaseCorpusLoader, GenericCorpusLoader, is_japanese
from natsume_simple.pattern_extraction import normalize_verb_span, process_sentence
from natsume_simple.server import get_npv_query, process_query_results
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


def test_source_adaptation_example(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "lesson_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "metadata.csv").write_text(
        "title,year,file_path\nLesson,2025,lesson.txt\n", encoding="utf-8"
    )
    received_paths: list[list[Path]] = []

    def load_fixture(_loader, paths: list[Path]) -> list[str]:
        received_paths.append(paths)
        return ["教材を読む。"]

    monkeypatch.setattr(GenericCorpusLoader, "_load_sentences", load_fixture)
    entry = next(
        GenericCorpusLoader(data_dir=tmp_path, corpus_name="lesson").load_metadata()
    )

    assert received_paths == [[Path("lesson.txt")]]
    assert entry.sentences == ["教材を読む。"]


def test_japanese_filtering_example():
    assert is_japanese("教材です", min_length=4)
    assert not is_japanese("lesson", min_length=4)


def test_segmentation_example():
    class FixtureSplitter:
        def split(self, paragraphs: list[str]) -> list[list[str]]:
            assert paragraphs == ["第一文。第二文。", "第三文。"]
            return [["第一文。", "第二文。"], ["第三文。", ""]]

    loader = BaseCorpusLoader(data_dir=Path(), corpus_name="lesson")

    assert loader.split_into_sentences(
        ["第一文。第二文。\n\n第三文。"], FixtureSplitter()
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

    words, occurrences = process_sentence(doc, sentence_id=7)

    assert words[0][:4] == (0, 2, "こと", "NOUN")
    assert occurrences == [(7, "こと", "を", "説明する", 0, 2, 2, 3, 3, 10)]


def test_aggregation_example():
    rows = pl.DataFrame(
        {
            "n": ["情報"],
            "p": ["を"],
            "v": ["集める"],
            "contributions": [
                [
                    {"corpus": "alpha", "frequency": 3},
                    {"corpus": "beta", "frequency": 1},
                ]
            ],
        }
    )

    result = process_query_results(rows, ["を"], {"alpha": 1 / 3, "beta": 1})
    collocate = result["を"]["collocates"][0]

    assert collocate["totalRawFrequency"] == 4
    assert collocate["totalNormalizedFrequency"] == 2


def test_query_semantics_example():
    noun_query, noun_params = get_npv_query("noun", "情報")
    verb_query, verb_params = get_npv_query("verb", "集める")

    assert "WHERE l1.string = ?" in noun_query
    assert noun_params == ["情報"]
    assert "WHERE l3.string = ?" in verb_query
    assert verb_params == ["集める"]
