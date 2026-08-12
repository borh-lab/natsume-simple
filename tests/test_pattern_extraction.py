import pytest
import spacy

from natsume_simple.pattern_extraction import normalize_verb_span, npv_matcher
from tests.token_observations import (
    COMPOUND_PARTICLE_OBSERVATIONS,
    EXTRACTION_OBSERVATIONS,
    NORMALIZATION_OBSERVATIONS,
    observed_doc,
)

NORMALIZATION_CASES = [
    ("飛び立つでしょう", ("飛び立つ", 0, 4)),
    ("考えられませんでした", ("考えられる", 0, 4)),
    ("扱うかです", ("扱う", 0, 2)),
    ("突入しちゃう", ("突入する", 0, 6)),
    ("で囲んである", ("囲む", 1, 6)),
    ("たらしめている", ("たらしめる", 0, 7)),
    ("いるからで", ("いる", 0, 2)),
    ("いるという", ("いる", 0, 2)),
    ("語ります", ("語る", 0, 2)),
    ("しました。", ("する", 0, 1)),
    ("作り上げたか", ("作り上げる", 0, 4)),
    ("見られなかったが", ("見られない", 0, 6)),
]

EXTRACTION_CASES = [
    (
        "東京では，銀座でランチをたべよう。",
        [("銀座", "で", "食べる"), ("ランチ", "を", "食べる")],
    ),
    ("京都にも行く。", []),
    ("ことを説明するならば", [("こと", "を", "説明する")]),
    ("ことにならない", [("こと", "に", "ならない")]),
]


@pytest.fixture(scope="module")
def japanese_pipeline():
    return spacy.load("ja_ginza")


@pytest.mark.nlp_model
@pytest.mark.parametrize(
    ("source", "expected"),
    NORMALIZATION_CASES,
)
def test_normalization_contract(japanese_pipeline, source, expected):
    assert normalize_verb_span(japanese_pipeline(source)) == expected


@pytest.mark.parametrize(("source", "expected"), NORMALIZATION_CASES)
def test_normalization_policy(source, expected):
    assert (
        normalize_verb_span(observed_doc(NORMALIZATION_OBSERVATIONS[source]))
        == expected
    )


@pytest.mark.nlp_model
@pytest.mark.parametrize(
    ("source", "expected"),
    EXTRACTION_CASES,
)
def test_extraction_contract(japanese_pipeline, source, expected):
    matches = npv_matcher(japanese_pipeline(source))
    assert [(noun, particle, verb) for noun, particle, verb, *_ in matches] == expected


@pytest.mark.parametrize(("source", "expected"), EXTRACTION_CASES)
def test_extraction_policy(monkeypatch, source, expected):
    doc = observed_doc(EXTRACTION_OBSERVATIONS[source])
    monkeypatch.setattr(
        "natsume_simple.pattern_extraction.ginza.bunsetu_span",
        lambda token: token.doc[token.i :],
    )
    matches = npv_matcher(doc)
    assert [(noun, particle, verb) for noun, particle, verb, *_ in matches] == expected


@pytest.mark.parametrize("source", COMPOUND_PARTICLE_OBSERVATIONS)
def test_compound_particles_are_excluded(source):
    assert npv_matcher(observed_doc(COMPOUND_PARTICLE_OBSERVATIONS[source])) == []
