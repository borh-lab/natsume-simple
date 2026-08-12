import pytest
import spacy

from natsume_simple.pattern_extraction import normalize_verb_span, npv_matcher


@pytest.fixture(scope="module")
def japanese_pipeline():
    return spacy.load("ja_ginza")


@pytest.mark.nlp_model
@pytest.mark.parametrize(
    ("source", "expected"),
    [
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
    ],
)
def test_normalization_contract(japanese_pipeline, source, expected):
    suru_token = japanese_pipeline("する")[0]
    assert normalize_verb_span(japanese_pipeline(source), suru_token) == expected


@pytest.mark.nlp_model
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "東京では，銀座でランチをたべよう。",
            [("銀座", "で", "食べる"), ("ランチ", "を", "食べる")],
        ),
        ("京都にも行く。", []),
        ("ことを説明するならば", [("こと", "を", "説明する")]),
        ("ことにならない", [("こと", "に", "ならない")]),
    ],
)
def test_extraction_contract(japanese_pipeline, source, expected):
    suru_token = japanese_pipeline("する")[0]
    matches = npv_matcher(japanese_pipeline(source), suru_token)
    assert [(noun, particle, verb) for noun, particle, verb, *_ in matches] == expected
