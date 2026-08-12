from pathlib import Path

import polars as pl

from natsume_simple.artifact_builder import SourceDocument
from natsume_simple.corpus_pipeline import (
    adapt_jnlp_directory,
    adapt_wikipedia_parquet,
    segment_documents,
)


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
        == "695e4d689ecfc01c37ca9ca8ec1aa386a386cebcde04bfec5fdb9d3daa93747e"
    )


def test_wikipedia_adapter_reads_local_parquet_in_identity_order(tmp_path: Path):
    second = tmp_path / "second.parquet"
    first = tmp_path / "first.parquet"
    pl.DataFrame(
        {
            "id": ["2"],
            "url": ["https://ja.wikipedia.org/wiki/第二"],
            "title": ["第二"],
            "text": ["第三文。"],
        }
    ).write_parquet(second)
    pl.DataFrame(
        {
            "id": ["1"],
            "url": ["https://ja.wikipedia.org/wiki/第一"],
            "title": ["第一"],
            "text": ["第一文。第二文。"],
        }
    ).write_parquet(first)

    result = adapt_wikipedia_parquet([second, first])

    assert result.rejections == {}
    assert [document.external_id for document in result.documents] == ["1", "2"]
    assert result.documents[0].text_units == ("第一文。第二文。",)
    assert (
        result.documents[0].content_sha256
        == "ff2f1c663b5c075a8459ea9e9a82eec69b8872d64bcd93a2f3f9f47784d0cb9e"
    )


def test_segmentation_assigns_stable_ordinals_and_drops_empty_results():
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

    sentences = segment_documents(documents, split)

    assert [
        (sentence.source_identity, sentence.ordinal, sentence.text)
        for sentence in sentences
    ] == [
        (("wiki", "1"), 0, "第一文。"),
        (("wiki", "1"), 1, "第二文。"),
        (("wiki", "2"), 0, "第三文。"),
    ]
