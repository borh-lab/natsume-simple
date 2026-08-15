from collections.abc import Iterable

from spacy.language import Language

REPRESENTATIVE_SENTENCES = (
    "情報を集めて判断する。",
    "研究者が日本語の文章を詳しく分析した。",
    "時間について考えながら結果を説明する。",
    "新しい方法で課題を解決できるか検証した。",
    "利用者は複数の資料から必要な事実を探す。",
    "自然言語処理の技術が社会で広く使われている。",
    "性能だけでなく解析結果の一致も確認する。",
    "明日の会議までに報告書を作成してください。",
)

TokenParse = tuple[str, str, str, str, int]
DocumentParse = tuple[TokenParse, ...]


def representative_parse(nlp: Language) -> tuple[DocumentParse, ...]:
    documents: Iterable = nlp.pipe(REPRESENTATIVE_SENTENCES, batch_size=8)
    return tuple(
        tuple(
            (token.text, token.lemma_, token.pos_, token.dep_, token.head.i)
            for token in document
        )
        for document in documents
    )


def assert_representative_parse(parses: tuple[DocumentParse, ...]) -> None:
    assert len(parses) == len(REPRESENTATIVE_SENTENCES)
    assert tuple(token[0] for token in parses[0]) == (
        "情報",
        "を",
        "集め",
        "て",
        "判断",
        "する",
        "。",
    )
    for document in parses:
        assert document
        assert sum(token[3] == "ROOT" for token in document) == 1
        assert all(token[1] and token[2] and token[3] for token in document)
