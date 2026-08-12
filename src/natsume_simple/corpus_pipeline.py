from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path

import polars as pl

from natsume_simple.artifact_builder import SentenceRecord, SourceDocument


@dataclass(frozen=True)
class AdaptationResult:
    documents: tuple[SourceDocument, ...]
    rejections: dict[str, int]


def adapt_jnlp_directory(
    corpus_root: Path, *, metadata_path: Path | None = None
) -> AdaptationResult:
    """Adapt converted JNLP plaintext and its metadata into source documents."""
    metadata = pl.read_excel(metadata_path or corpus_root / "file_DB.xls")
    documents: list[SourceDocument] = []
    rejections: Counter[str] = Counter()

    for row in metadata.iter_rows(named=True):
        raw_path = row["ファイル名"]
        if raw_path == "*NA*":
            rejections["missing_source_path"] += 1
            continue
        source_path = _jnlp_source_path(str(raw_path), int(row["Vol"]))
        plain_text_path = (corpus_root / source_path).with_suffix(".txt")
        if not plain_text_path.is_file():
            rejections["missing_plain_text"] += 1
            continue
        text = plain_text_path.read_text(encoding="utf-8").strip()
        if not text:
            rejections["empty_plain_text"] += 1
            continue
        documents.append(
            SourceDocument(
                corpus_id="jnlp",
                external_id=source_path.as_posix(),
                title=str(row["タイトル"]),
                year=1993 + int(row["Vol"]),
                author=_optional_text(row["著者"]),
                publisher="自然言語処理",
                url=_optional_text(row["J-Stageにおける論文URL"]),
                text_units=(text,),
                content_sha256=_text_sha256(text),
            )
        )

    return AdaptationResult(
        tuple(sorted(documents, key=lambda document: document.external_id)),
        dict(sorted(rejections.items())),
    )


def adapt_wikipedia_parquet(paths: Sequence[Path]) -> AdaptationResult:
    """Adapt local data-only Wikipedia Parquet shards without remote code."""
    documents: list[SourceDocument] = []
    rejections: Counter[str] = Counter()

    for path in sorted(paths):
        for row in pl.read_parquet(
            path, columns=["id", "url", "title", "text"]
        ).iter_rows(named=True):
            external_id = _optional_text(row["id"])
            title = _optional_text(row["title"])
            text = _optional_text(row["text"])
            if external_id is None:
                rejections["missing_external_id"] += 1
                continue
            if title is None:
                rejections["missing_title"] += 1
                continue
            if text is None:
                rejections["empty_text"] += 1
                continue
            documents.append(
                SourceDocument(
                    corpus_id="wiki",
                    external_id=external_id,
                    title=title,
                    year=2023,
                    author="Wikipedia Contributors",
                    publisher="Wikimedia Foundation",
                    url=_optional_text(row["url"]),
                    text_units=(text,),
                    content_sha256=_text_sha256(text),
                )
            )

    return AdaptationResult(
        tuple(sorted(documents, key=lambda document: document.external_id)),
        dict(sorted(rejections.items())),
    )


def segment_documents(
    documents: Iterable[SourceDocument],
    split: Callable[[tuple[str, ...]], Iterable[str]],
) -> tuple[SentenceRecord, ...]:
    """Split source text into stable, source-ordered sentence records."""
    sentences: list[SentenceRecord] = []
    for document in sorted(
        documents, key=lambda item: (item.corpus_id, item.external_id)
    ):
        ordinal = 0
        for text in split(document.text_units):
            if not text:
                continue
            sentences.append(
                SentenceRecord(
                    (document.corpus_id, document.external_id), ordinal, text
                )
            )
            ordinal += 1
    return tuple(sentences)


def _jnlp_source_path(raw_path: str, volume: int) -> Path:
    path = Path(raw_path)
    if len(path.parts) == 1:
        path = Path(f"V{volume:02d}") / path
    return path


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
