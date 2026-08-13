import hashlib
import subprocess
import zipfile
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
from spacy.tokens import Doc

from natsume_simple.artifact_builder import (
    ArtifactRecords,
    BuildMetadata,
    CollocationOccurrence,
    CorpusRecord,
    SentenceRecord,
    SourceDocument,
    build_artifact,
)
from natsume_simple.pattern_extraction import npv_matcher


class PipelineRejected(ValueError):
    """A rejection threshold was crossed for one bounded reason."""


@dataclass(frozen=True)
class RejectionLimits:
    max_count: int
    max_fraction: float


@dataclass(frozen=True)
class AdaptationResult:
    corpus_id: str
    documents: tuple[SourceDocument, ...]
    rejections: dict[str, int]


@dataclass(frozen=True)
class ExtractionResult:
    occurrences: tuple[CollocationOccurrence, ...]
    rejections: dict[str, int]


def prepare_jnlp_archive(archive: Path, output_directory: Path) -> Path:
    """Extract a local JNLP archive and convert each LaTeX source to plain text."""
    if output_directory.exists():
        raise FileExistsError(output_directory)

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
        output_directory.mkdir(parents=True)
        bundle.extractall(output_directory)

    metadata_files = sorted(output_directory.rglob("file_DB.xls"))
    if len(metadata_files) != 1:
        raise ValueError("JNLP archive must contain exactly one file_DB.xls")
    corpus_root = metadata_files[0].parent

    for source in sorted(corpus_root.rglob("*.tex")):
        subprocess.run(
            ["nkf", "-w", "--overwrite", "--in-place", str(source)], check=True
        )
        try:
            subprocess.run(
                [
                    "pandoc",
                    "--quiet",
                    "--from",
                    "latex+east_asian_line_breaks",
                    "--to",
                    "plain",
                    "--wrap=none",
                    "--strip-comments",
                    "-N",
                    "-s",
                    str(source),
                    "-o",
                    str(source.with_suffix(".txt")),
                ],
                check=True,
            )
        except subprocess.CalledProcessError:
            # The adapter records the missing plaintext with a bounded reason count.
            continue
    return corpus_root


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
        "jnlp",
        tuple(sorted(documents, key=lambda document: document.external_id)),
        dict(sorted(rejections.items())),
    )


def adapt_wikipedia_parquet(
    path: Path, *, article_ids: Collection[str]
) -> AdaptationResult:
    """Adapt exactly the selected articles from one local Parquet shard."""
    documents: list[SourceDocument] = []
    rejections: Counter[str] = Counter()

    requested = set(article_ids)
    selected = (
        pl.scan_parquet(path)
        .select("id", "url", "title", "text")
        .filter(pl.col("id").is_in(list(requested)))
        .collect()
    )
    selected_ids = [str(value) for value in selected["id"].to_list()]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("wikipedia_identity_duplicate")
    if set(selected_ids) != requested:
        raise ValueError("wikipedia_identity_missing")

    for row in selected.iter_rows(named=True):
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
        "wiki",
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


def extract_collocations(
    sentences: Iterable[SentenceRecord],
    parse: Callable[[str], Doc],
    *,
    extractor_id: str,
) -> ExtractionResult:
    """Parse canonical sentences into identity-preserving occurrences."""
    occurrences: list[CollocationOccurrence] = []
    rejections: Counter[str] = Counter()
    for sentence in sentences:
        doc = parse(sentence.text)
        if not doc.has_annotation("DEP"):
            rejections["missing_dependency_parse"] += 1
            continue
        for noun, particle, verb, *spans in npv_matcher(doc):
            occurrences.append(
                CollocationOccurrence(
                    source_identity=sentence.source_identity,
                    sentence_ordinal=sentence.ordinal,
                    noun=noun,
                    particle=particle,
                    verb=verb,
                    noun_span=(spans[0], spans[1]),
                    particle_span=(spans[2], spans[3]),
                    verb_span=(spans[4], spans[5]),
                    extractor_id=extractor_id,
                )
            )
    return ExtractionResult(tuple(occurrences), dict(sorted(rejections.items())))


def enforce_rejection_limits(
    rejections: dict[str, int],
    *,
    total: int,
    limits: RejectionLimits,
) -> None:
    """Reject a pipeline stage when any bounded reason crosses its limits."""
    if total <= 0:
        raise PipelineRejected("rejection_total_invalid")
    for reason, count in sorted(rejections.items()):
        if count > limits.max_count or count / total > limits.max_fraction:
            raise PipelineRejected(f"rejection_limit_exceeded:{reason}")


def build_corpus_artifact(
    output_directory: Path,
    *,
    corpora: tuple[CorpusRecord, ...],
    adaptations: tuple[AdaptationResult, ...],
    split: Callable[[tuple[str, ...]], Iterable[str]],
    parse: Callable[[str], Doc],
    metadata: BuildMetadata,
    rejection_limits: RejectionLimits,
    extractor_id: str,
) -> Path:
    """Compose adapted sources, segmentation, extraction, and persistence."""
    documents: list[SourceDocument] = []
    rejection_counts: dict[str, dict[str, int]] = {}
    for adaptation in adaptations:
        total = len(adaptation.documents) + sum(adaptation.rejections.values())
        enforce_rejection_limits(
            adaptation.rejections, total=total, limits=rejection_limits
        )
        documents.extend(adaptation.documents)
        rejection_counts[adaptation.corpus_id] = adaptation.rejections

    sentences = segment_documents(documents, split)
    extraction = extract_collocations(sentences, parse, extractor_id=extractor_id)
    enforce_rejection_limits(
        extraction.rejections, total=len(sentences), limits=rejection_limits
    )
    rejection_counts["extraction"] = extraction.rejections

    return build_artifact(
        output_directory,
        records=ArtifactRecords(
            corpora=corpora,
            sources=tuple(documents),
            sentences=sentences,
            occurrences=extraction.occurrences,
        ),
        metadata=replace(metadata, rejection_counts=rejection_counts),
    )


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
