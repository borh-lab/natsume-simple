import hashlib
import html
import io
import logging
import re
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

logger = logging.getLogger(__name__)
SOURCE_CONTENT_HASH = "natsume-source-content-v1"
TED_TRAINING_MEMBER = "ja-en/train.tags.ja-en.ja"
TED_METADATA = re.compile(r"^<([a-z]+)>(.*)</\1>$")
TED_DOCUMENT_START = re.compile(r"^<doc(?:\s[^>]*)?>$")


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
                content_sha256=source_content_sha256((text,)),
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
                content_sha256=source_content_sha256((text,)),
            )
        )

    return AdaptationResult(
        "wiki",
        tuple(sorted(documents, key=lambda document: document.external_id)),
        dict(sorted(rejections.items())),
    )


def adapt_ted_iwslt_archive(archive_path: Path) -> AdaptationResult:
    """Adapt the locked IWSLT Japanese training member into TED talks."""
    documents: list[SourceDocument] = []
    rejections: Counter[str] = Counter()
    seen_talk_ids: set[str] = set()
    metadata: dict[str, str] | None = None
    text_units: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if member.filename == TED_TRAINING_MEMBER
        ]
        if len(members) != 1:
            raise ValueError("ted_archive_member_invalid")

        with archive.open(members[0]) as raw_stream:
            stream = io.TextIOWrapper(raw_stream, encoding="utf-8")
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                if TED_DOCUMENT_START.fullmatch(line):
                    if metadata is not None:
                        raise ValueError("ted_archive_structure_invalid")
                    metadata = {}
                    text_units = []
                    continue
                if line == "</doc>":
                    if metadata is None:
                        raise ValueError("ted_archive_structure_invalid")
                    talk_id = metadata.get("talkid")
                    if not talk_id:
                        raise ValueError("ted_identity_missing")
                    if talk_id in seen_talk_ids:
                        raise ValueError("ted_identity_duplicate")
                    seen_talk_ids.add(talk_id)
                    units = tuple(text_units)
                    if not units:
                        rejections["empty_subtitle_text"] += 1
                    else:
                        documents.append(
                            SourceDocument(
                                corpus_id="ted",
                                external_id=talk_id,
                                title=metadata.get("title", f"TED Talk {talk_id}"),
                                year=None,
                                author=_optional_text(metadata.get("speaker")),
                                publisher="TED Conference LLC",
                                url=_optional_text(metadata.get("url")),
                                text_units=units,
                                content_sha256=source_content_sha256(units),
                            )
                        )
                    metadata = None
                    text_units = []
                    continue

                match = TED_METADATA.fullmatch(line)
                if match:
                    if metadata is None:
                        continue
                    key, raw_value = match.groups()
                    if key in {"talkid", "title", "speaker", "url"}:
                        value = html.unescape(raw_value.strip())
                        previous = metadata.get(key)
                        if previous is not None and previous != value:
                            raise ValueError("ted_metadata_conflict")
                        metadata[key] = value
                    continue
                if line.startswith("<") and line.endswith(">"):
                    continue
                if metadata is None:
                    raise ValueError("ted_archive_structure_invalid")
                text_units.append(html.unescape(line))

    if metadata is not None:
        raise ValueError("ted_archive_structure_invalid")
    return AdaptationResult(
        "ted",
        tuple(sorted(documents, key=lambda document: document.external_id)),
        dict(sorted(rejections.items())),
    )


def segment_documents(
    documents: Iterable[SourceDocument],
    split: Callable[[tuple[str, ...]], Iterable[str]],
) -> tuple[SentenceRecord, ...]:
    """Split source text into stable, source-ordered sentence records."""
    ordered_documents = tuple(
        sorted(documents, key=lambda item: (item.corpus_id, item.external_id))
    )
    sentences: list[SentenceRecord] = []
    for document_count, document in enumerate(ordered_documents, start=1):
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
        if document_count % 50 == 0 or document_count == len(ordered_documents):
            logger.info(
                "segmented documents=%d/%d sentences=%d",
                document_count,
                len(ordered_documents),
                len(sentences),
            )
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
    for parsed_count, sentence in enumerate(sentences, start=1):
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
        if parsed_count % 1_000 == 0:
            logger.info("parsed sentences=%d", parsed_count)
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
    rejection_totals: dict[str, int] = {}
    for adaptation in adaptations:
        total = len(adaptation.documents) + sum(adaptation.rejections.values())
        enforce_rejection_limits(
            adaptation.rejections, total=total, limits=rejection_limits
        )
        documents.extend(adaptation.documents)
        rejection_counts[adaptation.corpus_id] = adaptation.rejections
        rejection_totals[adaptation.corpus_id] = total
        logger.info(
            "adapted corpus=%s accepted=%d rejected=%d",
            adaptation.corpus_id,
            len(adaptation.documents),
            sum(adaptation.rejections.values()),
        )

    sentences = segment_documents(documents, split)
    logger.info("segmented sentences=%d", len(sentences))
    extraction = extract_collocations(sentences, parse, extractor_id=extractor_id)
    enforce_rejection_limits(
        extraction.rejections, total=len(sentences), limits=rejection_limits
    )
    rejection_counts["extraction"] = extraction.rejections
    rejection_totals["extraction"] = len(sentences)
    logger.info("extracted occurrences=%d", len(extraction.occurrences))

    artifact = build_artifact(
        output_directory,
        records=ArtifactRecords(
            corpora=corpora,
            sources=tuple(documents),
            sentences=sentences,
            occurrences=extraction.occurrences,
        ),
        metadata=replace(
            metadata,
            rejection_counts=rejection_counts,
            rejection_limits={
                "maxCount": rejection_limits.max_count,
                "maxFraction": rejection_limits.max_fraction,
            },
            rejection_totals=rejection_totals,
        ),
    )
    logger.info("completed artifact=%s", artifact.name)
    return artifact


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


def source_content_sha256(text_units: tuple[str, ...]) -> str:
    """Hash ordered source text units with unambiguous byte framing."""
    digest = hashlib.sha256(SOURCE_CONTENT_HASH.encode("ascii") + b"\0")
    for text_unit in text_units:
        encoded = text_unit.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()
