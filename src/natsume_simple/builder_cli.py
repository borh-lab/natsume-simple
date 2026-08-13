"""Operator commands for the immutable corpus pipeline."""

import argparse
import hashlib
import importlib.metadata
import os
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from natsume_simple.artifact_registry import current_artifact, publish_artifact


def new_artifact_instance_id(
    *, timestamp: str | None = None, random_hex: str | None = None
) -> str:
    """Return one readable, collision-resistant artifact identity."""
    timestamp = timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    random_hex = random_hex or secrets.token_hex(16)
    if len(random_hex) != 32 or any(c not in "0123456789abcdef" for c in random_hex):
        raise ValueError("random_hex must contain 128 lowercase hexadecimal bits")
    return f"{timestamp}-{random_hex}"


def path_sha256(path: Path) -> str:
    """Hash a file, or a directory's ordered relative names and file contents."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for item in sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    ):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="natsume-corpus")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare-jnlp", help="convert one already-local JNLP archive"
    )
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("output", type=Path)

    acquire = commands.add_parser(
        "acquire-release-inputs", help="download and verify the locked release files"
    )
    acquire.add_argument("--source-lock", type=Path, required=True)
    acquire.add_argument("--output-directory", type=Path, required=True)

    build = commands.add_parser(
        "build", help="build one immutable artifact from already-local inputs"
    )
    build.add_argument("--artifacts-directory", type=Path, required=True)
    build.add_argument("--jnlp-root", type=Path)
    build.add_argument("--wikipedia-parquet", type=Path, action="append", default=[])
    build.add_argument("--splitter-model", type=Path, required=True)
    build.add_argument("--content-license", type=Path, required=True)
    build.add_argument("--attribution", type=Path, required=True)
    build.add_argument("--artifact-instance-id")
    build.add_argument("--max-rejections", type=int, default=100)
    build.add_argument("--max-rejection-fraction", type=float, default=0.01)

    publish = commands.add_parser(
        "publish", help="atomically select a validated artifact"
    )
    publish.add_argument("artifact", type=Path)
    publish.add_argument("deploy_directory", type=Path)

    current = commands.add_parser("current", help="print the selected artifact")
    current.add_argument("deploy_directory", type=Path)
    return parser


def _build(args: argparse.Namespace) -> Path:
    if args.jnlp_root is None and not args.wikipedia_parquet:
        raise ValueError("at least one local corpus input is required")
    if not args.splitter_model.exists():
        raise FileNotFoundError(args.splitter_model)

    # Heavy builder dependencies remain outside import and server paths.
    import spacy
    import torch
    from wtpsplit import SaT

    from natsume_simple.artifact_builder import BuildMetadata, CorpusRecord
    from natsume_simple.corpus_pipeline import (
        RejectionLimits,
        adapt_jnlp_directory,
        adapt_wikipedia_parquet,
        build_corpus_artifact,
    )

    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    splitter = SaT(str(args.splitter_model))
    splitter.eval().to("cpu")
    nlp = spacy.load("ja_ginza")

    def split(text_units: tuple[str, ...]):
        paragraphs = [
            paragraph.strip()
            for text in text_units
            for paragraph in text.splitlines()
            if paragraph.strip()
        ]
        return (
            sentence.strip()
            for group in splitter.split(paragraphs)
            for sentence in group
            if sentence.strip()
        )

    adaptations = []
    corpora = []
    if args.jnlp_root is not None:
        adaptations.append(adapt_jnlp_directory(args.jnlp_root))
        corpora.append(CorpusRecord("jnlp", "自然言語処理"))
    if args.wikipedia_parquet:
        adaptations.append(adapt_wikipedia_parquet(args.wikipedia_parquet))
        corpora.append(CorpusRecord("wiki", "日本語版Wikipedia"))

    instance_id = args.artifact_instance_id or new_artifact_instance_id()
    args.artifacts_directory.mkdir(parents=True, exist_ok=True)
    output = args.artifacts_directory / instance_id
    package_version = importlib.metadata.version("natsume-simple")
    spacy_version = importlib.metadata.version("spacy")
    ginza_version = importlib.metadata.version("ja-ginza")
    wtpsplit_version = importlib.metadata.version("wtpsplit")
    extractor_id = f"ja-ginza-{ginza_version}:npv-v1"
    return build_corpus_artifact(
        output,
        corpora=tuple(corpora),
        adaptations=tuple(adaptations),
        split=split,
        parse=nlp,
        metadata=BuildMetadata(
            artifact_instance_id=instance_id,
            identity_inputs={
                "builderRevision": os.environ.get(
                    "NATSUME_BUILDER_REVISION", f"natsume-simple-{package_version}"
                ),
                "sourceAdapters": [adaptation.corpus_id for adaptation in adaptations],
                "sentenceSplitter": {
                    "name": "wtpsplit",
                    "version": wtpsplit_version,
                    "modelSha256": path_sha256(args.splitter_model),
                },
                "nlpModel": {
                    "name": "ja_ginza",
                    "version": ginza_version,
                    "spacyVersion": spacy_version,
                    "extractionPolicy": "npv-v1",
                },
                "executionProfile": {
                    "backend": "cpu",
                    "precision": "float32",
                    "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
                    "torchVersion": torch.__version__,
                    "torchThreads": torch.get_num_threads(),
                    "torchInteropThreads": torch.get_num_interop_threads(),
                },
            },
            built_at=datetime.now(UTC),
            content_license=args.content_license.read_text(encoding="utf-8"),
            attribution=args.attribution.read_text(encoding="utf-8"),
        ),
        rejection_limits=RejectionLimits(
            max_count=args.max_rejections,
            max_fraction=args.max_rejection_fraction,
        ),
        extractor_id=extractor_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run an operator command and return a process exit status."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-jnlp":
            from natsume_simple.corpus_pipeline import prepare_jnlp_archive

            result = prepare_jnlp_archive(args.archive, args.output)
        elif args.command == "acquire-release-inputs":
            from natsume_simple.release_inputs import (
                acquire_release_inputs,
                load_release_sources,
            )

            result = acquire_release_inputs(
                load_release_sources(args.source_lock), args.output_directory
            )
        elif args.command == "build":
            result = _build(args)
        elif args.command == "publish":
            result = publish_artifact(args.artifact, args.deploy_directory)
        else:
            result = current_artifact(args.deploy_directory)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if isinstance(result, tuple):
        for path in result:
            print(path)
        return 0
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
