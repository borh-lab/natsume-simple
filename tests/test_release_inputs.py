import hashlib
import io
import json
from pathlib import Path

import pytest

from natsume_simple.release_inputs import (
    ReleaseInputError,
    acquire_locked_file,
    acquire_release_inputs,
    canonical_article_ids_sha256,
    load_release_sources,
    load_wikipedia_subset,
    validate_wikipedia_paths,
    verify_file,
)

SOURCE_LOCK = Path(__file__).parents[1] / "docs" / "corpus-sources.lock.json"
WIKIPEDIA_SUBSET = (
    Path(__file__).parents[1] / "docs" / "wikipedia-ja-20231101-subset.json"
)


def write_source_lock(path: Path, article_ids: list[str]) -> Path:
    identity_sha256 = hashlib.sha256(
        json.dumps(article_ids, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "corpusId": "jnlp",
                        "status": "ready",
                        "candidate": {
                            "observedRelease": "2026-06-15",
                            "url": "https://example.test/jnlp.zip",
                            "size": 4,
                            "sha256": hashlib.sha256(b"jnlp").hexdigest(),
                        },
                    },
                    {
                        "corpusId": "wikipedia-ja-20231101",
                        "status": "ready",
                        "candidate": {
                            "urlTemplate": "https://example.test/{name}",
                            "verification": {
                                "verifiedShard": "train-00000-of-00015.parquet",
                                "orderedIdentityListSha256": identity_sha256,
                            },
                            "files": [
                                {
                                    "name": "train-00000-of-00015.parquet",
                                    "size": 4,
                                    "sha256": hashlib.sha256(b"wiki").hexdigest(),
                                },
                                {
                                    "name": "train-00001-of-00015.parquet",
                                    "size": 5,
                                    "sha256": "f" * 64,
                                },
                            ],
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def write_subset(
    path: Path,
    article_ids: list[object],
    *,
    source_lock_corpus_id: str = "wikipedia-ja-20231101",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "sourceLockCorpusId": source_lock_corpus_id,
                "articleIds": article_ids,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_only_the_locked_production_files(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    )

    assert sources.jnlp_archive.local_name == "NLP_LATEX_CORPUS-2026-06-15.zip"
    assert sources.jnlp_archive.url == "https://example.test/jnlp.zip"
    assert sources.jnlp_archive.size == 4
    assert sources.wikipedia_shard.name == "train-00000-of-00015.parquet"
    assert sources.wikipedia_shard.url.endswith("train-00000-of-00015.parquet")
    assert sources.wikipedia_identity_sha256 == canonical_article_ids_sha256(
        article_ids
    )


def test_repository_lock_selects_the_planned_release_files():
    sources = load_release_sources(SOURCE_LOCK)

    assert sources.jnlp_archive.local_name == "NLP_LATEX_CORPUS-2026-06-15.zip"
    assert sources.jnlp_archive.size == 19_166_717
    assert (
        sources.jnlp_archive.sha256
        == "8610f8c391634de11a816950008d63675c52e940c6c0c7df29d7ded005547fdb"
    )
    assert sources.wikipedia_shard.name == "train-00000-of-00015.parquet"
    assert sources.wikipedia_shard.size == 611_504_422
    assert (
        sources.wikipedia_shard.sha256
        == "4751c14478e712fd637bd83c2cf3537b0e299ea5115e9a78ddededf42f34c29d"
    )


def test_repository_subset_matches_the_locked_identity():
    sources = load_release_sources(SOURCE_LOCK)

    subset = load_wikipedia_subset(WIKIPEDIA_SUBSET, sources=sources)

    assert len(subset.article_ids) == len(set(subset.article_ids)) == 971
    assert canonical_article_ids_sha256(subset.article_ids) == (
        "249dc639f646da4db3711571ea97231a22421c22a5a90ebedf86e7ee3b471991"
    )


def test_article_identity_uses_compact_utf8_json():
    article_ids = ["1", "日本語"]

    assert (
        canonical_article_ids_sha256(article_ids)
        == hashlib.sha256(b'["1","\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"]').hexdigest()
    )


def test_loads_a_unique_971_article_subset(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    )

    subset = load_wikipedia_subset(
        write_subset(tmp_path / "subset.json", article_ids), sources=sources
    )

    assert subset.source_lock_corpus_id == "wikipedia-ja-20231101"
    assert subset.article_ids == tuple(article_ids)


@pytest.mark.parametrize(
    ("article_ids", "source_lock_corpus_id", "reason"),
    [
        (
            [str(index) for index in range(970)],
            "wikipedia-ja-20231101",
            "subset_count_invalid",
        ),
        (["same"] * 971, "wikipedia-ja-20231101", "subset_identity_duplicate"),
        (
            [*map(str, range(970)), 970],
            "wikipedia-ja-20231101",
            "subset_identity_invalid",
        ),
        ([str(index) for index in range(971)], "unknown", "subset_source_mismatch"),
    ],
)
def test_rejects_invalid_subset_shape(
    tmp_path: Path,
    article_ids: list[object],
    source_lock_corpus_id: str,
    reason: str,
):
    expected_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", expected_ids)
    )

    with pytest.raises(ReleaseInputError, match=reason):
        load_wikipedia_subset(
            write_subset(
                tmp_path / "subset.json",
                article_ids,
                source_lock_corpus_id=source_lock_corpus_id,
            ),
            sources=sources,
        )


def test_rejects_subset_checksum_mismatch(tmp_path: Path):
    expected_ids = [str(index) for index in range(971)]
    actual_ids = [*expected_ids[:-1], "different"]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", expected_ids)
    )

    with pytest.raises(ReleaseInputError, match="subset_checksum_mismatch"):
        load_wikipedia_subset(
            write_subset(tmp_path / "subset.json", actual_ids), sources=sources
        )


def test_rejects_unknown_or_nonready_source_entries(tmp_path: Path):
    source_lock = tmp_path / "lock.json"
    source_lock.write_text(json.dumps({"sources": []}), encoding="utf-8")

    with pytest.raises(ReleaseInputError, match="source_lock_entry_missing"):
        load_release_sources(source_lock)


def test_accepts_only_the_locked_wikipedia_shard_path(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    )
    selected = tmp_path / "train-00000-of-00015.parquet"

    assert validate_wikipedia_paths([selected], sources=sources) == selected
    for paths in [
        [],
        [selected, tmp_path / "extra.parquet"],
        [tmp_path / "train-00001-of-00015.parquet"],
    ]:
        with pytest.raises(ReleaseInputError, match="wikipedia_shard_invalid"):
            validate_wikipedia_paths(paths, sources=sources)


def test_verifies_locked_file_size_and_checksum(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    )
    archive = tmp_path / sources.jnlp_archive.local_name
    archive.write_bytes(b"jnlp")

    verify_file(archive, sources.jnlp_archive)
    archive.write_bytes(b"nope")
    with pytest.raises(ReleaseInputError, match="source_checksum_mismatch"):
        verify_file(archive, sources.jnlp_archive)


def test_acquires_a_locked_file_before_exposing_its_final_name(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    locked = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    ).jnlp_archive
    calls: list[tuple[str, int]] = []

    class ObservedStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            assert not (tmp_path / locked.local_name).exists()
            assert list(tmp_path.glob(f"{locked.local_name}.part-*"))
            return super().read(size)

    def opener(url: str, *, timeout: int):
        calls.append((url, timeout))
        return ObservedStream(b"jnlp")

    destination = acquire_locked_file(locked, tmp_path, opener=opener)

    assert destination == tmp_path / locked.local_name
    assert destination.read_bytes() == b"jnlp"
    assert calls == [(locked.url, 60)]
    assert not list(tmp_path.glob(f"{locked.local_name}.part-*"))


def test_reuses_a_valid_acquired_file_without_network(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    locked = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    ).jnlp_archive
    destination = tmp_path / locked.local_name
    destination.write_bytes(b"jnlp")

    assert (
        acquire_locked_file(
            locked,
            tmp_path,
            opener=lambda *_args, **_kwargs: pytest.fail("network opened"),
        )
        == destination
    )


def test_rejects_an_invalid_existing_destination_without_overwriting(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    locked = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    ).jnlp_archive
    destination = tmp_path / locked.local_name
    destination.write_bytes(b"bad")

    with pytest.raises(ReleaseInputError, match="source_size_mismatch"):
        acquire_locked_file(
            locked,
            tmp_path,
            opener=lambda *_args, **_kwargs: pytest.fail("network opened"),
        )

    assert destination.read_bytes() == b"bad"


def test_removes_a_download_that_fails_verification(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    locked = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    ).jnlp_archive

    with pytest.raises(ReleaseInputError, match="source_size_mismatch"):
        acquire_locked_file(
            locked,
            tmp_path,
            opener=lambda *_args, **_kwargs: io.BytesIO(b"bad"),
        )

    assert not (tmp_path / locked.local_name).exists()
    assert not list(tmp_path.glob(f"{locked.local_name}.part-*"))


def test_acquires_both_release_inputs(tmp_path: Path):
    article_ids = [str(index) for index in range(971)]
    sources = load_release_sources(
        write_source_lock(tmp_path / "lock.json", article_ids)
    )
    payloads = {sources.jnlp_archive.url: b"jnlp", sources.wikipedia_shard.url: b"wiki"}

    acquired = acquire_release_inputs(
        sources,
        tmp_path / "inputs",
        opener=lambda url, **_kwargs: io.BytesIO(payloads[url]),
    )

    assert acquired == (
        tmp_path / "inputs" / sources.jnlp_archive.local_name,
        tmp_path / "inputs" / sources.wikipedia_shard.local_name,
    )
