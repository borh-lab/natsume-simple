import tempfile
from pathlib import Path

import polars as pl
import pytest
import spacy

from natsume_simple.data import GenericCorpusLoader


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_corpus(temp_data_dir):
    """Create a sample corpus with metadata and text files."""
    corpus_dir = temp_data_dir / "test_corpus"
    corpus_dir.mkdir(parents=True)

    # Create metadata.csv
    metadata = {
        "title": ["Test Article 1", "Test Article 2"],
        "year": [2023, 2024],
        "file_path": ["article1.txt", "article2.txt"],
        "author": ["Author 1", "Author 2"],
        "publisher": ["Publisher 1", "Publisher 2"],
        "url": ["http://example1.com", "http://example2.com"],
    }
    df = pl.DataFrame(metadata)
    df.write_csv(corpus_dir / "metadata.csv")

    # Create text files
    (corpus_dir / "article1.txt").write_text(
        "これはテスト文章です。\nこれは二番目の文です。", encoding="utf-8"
    )
    (corpus_dir / "article2.txt").write_text(
        "これは別の記事です。\nもう一つの文章です。", encoding="utf-8"
    )

    return corpus_dir


def test_model_loading():
    try:
        nlp = spacy.load("ja_ginza_electra")
    except Exception:
        nlp = spacy.load("ja_ginza")

    doc = nlp("これはテストです")
    assert len(doc) == 4
    assert [t.text for t in doc] == ["これ", "は", "テスト", "です"]


def test_generic_corpus_loader(temp_data_dir, sample_corpus):
    """Test GenericCorpusLoader functionality."""
    loader = GenericCorpusLoader(data_dir=temp_data_dir, corpus_name="test")

    # Test that corpus directory is set correctly
    assert loader.corpus_dir == sample_corpus

    # Test metadata loading
    entries = list(loader.load_metadata())
    assert len(entries) == 2

    # Test first entry
    entry1 = entries[0]
    assert entry1.corpus == "test"
    assert entry1.title == "Test Article 1"
    assert entry1.year == 2023
    assert entry1.author == "Author 1"
    assert entry1.publisher == "Publisher 1"
    assert entry1.url == "http://example1.com"
    assert len(entry1.sentences) > 0
    assert "これはテスト文章です" in entry1.sentences[0]

    # Test second entry
    entry2 = entries[1]
    assert entry2.corpus == "test"
    assert entry2.title == "Test Article 2"
    assert entry2.year == 2024
    assert entry2.author == "Author 2"
    assert entry2.publisher == "Publisher 2"
    assert entry2.url == "http://example2.com"
    assert len(entry2.sentences) > 0
    assert "これは別の記事です" in entry2.sentences[0]


def test_generic_corpus_loader_missing_metadata(temp_data_dir):
    """Test GenericCorpusLoader behavior with missing metadata."""
    loader = GenericCorpusLoader(data_dir=temp_data_dir, corpus_name="missing")

    # Should not raise an error, but should log a warning
    entries = list(loader.load_metadata())
    assert len(entries) == 0


def test_generic_corpus_loader_invalid_metadata(temp_data_dir):
    """Test GenericCorpusLoader with invalid metadata."""
    corpus_dir = temp_data_dir / "invalid_corpus"
    corpus_dir.mkdir(parents=True)

    # Create invalid metadata.csv (missing required fields)
    invalid_metadata = {
        "title": ["Test Article 1"],
        # Missing year and file_path
        "author": ["Author 1"],
    }
    df = pl.DataFrame(invalid_metadata)
    df.write_csv(corpus_dir / "metadata.csv")

    loader = GenericCorpusLoader(data_dir=temp_data_dir, corpus_name="invalid")

    # Should raise a KeyError due to missing required fields
    with pytest.raises(KeyError):
        list(loader.load_metadata())
