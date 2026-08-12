import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

import polars as pl  # type: ignore
import torch
from pydantic import BaseModel, Field
from wtpsplit import SaT  # type: ignore

logger = logging.getLogger(__name__)


class CorpusEntry(BaseModel):
    """Standard metadata format for all corpus entries."""

    corpus: str
    title: str
    year: int
    author: Optional[str] = None
    publisher: Optional[str] = None
    sentences: List[str]
    url: Optional[str] = None


class BaseCorpusLoader(BaseModel):
    """Base class for all corpus loaders."""

    data_dir: Path
    corpus_dir: Path = Field(default_factory=Path)
    corpus_name: str

    def setup_corpus_dir(self) -> Path:
        """Set up and return the corpus directory."""
        return self.data_dir / f"{self.corpus_name}_corpus"

    def split_into_sentences(self, texts: List[str], splitter: SaT) -> List[str]:
        """Split texts into sentences using wtpsplit.

        Args:
            texts: List of texts to split
            splitter: WTP sentence splitter model

        Returns:
            List of sentences from all input texts
        """
        # First split on newlines and filter empty lines for each text
        paragraphs_per_text = [
            [p.strip() for p in re.split(r"\n+", text) if p.strip()] for text in texts
        ]

        # Flatten paragraphs for batch processing
        all_paragraphs = [p for paragraphs in paragraphs_per_text for p in paragraphs]

        # Process all paragraphs at once with wtpsplit and flatten results
        return [
            sentence.strip()
            for sentences in splitter.split(all_paragraphs)
            for sentence in sentences
            if sentence.strip()
        ]

    def _load_sentences(self, file_paths: List[Path]) -> List[str]:
        """Load and filter sentences from text files.

        Args:
            file_paths: List of paths to text files

        Returns:
            List of sentences from all files
        """
        texts = []
        for txt_path in file_paths:
            full_path = self.corpus_dir / txt_path
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except (UnicodeDecodeError, IOError) as e:
                logger.warning(f"Error loading {full_path}: {e}")
                texts.append("")  # Add empty text to maintain alignment

        # Initialize sentence splitter (do this once and store as class attribute)
        if not hasattr(self, "_splitter"):
            self._splitter = SaT("sat-3l-sm")
            if torch.cuda.is_available():
                self._splitter.half().to("cuda")

        # Split all texts at once and filter Japanese sentences
        all_sentences = self.split_into_sentences(texts, self._splitter)

        # Filter Japanese sentences
        return [sent for sent in all_sentences if is_japanese(sent, min_length=5)]

    def load_metadata(self) -> Iterator[CorpusEntry]:
        """Load metadata from standard metadata.csv if it exists."""
        raise NotImplementedError


def is_japanese(line: str, min_length: int = 200) -> bool:
    """Filter a single line to determine if it is likely Japanese text.

    Args:
        line: The line of text to filter.
        min_length: Minimum length of the line to keep (default: 200).

    Returns:
        True if Japanese characters make up at least 50% of the text.

    Examples:
        >>> is_japanese("これは日本語の文章です。", min_length=5)
        True
        >>> is_japanese("abc", min_length=5)
        False
        >>> is_japanese("This is English text", min_length=5)
        False
        >>> is_japanese("123.456.789", min_length=5)
        False
        >>> is_japanese("日本語とEnglishの混ざった文", min_length=5)  # Mixed but mostly Japanese
        True
        >>> is_japanese("This is mostly English with some 日本語", min_length=5)  # Mixed but mostly English
        False
        >>> is_japanese("テスト", min_length=2)  # Katakana
        True
        >>> is_japanese("ひらがな", min_length=2)  # Hiragana
        True
        >>> is_japanese("漢字", min_length=2)  # Kanji
        True
        >>> is_japanese("！？＆", min_length=2)  # Japanese punctuation
        True
        >>> is_japanese("Ｈｅｌｌｏ", min_length=2)  # Fullwidth romaji
        True
    """
    line = line.strip()

    if not line:
        return False

    def is_japanese_char(c: str) -> bool:
        code = ord(c)
        return any(
            [
                # Hiragana (3040-309F)
                0x3040 <= code <= 0x309F,
                # Katakana (30A0-30FF)
                0x30A0 <= code <= 0x30FF,
                # Kanji (4E00-9FFF)
                0x4E00 <= code <= 0x9FFF,
                # Fullwidth ASCII variants (FF00-FF5E)
                0xFF00 <= code <= 0xFF5E,
                # Japanese punctuation and symbols (3000-303F)
                0x3000 <= code <= 0x303F,
                # Additional CJK symbols and punctuation (31F0-31FF)
                0x31F0 <= code <= 0x31FF,
                # Additional Kanji (3400-4DBF)
                0x3400 <= code <= 0x4DBF,
            ]
        )

    if len(line) < min_length:
        # For short strings, require 100% Japanese characters
        return all(is_japanese_char(c) for c in line)

    # For longer strings, require at least 50% Japanese characters
    japanese_char_count = sum(1 for c in line if is_japanese_char(c))
    return (japanese_char_count / len(line)) >= 0.5


class GenericCorpusLoader(BaseCorpusLoader):
    """Generic loader for any corpus with a metadata.csv file."""

    def __init__(self, data_dir: Path, corpus_name: str):
        super().__init__(data_dir=data_dir, corpus_name=corpus_name)

    def model_post_init(self, _context) -> None:
        self.corpus_dir = self.setup_corpus_dir()
        if not (self.corpus_dir / "metadata.csv").exists():
            logger.warning(
                f"No metadata.csv found in {self.corpus_dir}. "
                "Please ensure it contains:\n"
                "- title: str\n"
                "- year: int\n"
                "- file_path: str\n"
                "Optional:\n"
                "- author: str\n"
                "- publisher: str\n"
                "- url: str"
            )

    def load_metadata(self) -> Iterator[CorpusEntry]:
        """Load metadata from standard metadata.csv if it exists."""
        metadata_path = self.corpus_dir / "metadata.csv"
        if metadata_path.exists():
            df = pl.read_csv(metadata_path)
            for row in df.iter_rows(named=True):
                yield CorpusEntry(
                    corpus=self.corpus_name,
                    title=row["title"],
                    year=row["year"],
                    author=row.get("author"),
                    publisher=row.get("publisher"),
                    sentences=self._load_sentences([Path(row["file_path"])]),
                    url=row.get("url"),
                )
