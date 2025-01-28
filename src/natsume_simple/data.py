import argparse
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import datasets  # type: ignore
import polars as pl  # type: ignore
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from wtpsplit import SaT  # type: ignore

from natsume_simple.database import (
    bulk_insert_sentences,
    init_database,
    insert_sources_batch,
)
from natsume_simple.log import setup_logger
from natsume_simple.utils import set_random_seed

logger = setup_logger(__name__)


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

    def download(self) -> None:
        """Download corpus data if needed."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(
        self,
    ) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], List[str]]]:
        """Prepare batch of sources and sentences for database insertion.

        Returns:
            Tuple containing:
            - List of source dictionaries (corpus, title, year, author, publisher)
            - Dictionary mapping (corpus, title) to list of sentences
        """
        sources = []
        source_map: Dict[Tuple[str, str], List[str]] = {}

        for entry in self.load_metadata():
            sources.append(
                {
                    "corpus": entry.corpus,
                    "title": entry.title,
                    "year": entry.year,
                    "author": entry.author,
                    "publisher": entry.publisher,
                }
            )
            # Group sentences by source
            key = (entry.corpus, entry.title)
            if key not in source_map:
                source_map[key] = []
            source_map[key].extend(entry.sentences)

        return sources, source_map

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


def filter_non_japanese(dir: Path, min_length: int = 200) -> Iterator[str]:
    """Filter out non-Japanese text from converted files.

    This function reads text files and filters out lines that are likely not Japanese text.

    Args:
        dir: Path to directory containing text files to filter
        min_length: Minimum length of lines to keep (default: 200)

    Yields:
        Lines of text that pass the Japanese text filters, stripped of whitespace on the end
    """
    files = dir.rglob("*.txt")
    for file in files:
        with open(file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if is_japanese(line, min_length):
                    yield line


class JNLPCorpusLoader(BaseCorpusLoader):
    corpus_name: str = "jnlp"

    def model_post_init(self, _context) -> None:
        base_dir = self.setup_corpus_dir()
        self.corpus_dir = base_dir / "NLP_LATEX_CORPUS"

    def download(self) -> None:
        """Download the JNLP corpus."""
        file = self.data_dir / "NLP_LATEX_CORPUS.zip"
        if not file.is_file():
            logger.info("Downloading JNLP corpus...")
            urllib.request.urlretrieve(
                "https://www.anlp.jp/resource/journal_latex/NLP_LATEX_CORPUS.zip",
                file,
            )

        # Extract to the base corpus directory
        base_dir = self.setup_corpus_dir()
        with zipfile.ZipFile(file, "r") as z:
            z.extractall(base_dir)

    def load_metadata(self) -> Iterator[CorpusEntry]:
        """Load JNLP-specific metadata."""
        xls_path = self.corpus_dir / "file_DB.xls"
        if not xls_path.exists():
            logger.error(f"JNLP metadata Excel not found at {xls_path}")
            return

        df = pl.read_excel(xls_path)
        volume_year_map = dict(zip(range(1, 31), range(1994, 2024)))

        # Group files by batch for processing
        batch_size = 100  # Adjust based on memory constraints
        file_batches = [
            list(batch) for batch in zip(*[iter(df.iter_rows(named=True))] * batch_size)
        ]

        for batch in file_batches:
            # Process batch of files
            valid_paths = []
            batch_map = {}  # Map file paths to their metadata
            for row in batch:
                if row["ファイル名"] == "*NA*":
                    continue
                if "/" not in row["ファイル名"]:
                    # Fix missing directory in some file paths
                    fixed_path = f"V{row['Vol']}/{row['ファイル名']}"
                    logger.info(
                        f"Fixing file path for {row['ファイル名']} -> {fixed_path}"
                    )
                    row["ファイル名"] = fixed_path

                # Use relative path for tex_path
                tex_path = Path(
                    row["ファイル名"]
                )  # Don't combine with self.corpus_dir yet
                full_tex_path = self.corpus_dir / tex_path
                if not full_tex_path.exists():
                    logger.warning(f"File not found: {full_tex_path}, skipping...")
                    continue
                txt_path = tex_path.with_suffix(".txt")
                full_txt_path = self.corpus_dir / txt_path
                if not full_txt_path.exists():
                    if not self._convert_latex_file(full_tex_path, full_txt_path):
                        continue

                valid_paths.append(txt_path)  # Store relative path
                batch_map[txt_path] = row

            # Load all sentences for the batch
            if valid_paths:
                sentences = self._load_sentences(valid_paths)
                if sentences:  # Only yield if we have sentences
                    # Take first valid row for metadata since we can't map sentences back
                    row = batch_map[valid_paths[0]]
                    year = volume_year_map.get(row["Vol"])
                    if year:
                        yield CorpusEntry(
                            corpus=self.corpus_name,
                            title=row["タイトル"],
                            year=year,
                            author=row["著者"] if row["著者"] else "Unknown",
                            publisher="自然言語処理",
                            sentences=sentences,
                            url=row["J-Stageにおける論文URL"] or "",
                        )

    def _convert_latex_file(self, tex_path: Path, txt_path: Path) -> bool:
        """Convert LaTeX to text."""
        logger.info(f"Converting {tex_path} => {txt_path}")
        if not convert_encoding(tex_path):
            return False
        if not convert_latex_to_plaintext(tex_path, txt_path):
            return False
        return True


def convert_encoding(file_path: Path) -> bool:
    """Convert file encoding to UTF-8 using nkf."""
    try:
        subprocess.run(
            ["nkf", "-w", "--overwrite", "--in-place", str(file_path)], check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Encoding conversion failed for {file_path}: {e}")
        return False


def convert_latex_to_plaintext(source: Path, dest: Path) -> bool:
    """Convert LaTeX file to plain text using pandoc."""
    try:
        subprocess.run(
            [
                "pandoc",
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
                str(dest),
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Pandoc conversion failed for {source}: {e}")
        return False


class WikipediaCorpusLoader(BaseCorpusLoader):
    corpus_name: str = "wiki"

    def model_post_init(self, _context) -> None:
        self.corpus_dir = self.setup_corpus_dir()

    def load_metadata(self) -> Iterator[CorpusEntry]:
        """Load Wikipedia-specific metadata."""
        logger.info("Downloading Wikipedia corpus...")
        ds = datasets.load_dataset(
            "wikimedia/wikipedia", "20231101.ja", streaming=True, split="train"
        )

        # Process articles in batches for efficient sentence splitting
        batch_size = 10  # Process 10 articles at a time
        current_batch: List[Tuple[str, str]] = []  # (title, text) pairs

        # Initialize sentence splitter once
        if not hasattr(self, "_splitter"):
            self._splitter = SaT("sat-3l-sm")
            if torch.cuda.is_available():
                self._splitter.half().to("cuda")

        for article in tqdm(ds.take(500), desc="Loading Wikipedia articles", total=500):
            title = article["title"]
            text = article["text"]
            if is_japanese(text):  # Basic filter before more expensive processing
                current_batch.append((title, text))

                if len(current_batch) >= batch_size:
                    # Process batch
                    texts = [text for _, text in current_batch]
                    sentences = self.split_into_sentences(texts, self._splitter)

                    # Filter Japanese sentences
                    sentences = [
                        sent for sent in sentences if is_japanese(sent, min_length=5)
                    ]

                    # Yield entries with sentences
                    if sentences:
                        yield CorpusEntry(
                            corpus=self.corpus_name,
                            title=current_batch[0][0],  # Use first article's title
                            year=2023,
                            author="Wikipedia Contributors",
                            publisher="Wikimedia Foundation",
                            sentences=sentences,
                        )

                    current_batch = []

        # Process any remaining articles
        if current_batch:
            texts = [text for _, text in current_batch]
            sentences = self.split_into_sentences(texts, self._splitter)
            sentences = [sent for sent in sentences if is_japanese(sent, min_length=5)]

            if sentences:
                yield CorpusEntry(
                    corpus=self.corpus_name,
                    title=current_batch[0][0],
                    year=2023,
                    author="Wikipedia Contributors",
                    publisher="Wikimedia Foundation",
                    sentences=sentences,
                )


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
                    sentences=self._load_sentences(row["file_path"]),
                    url=row.get("url"),
                )


class TEDCorpusLoader(BaseCorpusLoader):
    corpus_name: str = "TED"

    def model_post_init(self, _context) -> None:
        self.corpus_dir = self.setup_corpus_dir()

    def load_metadata(self) -> Iterator[CorpusEntry]:
        """Load TED-specific metadata."""
        ted_talks: dict[str | int, Tuple[str, int, List[str]]] = {}

        datasets_to_load = [
            ("ted_talks_iwslt", "iwslt2014", 2014),
            ("ted_talks_iwslt", "iwslt2015", 2015),
            ("ted_talks_iwslt", "iwslt2016", 2016),
            ("iwslt2017", "iwslt2017-ja-en", 2017),
        ]

        for year_dataset in tqdm(datasets_to_load, desc="Loading TED datasets"):
            dataset = self._load_dataset(year_dataset)
            for example in tqdm(
                dataset, desc=f"Processing {year_dataset[1]} talks", leave=False
            ):
                talk_id = example.get("id", hash(example["translation"]["en"]))
                title = example.get("talk_name", f"TED Talk {talk_id}")
                year = year_dataset[2]
                text = example["translation"]["ja"].rstrip()

                if talk_id not in ted_talks:
                    ted_talks[talk_id] = (title, year, [])
                ted_talks[talk_id][2].append(text)

        for title, year, sentences in ted_talks.values():
            yield CorpusEntry(
                corpus=self.corpus_name,
                title=title,
                year=year,
                author="Various Speakers",
                publisher="TED Conference LLC",
                sentences=sentences,
            )

    def _load_dataset(self, year_dataset: Tuple[str, str, int]):
        if year_dataset[0] == "iwslt2017":
            return datasets.load_dataset(
                "iwslt2017", "iwslt2017-ja-en", trust_remote_code=True
            )["train"]
        return datasets.load_dataset(
            year_dataset[0],
            language_pair=("en", "ja"),
            year=str(year_dataset[2]),
            trust_remote_code=True,
        )["train"]


def get_wiki_corpus() -> list[tuple[str, list[str]]]:
    """Return structured wiki data with (title, sentences) tuples"""
    logger.info("Downloading Wikipedia corpus...")

    wiki_data = []
    ds = datasets.load_dataset(
        "wikimedia/wikipedia", "20231101.ja", streaming=True, split="train"
    )

    for article in ds.take(500):
        title = article["title"]
        text = article["text"]
        paragraphs = [p.strip() for p in text.split("\n\n") if is_japanese(p)]
        if paragraphs:
            wiki_data.append((title, paragraphs))

    return wiki_data


def prepare_corpus(loader: BaseCorpusLoader) -> int:
    """Prepare a corpus using its loader.

    Args:
        loader: The corpus loader to use

    Returns:
        Number of sentences processed
    """
    loader.download()

    db_path = loader.data_dir / "corpus.db"
    conn = init_database(db_path)
    total = 0

    try:
        # Get batched sources and sentences
        sources, source_map = loader.prepare_batch()

        if not sources:
            logger.warning(f"No sources found for corpus {loader.corpus_name}")
            return 0

        logger.info(f"Inserting {len(sources)} sources...")
        source_id_map = insert_sources_batch(conn, sources)

        # Prepare all sentences for bulk insert
        all_sentences = []
        for (corpus, title), sentences in source_map.items():
            if source_id := source_id_map.get((corpus, title)):
                all_sentences.extend(
                    [{"text": text, "source_id": source_id} for text in sentences]
                )
                total += len(sentences)
            else:
                logger.warning(f"Source {title} not found in ID map")

        # Bulk insert all sentences
        if all_sentences:
            sentences_df = pl.DataFrame(all_sentences)
            bulk_insert_sentences(conn, sentences_df)

    finally:
        conn.close()

    return total


def prepare_corpora(data_dir: Path) -> Dict[str, int]:
    """Prepare all corpora.

    Args:
        data_dir: Base directory for data

    Returns:
        Dictionary mapping corpus names to number of sentences processed
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    loaders = [
        JNLPCorpusLoader(data_dir=data_dir),
        TEDCorpusLoader(data_dir=data_dir),
        WikipediaCorpusLoader(data_dir=data_dir),
    ]

    return {loader.corpus_name: prepare_corpus(loader) for loader in loaders}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and load corpora for use in NLP tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store or load the prepared corpora (default: './data').",
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Standard corpus command
    corpus_parser = subparsers.add_parser("corpus", help="Process a standard corpus")
    corpus_parser.add_argument(
        "--name",
        choices=["all", "jnlp", "ted", "wiki"],
        default="all",
        help="Which standard corpus to prepare (default: all)",
    )

    # Generic corpus command
    generic_parser = subparsers.add_parser("generic", help="Process a generic corpus")
    generic_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the corpus (will be used as directory name)",
    )
    generic_parser.add_argument(
        "--dir",
        type=Path,
        help="Directory containing metadata.csv and corpus files (defaults to <data-dir>/<name>_corpus)",
    )

    args = parser.parse_args()

    set_random_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # Create data directory if it doesn't exist
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "corpus":
        loader_map = {
            "jnlp": JNLPCorpusLoader,
            "ted": TEDCorpusLoader,
            "wiki": WikipediaCorpusLoader,
        }

        if args.name == "all":
            results = prepare_corpora(args.data_dir)
            for corpus_name, count in results.items():
                logger.info(f"{corpus_name.upper()} corpus: {count} sentences prepared")
        else:
            loader_class = loader_map[args.name]
            loader = loader_class(args.data_dir)
            count = prepare_corpus(loader)
            logger.info(f"{args.name.upper()} corpus: {count} sentences prepared")

    elif args.command == "generic":
        corpus_dir = args.dir if args.dir else args.data_dir / f"{args.name}_corpus"
        loader = GenericCorpusLoader(args.data_dir, args.name)
        if args.dir:
            # If custom directory specified, ensure metadata.csv exists there
            loader.corpus_dir = args.dir
        count = prepare_corpus(loader)
        logger.info(f"{args.name.upper()} corpus: {count} sentences prepared")

    else:
        parser.print_help()
