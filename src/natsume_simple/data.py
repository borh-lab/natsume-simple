import argparse
import random
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple

import datasets  # type: ignore

from natsume_simple.log import setup_logger
from natsume_simple.utils import set_random_seed

logger = setup_logger(__name__)


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


def download_jnlp_corpus(data_dir: Path) -> None:
    """Download the JNLP corpus from the official website.

    Downloads the NLP_LATEX_CORPUS.zip file and extracts it to the specified directory.

    Args:
        data_dir: Directory where the corpus will be downloaded and extracted
    """
    file = data_dir / "NLP_LATEX_CORPUS.zip"
    if not file.is_file():
        logger.info("Downloading JNLP corpus...")
        urllib.request.urlretrieve(
            "https://www.anlp.jp/resource/journal_latex/NLP_LATEX_CORPUS.zip",
            file,
        )

    with zipfile.ZipFile(file, "r") as z:
        z.extractall(data_dir)


def convert_latex_to_text(corpus_dir: Path) -> None:
    """Convert LaTeX files to plain text using pandoc.

    Finds all .tex files in the directory (recursively) and converts them to .txt files.
    Skips files that have already been converted.

    Args:
        corpus_dir: Directory containing LaTeX files to convert
    """
    latex_files = corpus_dir.rglob("*.tex")

    for latex_file in latex_files:
        text_file = latex_file.with_suffix(".txt")
        logger.info(f"{latex_file} => {text_file}")
        if text_file.is_file():
            logger.info("File exists, skipping.")
            continue

        p = subprocess.run(
            [
                "pandoc",
                "--quiet",
                "-f",
                "latex+east_asian_line_breaks",
                "-t",
                "plain",
                "--wrap=none",
                "--strip-comments",
                "-N",
                "-s",
                latex_file,
                "-o",
                text_file,
            ]
        )
        if p.returncode != 0:
            logger.error("Failed to convert using pandoc, skipping!")


def prepare_jnlp_corpus(data_dir: Path) -> int:
    """Prepare the JNLP corpus by downloading, converting, and filtering.

    Downloads the corpus, converts LaTeX to text, filters non-Japanese content,
    and saves the result to a single text file.

    Args:
        data_dir: Directory for storing the corpus data

    Returns:
        Number of lines in the prepared corpus
    """
    corpus_dir = data_dir / "NLP_LATEX_CORPUS"
    download_jnlp_corpus(data_dir)
    convert_latex_to_text(corpus_dir)

    logger.info("Filtering and writing JNLP corpus...")
    lines = list(filter_non_japanese(dir=corpus_dir))

    output_file = data_dir / "jnlp-corpus.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

    logger.info(f"JNLP corpus saved to {output_file}")
    return len(lines)


def get_ted_corpus() -> List[str]:
    """Download and combine TED talk transcripts from multiple years.

    Downloads Japanese transcripts from TED talks using the datasets library,
    combining data from 2014-2017.

    Returns:
        List of Japanese sentences from TED talks, stripped of whitespace
    """
    logger.info("Downloading TED corpus...")

    ted_dataset_2014 = datasets.load_dataset(
        "ted_talks_iwslt",
        language_pair=("en", "ja"),
        year="2014",
        trust_remote_code=True,
    )
    ted_dataset_2015 = datasets.load_dataset(
        "ted_talks_iwslt", language_pair=("en", "ja"), year="2015"
    )
    ted_dataset_2016 = datasets.load_dataset(
        "ted_talks_iwslt", language_pair=("en", "ja"), year="2016"
    )
    ted_dataset_2017jaen = datasets.load_dataset(
        "iwslt2017", "iwslt2017-ja-en", trust_remote_code=True
    )

    ted_corpus: List[str] = (
        [d["ja"].rstrip() for d in ted_dataset_2014["train"]["translation"]]
        + [d["ja"].rstrip() for d in ted_dataset_2015["train"]["translation"]]
        + [d["ja"].rstrip() for d in ted_dataset_2016["train"]["translation"]]
        + [d["ja"].rstrip() for d in ted_dataset_2017jaen["train"]["translation"]]
    )

    return ted_corpus


def get_wiki_corpus() -> List[str]:
    """Download a subset of Wikipedia.


    Returns:
        List of Japanese sentences from Wikipedia, stripped of whitespace
    """
    logger.info("Downloading TED corpus...")

    ds = datasets.load_dataset(
        "wikimedia/wikipedia", "20231101.ja", streaming=True, split="train"
    )

    wiki_corpus = [
        paragraph
        for article in ds.take(100)
        for paragraph in article["text"].split("\n\n")
        if is_japanese(paragraph)
    ]

    return wiki_corpus


def save_corpus(data_dir: Path, corpus_name: str, corpus: List[str]) -> None:
    """Save a corpus to a file using standard naming convention.

    Args:
        data_dir: Directory to save the corpus file
        corpus_name: Name of the corpus (e.g., 'jnlp', 'ted')
        corpus: List of corpus lines to save (assumed to not have newline at end)
    """
    output_file = data_dir / f"{corpus_name}-corpus.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(f"{line}\n")
    logger.info(f"Sampled {corpus_name} corpus saved to {output_file}")


def save_corpora(data_dir: Path, corpora: dict[str, List[str]]) -> None:
    """Save multiple corpora to files.

    Args:
        data_dir: Directory to save the corpus files
        corpora: Dictionary mapping corpus names to their content
    """
    for corpus_name, corpus in corpora.items():
        save_corpus(data_dir, corpus_name, corpus)


def prepare_ted_corpus(data_dir: Path) -> int:
    """Prepare the TED corpus by downloading and saving to file.

    Downloads the TED corpus and saves it to a single text file.

    Args:
        data_dir: Directory where the corpus will be saved

    Returns:
        Number of sentences in the corpus
    """
    ted_corpus = get_ted_corpus()
    save_corpus(data_dir, "ted", ted_corpus)
    return len(ted_corpus)


def prepare_wiki_corpus(data_dir: Path) -> int:
    """Prepare the Wikipedia corpus by downloading and saving to file.

    Downloads the Wikipedia corpus and saves it to a single text file.

    Args:
        data_dir: Directory where the corpus will be saved

    Returns:
        Number of sentences in the corpus
    """
    wiki_corpus = get_wiki_corpus()
    save_corpus(data_dir, "wiki", wiki_corpus)
    return len(wiki_corpus)


def prepare_corpora(data_dir: Path) -> Tuple[int, int, int]:
    """Prepare both corpora.

    Creates the data directory if needed and prepares the corpora.

    Args:
        data_dir: Directory for storing corpora

    Returns:
        Tuple of (JNLP corpus size, TED corpus size, Wikipedia corpus size)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    jnlp_count = prepare_jnlp_corpus(data_dir)
    ted_count = prepare_ted_corpus(data_dir)
    wiki_count = prepare_wiki_corpus(data_dir)
    return jnlp_count, ted_count, wiki_count


def load_corpus(data_dir: Path, corpus_name: str, sample_size: int) -> List[str]:
    """Load and sample from a prepared corpus.

    Args:
        data_dir: Directory containing the corpus files
        corpus_name: Name of the corpus ('jnlp' or 'ted')
        sample_size: Number of lines to randomly sample

    Returns:
        List of sampled lines from the corpus

    Examples:
        >>> from io import StringIO
        >>> from unittest.mock import patch
        >>> test_data = StringIO("line1\\nline2\\nline3\\nline4\\nline5")
        >>> with patch('builtins.open', return_value=test_data):
        ...     lines = load_corpus(Path("."), "test", 3)
        ...     len(lines)
        3
    """
    corpus_file = data_dir / f"{corpus_name}-corpus.txt"
    with open(corpus_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    return lines[:sample_size]


def load_corpora(
    data_dir: Path,
    jnlp_sample_size: int = 3000,
    ted_sample_size: int = 30000,
    wiki_sample_size: int = 3000,
) -> Tuple[List[str], List[str], List[str]]:
    """Load and sample from both JNLP and TED corpora.

    Args:
        data_dir: Directory containing the corpus files
        jnlp_sample_size: Number of lines to sample from JNLP corpus
        ted_sample_size: Number of lines to sample from TED corpus
        wiki_sample_size: Number of lines to sample from Wikipedia corpus

    Returns:
        Tuple of (JNLP corpus samples, TED corpus samples)
    """
    jnlp_corpus = load_corpus(data_dir, "jnlp", jnlp_sample_size)
    ted_corpus = load_corpus(data_dir, "ted", ted_sample_size)
    wiki_corpus = load_corpus(data_dir, "wiki", wiki_sample_size)
    return jnlp_corpus, ted_corpus, wiki_corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and load JNLP and TED corpora for use in NLP tasks.",
        epilog="This script can download, process, prepare, and load the JNLP and TED corpora for further use in natural language processing tasks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare the corpora by downloading and processing the data.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the prepared corpora.",
    )
    parser.add_argument(
        "--jnlp-sample-size",
        type=int,
        default=3000,
        help="Sample size for the JNLP corpus (default: 3000). This determines how many sentences from the JNLP corpus will be randomly selected when loading.",
    )
    parser.add_argument(
        "--wiki-sample-size",
        type=int,
        default=3000,
        help="Sample size for the Wikipedia corpus (default: 3000). This determines how many sentences from the JNLP corpus will be randomly selected when loading.",
    )
    parser.add_argument(
        "--ted-sample-size",
        type=int,
        default=30000,
        help="Sample size for the TED corpus (default: 30000). This determines how many sentences from the TED corpus will be randomly selected when loading.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to store or load the prepared corpora (default: './data').",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    if not (args.prepare or args.load):
        parser.print_help()
        print("\nNo action specified. Use --prepare or --load to perform operations.")
    else:
        if args.prepare:
            jnlp_count, ted_count, wiki_count = prepare_corpora(args.data_dir)
            logger.info(f"Corpora prepared and saved in {args.data_dir}")
            logger.info(f"JNLP corpus: {jnlp_count} sentences/paragraphs prepared")
            logger.info(f"TED corpus: {ted_count} sentences/paragraphs prepared")
            logger.info(f"Wikipedia corpus: {wiki_count} sentences/paragraphs prepared")

        if args.load:
            jnlp_corpus, ted_corpus, wiki_corpus = load_corpora(
                args.data_dir,
                args.jnlp_sample_size,
                args.ted_sample_size,
                args.wiki_sample_size,
            )
            logger.info(
                f"Loaded {len(jnlp_corpus)} sentences from JNLP corpus (sample size: {args.jnlp_sample_size})"
            )
            logger.info(
                f"Loaded {len(ted_corpus)} sentences from TED corpus (sample size: {args.ted_sample_size})"
            )
            logger.info(
                f"Loaded {len(wiki_corpus)} sentences from Wikipedia corpus (sample size: {args.wiki_sample_size})"
            )
            save_corpora(
                args.data_dir,
                {"jnlp": jnlp_corpus, "ted": ted_corpus, "wiki": wiki_corpus},
            )
