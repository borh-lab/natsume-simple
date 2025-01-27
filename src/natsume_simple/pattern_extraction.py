import argparse
import re
from collections.abc import Iterator
from itertools import chain, dropwhile, takewhile, tee
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import ginza  # type: ignore
import spacy  # type: ignore
import torch  # type: ignore
from spacy.symbols import (  # type: ignore
    ADJ,
    ADP,
    AUX,
    CCONJ,
    NOUN,
    NUM,
    PART,
    PRON,
    PROPN,
    PUNCT,
    SCONJ,
    SYM,
    VERB,
    nsubj,
    obj,
    obl,
)
from spacy.tokens import Doc, Span, Token  # type: ignore
from tqdm import tqdm

from natsume_simple.database import (
    bulk_insert_lemmas,
    bulk_insert_sentence_words,
    bulk_insert_words,
    create_indices,
    save_collocations,
)
from natsume_simple.log import setup_logger
from natsume_simple.utils import set_random_seed

logger = setup_logger(__name__)


def load_nlp_model(
    model_name: Optional[str] = None,
) -> Tuple[spacy.language.Language, Token]:
    """
    Load and return the NLP model and a constant する token.

    Args:
        model_name (Optional[str]): The name of the model to load. If None, tries to load 'ja_ginza_bert_large' first, then falls back to 'ja_ginza'.

    Returns:
        Tuple[spacy.language.Language, Token]: The loaded NLP model and a constant する token.
    """

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        logger.info("GPU is available. Enabling GPU support for spaCy.")
        try:
            spacy.require_gpu()
        except ValueError as e:
            logger.error(f"Error enabling GPU support: {e}, using CPU.")
    else:
        logger.info("GPU is not available. Using whatever spaCy finds or the CPU.")
        spacy.prefer_gpu()

    if model_name:
        nlp = spacy.load(model_name)
    else:
        try:
            nlp = spacy.load("ja_ginza_bert_large")
        except Exception:
            nlp = spacy.load("ja_ginza")

    suru_token = nlp("する")[0]

    return nlp, suru_token


def pairwise(iterable: Iterable[Any]) -> Iterator[Tuple[Any, Any]]:
    """Create pairwise iterator from an iterable.

    Args:
        iterable (Iterable[Any]): The input iterable.

    Returns:
        Iterator[Tuple[Any, Any]]: An iterator of pairs.

    Examples:
        >>> list(pairwise([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4)]
        >>> list(pairwise("abc"))
        [('a', 'b'), ('b', 'c')]
        >>> list(pairwise([]))
        []
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def simple_lemma(token: Token) -> str:
    """Get a simplified lemma for a UniDic token.

    Args:
        token (Token): The input token.

    Returns:
        str: The simplified lemma.

    Examples:
        >>> nlp = spacy.load("ja_ginza")
        >>> simple_lemma(nlp("する")[0])
        'する'
        >>> simple_lemma(nlp("居る")[0])
        '居る'
        >>> simple_lemma(nlp("食べる")[0])
        '食べる'
    """
    if token.norm_ == "為る":
        return "する"
    elif token.norm_ in ["居る", "成る", "有る"]:
        return token.lemma_
    else:
        return token.norm_


def normalize_verb_span(tokens: Doc | Span, suru_token: Token) -> Optional[str]:
    """
    Normalize a verb span.

    Args:
        tokens (Doc | Span): The input tokens.
        suru_token (Token): The constant する token.

    Returns:
        Optional[str]: The normalized verb string, or None if normalization fails.

    Examples:
        >>> nlp = spacy.load("ja_ginza")
        >>> doc, suru_token = nlp("飛び立つでしょう"), nlp("する")[0]
        >>> normalize_verb_span(doc, suru_token)
        '飛び立つ'
        >>> normalize_verb_span(nlp("考えられませんでした"), suru_token)
        '考えられる'
        >>> normalize_verb_span(nlp("扱うかです"), suru_token)
        '扱う'
        >>> normalize_verb_span(nlp("突入しちゃう"), suru_token)
        '突入する'
        >>> normalize_verb_span(nlp("で囲んである"), suru_token)
        '囲む'
        >>> normalize_verb_span(nlp("たらしめている"), suru_token)
        'たらしめる'
    """
    # Chain filtering steps together
    clean_tokens = list(
        chain(
            takewhile(
                lambda token: (
                    token.pos not in {ADP, SCONJ, PART}
                    and token.tag_ not in {"助詞-接続助詞"}
                    or (
                        token.pos == AUX and token.lemma_ == "れる"
                    )  # Keep れる auxiliary
                    or (
                        token.pos == ADJ and token.dep_ == "amod"
                    )  # Keep ADJ when it modifies
                ),
                dropwhile(
                    lambda token: token.pos in {ADP, CCONJ, SCONJ, PART},
                    (token for token in tokens if token.pos not in {PUNCT, SYM}),
                ),
            )
        )
    )

    if len(clean_tokens) == 1:
        return simple_lemma(clean_tokens[0])

    if not clean_tokens:
        logger.warning(
            f"Failed to normalize verb span: {[t for t in tokens]}; clean_tokens: {clean_tokens}"
        )
        return None

    normalized_tokens: List[Token] = []
    for i, (token, next_token) in enumerate(pairwise(clean_tokens)):
        normalized_tokens.append(token)
        if next_token.lemma_ in ["ます", "た"]:
            # Check UniDic POS tag for サ変可能 nouns
            if "サ変可能" in token.tag_ or token.tag_.startswith("名詞-普通名詞-サ変"):
                logger.debug(
                    f"Adding suru token to: {normalized_tokens} in {tokens}/{clean_tokens}"
                )
                normalized_tokens.append(suru_token)
                break
            elif token.tag_.startswith("動詞") or re.match(
                r"^(五|上|下|サ|.変格|助動詞).+", ginza.inflection(token)
            ):
                break
            else:
                logger.warning(
                    f"Unexpected token pattern: {token.text}/{token.tag_} in {tokens}/{clean_tokens}"
                )
        elif next_token.lemma_ in {
            "から",
            "ため",
            "たり",
            "こと",
            "よう",
            "です",
            "べし",
            "だ",
        }:
            # Stop here and don't include the stopword token
            break
        elif i == len(clean_tokens) - 2:
            normalized_tokens.append(next_token)

    logger.debug(f"Normalized tokens: {[t.text for t in normalized_tokens]}")
    if len(normalized_tokens) == 1:
        return simple_lemma(normalized_tokens[0])

    if not normalized_tokens:
        logger.warning(
            f"Failed to normalize verb span: {[t for t in tokens]}; clean_tokens: {clean_tokens}"
        )
        return None

    stem = normalized_tokens[0]
    affixes = normalized_tokens[1:-1]
    suffix = normalized_tokens[-1]
    return "{}{}{}".format(
        stem.text,
        "".join(t.text for t in affixes),
        simple_lemma(suffix),
    )


def npv_matcher(
    doc: Doc, suru_token: Token
) -> List[Tuple[str, str, str, int, int, int, int, int, int]]:
    """
    Extract NPV (Noun-Particle-Verb) patterns from a document with character positions.

    Args:
        doc (Doc): The input spaCy document.

    Returns:
        List[Tuple[str, str, str, int, int, int, int, int, int]]: A list of NPV patterns with positions.
    """
    matches: List[Tuple[str, str, str, int, int, int, int, int, int]] = []
    for token in doc[:-2]:
        noun = token
        case_particle = noun.nbor(1)
        verb = token.head
        if (
            noun.pos in {NOUN, PROPN, PRON, NUM}
            and noun.dep in {obj, obl, nsubj}
            and verb.pos == VERB
            and case_particle.dep_ == "case"
            and case_particle.lemma_
            in {"が", "を", "に", "で", "から", "より", "と", "へ"}
            and case_particle.nbor().dep_ != "fixed"
            and case_particle.nbor().head != case_particle.head
        ):
            verb_bunsetu_span = ginza.bunsetu_span(verb)
            vp_string = normalize_verb_span(verb_bunsetu_span, suru_token)
            if not vp_string:
                logger.error(
                    f"Error normalizing verb phrase: {verb_bunsetu_span} in document {doc}"
                )
                continue

            # Get positions
            n_begin = noun.idx
            n_end = noun.idx + len(noun.text)
            p_begin = case_particle.idx
            p_end = case_particle.idx + len(case_particle.text)
            # Note that this is not the whole verb, just the head token
            # TODO: normalize_verb_span should return the a token span instead of a string
            v_begin = verb.idx
            v_end = verb.idx + len(verb.text)

            matches.append(
                (
                    noun.norm_,
                    case_particle.norm_,
                    vp_string,
                    n_begin,
                    n_end,
                    p_begin,
                    p_end,
                    v_begin,
                    v_end,
                )
            )
    return matches


def process_sentence(
    doc: Doc,
    sentence_id: int,
    suru_token: Token,
) -> Tuple[
    List[Tuple[int, int, str, str, str, Optional[str], str]],
    List[Tuple[int, str, str, str, int, int, int, int, int, int]],
]:
    """Process a single sentence, extracting both word info and patterns.

    Returns:
        Tuple of (word_entries, npv_patterns) where:
        - word_entries is List of (begin, end, lemma, pos, pron, inf, dep)
        - npv_patterns is List of (sentence_id, noun, particle, verb, n_begin, n_end, p_begin, p_end, v_begin, v_end)
    """
    # Extract word information with morphological features
    word_entries = []
    for token in doc:
        # Get pronunciation from morph features
        pron = token.morph.get("Reading") or token.text
        # Get inflection type from morph features
        inf = token.morph.get("Inflection") or None
        # Get dependency relation
        dep = token.dep_

        word_entries.append(
            (
                token.idx,
                token.idx + len(token.text),
                token.lemma_,
                token.pos_,
                pron[0] if isinstance(pron, list) else pron,  # Take first reading
                inf[0] if isinstance(inf, list) else inf,  # Take first inflection
                dep,
            )
        )

    # Extract NPV patterns
    patterns = npv_matcher(doc, suru_token)

    return word_entries, [(sentence_id, *p) for p in patterns]


def save_sentence_processing_results(
    conn: duckdb.DuckDBPyConnection,
    word_entries: List[Tuple[int, int, str, str, str, Optional[str], str]],
    sentence_id: int,
) -> Dict[Tuple[int, int], int]:
    """Save word information and return mapping of spans to sentence_word IDs."""
    # Prepare data for bulk inserts
    lemmas = [
        {"string": lemma, "pos": pos} for _, _, lemma, pos, _, _, _ in word_entries
    ]

    # Get lemma IDs in bulk
    lemma_id_map = bulk_insert_lemmas(conn, lemmas)

    # Prepare word data with lemma IDs
    words = [
        {
            "string": lemma,
            "pron": pron,
            "inf": inf,
            "dep": dep,
            "lemma_id": lemma_id_map[(lemma, pos)],
        }
        for _, _, lemma, pos, pron, inf, dep in word_entries
    ]

    # Get word IDs in bulk
    word_id_map = bulk_insert_words(conn, words)

    # Prepare sentence_word data
    sentence_words = [
        {
            "sentence_id": sentence_id,
            "word_id": word_id_map[(lemma, lemma_id_map[(lemma, pos)])],
            "begin": begin,
            "end": end,
        }
        for begin, end, lemma, pos, _, _, _ in word_entries
    ]

    # Get sentence_word IDs in bulk
    sw_id_map = bulk_insert_sentence_words(conn, sentence_words)

    # Create span to ID mapping
    return {
        (begin, end): sw_id_map[
            (sentence_id, word_id_map[(lemma, lemma_id_map[(lemma, pos)])], begin, end)
        ]
        for begin, end, lemma, pos, _, _, _ in word_entries
    }


def process_corpus(
    sentences: List[Tuple[str, int]],
    nlp: spacy.language.Language,
    suru_token: Token,
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 1000,
) -> None:
    """Process sentences and save results in one pass.

    Args:
        sentences: List of (text, sentence_id) pairs
        nlp: The loaded NLP model
        suru_token: The constant する token
        conn: Database connection
        batch_size: Number of sentences to process in each batch
    """
    # First phase: Process all sentences with spaCy in batches
    logger.info("Processing sentences with spaCy...")
    processed_batches = []

    # Create a single progress bar for all batches
    pbar = tqdm(total=len(sentences) // batch_size + 1, desc="Processing sentences")

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        # Process batch with spaCy
        docs = list(nlp.pipe(batch, as_tuples=True, batch_size=batch_size))

        # Process each document in the batch
        batch_results = []
        for doc, sentence_id in docs:
            # Process sentence
            word_entries, patterns = process_sentence(doc, sentence_id, suru_token)
            batch_results.append((word_entries, patterns, sentence_id))

        processed_batches.append(batch_results)
        pbar.update(1)  # Update progress after each batch
        pbar.set_postfix(
            {"sentences": i + len(batch)}
        )  # Show total sentences processed

    pbar.close()

    # Second phase: Save all results to database
    logger.info("Saving results to database...")
    all_patterns = []

    for batch in tqdm(processed_batches, desc="Saving batches"):
        # Process each sentence in the batch
        for word_entries, patterns, sentence_id in batch:
            # Save word information and get span mapping
            span_to_id = save_sentence_processing_results(
                conn, word_entries, sentence_id
            )

            # Convert patterns to use sentence_word IDs
            for pattern in patterns:
                (
                    sentence_id,
                    _noun,
                    _particle,
                    _verb,
                    n_begin,
                    n_end,
                    p_begin,
                    p_end,
                    v_begin,
                    v_end,
                ) = pattern

                word_1_id = span_to_id.get((n_begin, n_end))
                particle_id = span_to_id.get((p_begin, p_end))
                word_2_id = span_to_id.get((v_begin, v_end))

                if None not in (word_1_id, particle_id, word_2_id):
                    all_patterns.append((word_1_id, particle_id, word_2_id))

        # Commit after each batch to avoid huge transactions
        conn.commit()

    logger.info(f"Saving {len(all_patterns)} patterns to database...")
    # Bulk save all patterns at once
    save_collocations(conn, all_patterns)


nlp, suru_token = load_nlp_model()


def main(
    data_dir: Path,
    model_name: Optional[str] = None,
    corpus_name: Optional[str] = None,
    unprocessed_only: bool = False,
    sample_ratio: Optional[float] = None,
    batch_size: int = 1000,
) -> None:
    """Process sentences and extract collocations.

    Args:
        data_dir: Directory containing the database
        model_name: Optional name of the spaCy model to use
        corpus_name: Optional corpus name to process (if None, process all)
        unprocessed_only: Only process sentences without existing collocations
        sample_ratio: If set, process only this ratio of sentences (0.0-1.0)
        batch_size: Number of sentences to process in each batch (default: 1000)
    """
    global nlp, suru_token
    if model_name:
        nlp, suru_token = load_nlp_model(model_name)

    conn = duckdb.connect(str(data_dir / "corpus.db"))

    try:
        # Build query based on options - note the SELECT order is now (text, id)
        if unprocessed_only:
            query = """
                SELECT DISTINCT s.text, s.id
                FROM sentence s
                JOIN source src ON s.source_id = src.id
                LEFT JOIN sentence_word sw ON s.id = sw.sentence_id
                LEFT JOIN collocation c ON 
                    sw.id = c.word_1_sw_id OR 
                    sw.id = c.particle_sw_id OR 
                    sw.id = c.word_2_sw_id
                WHERE c.word_1_sw_id IS NULL
            """
        else:
            query = """
                SELECT s.text, s.id
                FROM sentence s
                JOIN source src ON s.source_id = src.id
            """

        params = []

        if corpus_name:
            query += (
                " AND src.corpus = ?" if unprocessed_only else " WHERE src.corpus = ?"
            )
            params.append(corpus_name)
            logger.info(f"Processing corpus: {corpus_name}")
        else:
            logger.info("Processing all corpora")

        # Add random sampling if requested
        if sample_ratio is not None:
            if not (0.0 < sample_ratio <= 1.0):
                raise ValueError("Sample ratio must be between 0.0 and 1.0")
            query += f" USING SAMPLE {sample_ratio * 100}%"
            logger.info(f"Using {sample_ratio * 100}% sample of sentences")

        # Get sentences with their database IDs
        sentences = conn.execute(query, params).fetchall()
        if not sentences:
            logger.warning("No sentences found to process")
            return

        logger.info(f"Found {len(sentences)} sentences to process")

        # Process everything in one pass
        process_corpus(sentences, nlp, suru_token, conn, batch_size)

        # Create indices after all data is loaded
        create_indices(conn)

        conn.commit()

        logger.info(f"Processed {len(sentences)} sentences")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NPV patterns from corpora.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the database",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the spaCy model to use",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        help="Name of specific corpus to process (default: process all)",
    )
    parser.add_argument(
        "--unprocessed-only",
        action="store_true",
        help="Only process sentences that haven't been processed for collocations yet",
    )
    parser.add_argument(
        "--sample",
        type=float,
        help="Process only a random sample of sentences (0.0-1.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of sentences to process in each batch (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    set_random_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    main(
        args.data_dir,
        args.model,
        args.corpus,
        args.unprocessed_only,
        args.sample,
        args.batch_size,
    )
