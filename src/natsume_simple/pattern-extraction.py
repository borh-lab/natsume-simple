import spacy  # type: ignore
import ginza  # type: ignore
import polars as pl  # type: ignore
from spacy.tokens import Doc, Token, Span  # type: ignore
import torch  # type: ignore
from spacy.symbols import (  # type: ignore
    NOUN,
    PROPN,
    PRON,
    NUM,
    VERB,
    SYM,
    PUNCT,
    ADP,
    SCONJ,
    obj,
    obl,
    nsubj,
)
from typing import List, Tuple, Optional, Iterable, Any
import re
from pathlib import Path
from itertools import chain, takewhile, tee
from collections.abc import Iterator
import argparse
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

    if torch.cuda.is_available():
        logger.info("GPU is available. Enabling GPU support for spaCy.")
        spacy.require_gpu()
    else:
        logger.info("CUDA is not available. Using whatever spaCy finds or the CPU.")
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
        >>> from spacy.tokens import Token
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
    """
    clean_tokens = [token for token in tokens if token.pos not in {PUNCT, SYM}]
    clean_tokens = list(
        takewhile(
            lambda token: token.pos not in {ADP, SCONJ}
            and token.norm_ not in {"から", "ため", "たり", "こと", "よう"},
            clean_tokens,
        )
    )
    if len(clean_tokens) == 1:
        return simple_lemma(clean_tokens[0])

    normalized_tokens: List[Token] = []
    token_pairs: List[Tuple[Token, Token]] = list(pairwise(clean_tokens))
    for i, (token, next_token) in enumerate(token_pairs):
        normalized_tokens.append(token)
        if next_token.lemma_ in ["ます", "た"]:
            if re.match(r"^(五|上|下|サ|.変格|助動詞).+", ginza.inflection(token)):
                break
            else:
                normalized_tokens.append(suru_token)
                break
        elif next_token.lemma_ == "だ":
            break
        elif i == len(token_pairs) - 1:
            normalized_tokens.append(next_token)

    if len(normalized_tokens) == 1:
        return simple_lemma(normalized_tokens[0])

    if not normalized_tokens:
        return None

    stem = normalized_tokens[0]
    affixes = normalized_tokens[1:-1]
    suffix = normalized_tokens[-1]
    return "{}{}{}".format(
        stem.text,
        "".join(t.text for t in affixes),
        simple_lemma(suffix),
    )


def npv_matcher(doc: Doc, suru_token: Token) -> List[Tuple[str, str, str]]:
    """
    Extract NPV (Noun-Particle-Verb) patterns from a document.

    Args:
        doc (Doc): The input spaCy document.

    Returns:
        List[Tuple[str, str, str]]: A list of NPV patterns.
    """
    matches: List[Tuple[str, str, str]] = []
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
            matches.append(
                (
                    noun.norm_,
                    case_particle.norm_,
                    vp_string,
                )
            )
    return matches


def process_corpus(
    corpus: List[str], nlp: spacy.language.Language, suru_token: Token
) -> List[Tuple[str, str, str]]:
    """
    Process the entire corpus and extract NPV patterns.

    Args:
        corpus (List[str]): The input corpus as a list of strings.
        nlp (spacy.language.Language): The loaded NLP model.
        suru_token (Token): The constant する token.

    Returns:
        List[Tuple[str, str, str]]: A list of NPV patterns extracted from the corpus.
    """
    return list(
        chain.from_iterable(npv_matcher(doc, suru_token) for doc in nlp.pipe(corpus))
    )


def save_results(
    results: List[Tuple[str, str, str]],
    data_dir: Path,
    corpus_name: str,
    model_name: str,
):
    """
    Save results to a CSV file using the standard naming convention.

    Args:
        results (List[Tuple[str, str, str]]): The NPV patterns to save.
        data_dir (Path): Directory to save the output file.
        corpus_name (str): The name of the corpus ('ted' or 'jnlp').
        model_name (str): Name of the spaCy model used.
    """
    output_file = data_dir / f"{corpus_name}_npvs_{model_name}.csv"
    df = pl.DataFrame(results, schema=["n", "p", "v"], orient="row")
    df = df.with_columns(pl.lit(corpus_name).alias("corpus"))
    df.write_csv(output_file)


nlp, suru_token = load_nlp_model()


def main(
    input_file: Path,
    data_dir: Path,
    model_name: Optional[str] = None,
    corpus_name: str = "Unknown",
) -> None:
    """
    Main function to process a corpus file and save results.

    Args:
        input_file (Path): The path to the input corpus file.
        data_dir (Path): Directory to save the output file.
        model_name (Optional[str]): The name of the spaCy model to use.
        corpus_name (str): The name of the corpus ('ted' or 'jnlp').
    """
    global nlp, suru_token
    used_model = model_name if model_name else nlp.meta["name"]
    if model_name:
        nlp, suru_token = load_nlp_model(model_name)

    with open(input_file, "r", encoding="utf-8") as f:
        corpus = f.readlines()

    results = process_corpus(corpus, nlp, suru_token)
    save_results(results, data_dir, corpus_name, used_model)

    logger.info(f"Processed {len(corpus)} lines.")
    logger.info(f"Extracted {len(results)} NPV patterns.")
    logger.info(f"Results saved to {data_dir}/{corpus_name}_npvs_{used_model}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NPV patterns from a corpus.")
    parser.add_argument("input_file", type=Path, help="Path to the input corpus file")
    parser.add_argument(
        "data_dir", type=Path, help="Directory to save the output CSV file"
    )
    parser.add_argument("--model", type=str, help="Name of the spaCy model to use")
    parser.add_argument(
        "--corpus-name", type=str, default="Unknown", help="Name of the corpus"
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

    main(args.input_file, args.output_file, args.model, args.corpus_name)
