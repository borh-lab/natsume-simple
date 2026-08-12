from pathlib import Path

from natsume_simple.database import init_database


def build_search_database(path: Path) -> Path:
    """Build the smallest database that exercises every legacy query direction."""
    conn = init_database(path)
    conn.execute(
        """
        INSERT INTO source (id, year, title, corpus) VALUES
            (1, 2024, 'Alpha one', 'alpha'),
            (2, 2024, 'Alpha two', 'alpha'),
            (3, 2025, 'Beta one', 'beta');

        INSERT INTO sentence (id, text, source_id) VALUES
            (1, '情報を集める。', 1),
            (2, '情報を安全に集める。', 2),
            (3, '<img src=x onerror=alert(1)>情報を集める。', 3),
            (4, '研究が進む。', 1),
            (5, '情報が進む。', 3);

        INSERT INTO lemma (id, string, pos) VALUES
            (1, '情報', 'NOUN'),
            (2, 'を', 'ADP'),
            (3, '集める', 'VERB'),
            (4, '研究', 'NOUN'),
            (5, 'が', 'ADP'),
            (6, '進む', 'VERB');

        INSERT INTO word (id, string, pron, inf, dep, lemma_id) VALUES
            (1, '情報', 'ジョウホウ', NULL, 'obj', 1),
            (2, 'を', 'ヲ', NULL, 'case', 2),
            (3, '集める', 'アツメル', NULL, 'ROOT', 3),
            (4, '研究', 'ケンキュウ', NULL, 'nsubj', 4),
            (5, 'が', 'ガ', NULL, 'case', 5),
            (6, '進む', 'ススム', NULL, 'ROOT', 6);

        INSERT INTO sentence_word (id, sentence_id, word_id, begin, "end") VALUES
            (1, 1, 1, 0, 2), (2, 1, 2, 2, 3), (3, 1, 3, 3, 6),
            (4, 2, 1, 0, 2), (5, 2, 2, 2, 3), (6, 2, 3, 6, 9),
            (7, 3, 1, 28, 30), (8, 3, 2, 30, 31), (9, 3, 3, 31, 34),
            (10, 4, 4, 0, 2), (11, 4, 5, 2, 3), (12, 4, 6, 3, 5),
            (13, 5, 1, 0, 2), (14, 5, 5, 2, 3), (15, 5, 6, 3, 5);

        INSERT INTO collocation VALUES
            (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15);
        """
    )
    conn.close()
    return path
