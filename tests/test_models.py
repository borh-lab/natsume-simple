import pytest
import spacy


@pytest.mark.nlp_model
def test_model_loading():
    try:
        nlp = spacy.load("ja_ginza_electra")
    except OSError:
        nlp = spacy.load("ja_ginza")

    doc = nlp("これはテストです")
    assert len(doc) == 4
    assert [t.text for t in doc] == ["これ", "は", "テスト", "です"]
