import pytest
import spacy

from tests.model_smoke import assert_representative_parse, representative_parse


@pytest.mark.nlp_model
@pytest.mark.electra_model
def test_electra_model_loading() -> None:
    nlp = spacy.load("ja_ginza_electra")

    assert_representative_parse(representative_parse(nlp))
