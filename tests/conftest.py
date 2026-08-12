from pathlib import Path

import pytest
import spacy


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if (
            isinstance(item, pytest.DoctestItem)
            and Path(str(item.path)).name == "pattern_extraction.py"
        ):
            item.add_marker(pytest.mark.nlp_model)


@pytest.fixture(autouse=True)
def model_loading_requires_marker(request: pytest.FixtureRequest, monkeypatch) -> None:
    if request.node.get_closest_marker("nlp_model"):
        return

    def reject_unmarked_model_load(*_args, **_kwargs):
        raise AssertionError("spaCy model loading requires the nlp_model marker")

    monkeypatch.setattr(spacy, "load", reject_unmarked_model_load)
