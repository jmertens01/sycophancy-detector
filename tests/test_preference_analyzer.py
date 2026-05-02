"""Pytests to ensure proper operation of the PreferenceAnalyzer code."""

import pytest
from src.preference_data_generator import PreferenceDataGenerator


@pytest.fixture
def pref_analyzer() -> PreferenceDataGenerator:
    """Create a PreferenceAnalyzer to use throughout tests."""
    return PreferenceDataGenerator()


def test_ask_ollama(pref_analyzer: PreferenceDataGenerator) -> None:
    """Test that ollama is responding to prompts."""
    prompt = """
    Return the response 'agree' without any punctuation or other words -- just agree.
    """
    response = pref_analyzer.ask_ollama(prompt)
    content = response.message["content"]
    assert isinstance(content, str)
    assert content.lower() == "agree", f"Ollama returned {response}"
