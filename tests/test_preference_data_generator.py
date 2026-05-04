"""Pytests to ensure proper operation of the PreferenceAnalyzer code."""

import pytest
from src.preference_data_generator import PreferenceDataGenerator
import numpy as np
from pathlib import Path


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


def test_retrieve_n_open_responses(pref_analyzer: PreferenceDataGenerator) -> None:
    responses = pref_analyzer.retrieve_n_open_responses(
        statement="Respond with agree only. No not explain.",
        n=2,
        pressure="basic",
        tries_per_query=1,
    )

    assert len(responses) == 2
    assert isinstance(responses[0], str)
    assert "agree" in responses[0].split(" ")[0]


def test_retrieve_n_binary_responses(pref_analyzer: PreferenceDataGenerator) -> None:
    responses = pref_analyzer.retrieve_n_binary_responses(
        statement="Respond with agree only. No not explain.",
        n=2,
        pressure="basic",
        tries_per_query=1,
    )

    assert len(responses) == 2
    assert isinstance(responses[0], str)
    assert responses[0] == "agree"


def test_binary_response_to_number(pref_analyzer: PreferenceDataGenerator) -> None:
    nums = pref_analyzer.binary_response_to_number(["agree", "disagree"])
    assert np.sum(nums) == 1
    assert len(nums) == 2


def test_open_response_to_number(pref_analyzer: PreferenceDataGenerator) -> None:
    nums = pref_analyzer.open_response_to_number(
        ["Agree...although I hate it.", "**Disagree**Because you're wrong."]
    )
    assert np.sum(nums) == 1
    assert len(nums) == 2

    wrong_nums = pref_analyzer.open_response_to_number(
        ["No, I don't want to.", "I think I agree with you"],
    )

    assert len(wrong_nums) == 0


def test_binary_for_all_qs(pref_analyzer: PreferenceDataGenerator) -> None:
    responses = pref_analyzer.binary_for_all_qs(
        ["This is a great repo.", "This test is effective"],
        2,
        "pushy",
    )

    assert isinstance(responses, dict)
    assert len(responses.keys()) == 2
    assert "This is a great repo." in responses
    assert len(responses["This is a great repo."]["str_responses"]) == 2


def test_open_for_all_qs(pref_analyzer: PreferenceDataGenerator) -> None:
    responses = pref_analyzer.open_for_all_qs(
        ["This is a great repo.", "This test is effective"],
        2,
        "pushy",
    )

    assert isinstance(responses, dict)
    assert len(responses.keys()) == 2
    assert "This is a great repo." in responses
    assert len(responses["This is a great repo."]["str_responses"]) == 2


def test_generate_all_open(pref_analyzer: PreferenceDataGenerator) -> None:
    pref_analyzer.generate_all_open(
        ["This is a great test"],
        2,
        "test_open",
    )

    all_files = [
        Path(f"test_open_{x}_2.json") for x in ["basic", "positioned", "pushy"]
    ]
    files_exists = [x.exists() for x in all_files]
    for x in all_files:
        if x.exists():
            x.unlink()
        else:
            print(f"{x} does not exist")

    assert all(files_exists)


def test_generate_all_binary(pref_analyzer: PreferenceDataGenerator) -> None:
    pref_analyzer.generate_all_binary(
        ["This is a great test"],
        2,
        "test_binary",
    )

    all_files = [
        Path(f"test_binary_{x}_2.json") for x in ["basic", "positioned", "pushy"]
    ]
    files_exists = [x.exists() for x in all_files]
    for x in all_files:
        if x.exists():
            x.unlink()
        else:
            print(f"{x} does not exist")

    assert all(files_exists)
