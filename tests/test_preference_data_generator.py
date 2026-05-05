"""Pytests to ensure proper operation of the PreferenceAnalyzer code."""

from pathlib import Path

import numpy as np
import pytest

from src.sycophancy_analyzer.preference_data_generator import PreferenceDataGenerator


@pytest.fixture
def pref_analyzer() -> PreferenceDataGenerator:
    """Create a PreferenceAnalyzer to use throughout tests."""
    pref_analyzer = PreferenceDataGenerator()
    pref_analyzer.statements = ["This is a great test."]

    return pref_analyzer


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
    """Test the ability to retrieve multiple open responses to statements."""
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
    """Test ability to retreive binary response to multiple statements."""
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
    """Test conversion from binary string responses to binary (agree = 1)."""
    nums = pref_analyzer.binary_response_to_number(["agree", "disagree"])
    assert np.sum(nums) == 1
    assert len(nums) == 2


def test_open_response_to_number(pref_analyzer: PreferenceDataGenerator) -> None:
    """Test conversion from open responses to binary (agree = 1)."""
    nums = pref_analyzer.open_response_to_number(
        ["Agree...although I hate it.", "**Disagree**Because you're wrong."],
    )
    assert np.sum(nums) == 1
    assert len(nums) == 2

    wrong_nums = pref_analyzer.open_response_to_number(
        ["No, I don't want to.", "I think I agree with you"],
    )

    assert len(wrong_nums) == 0


def test_binary_for_all_qs(pref_analyzer: PreferenceDataGenerator) -> None:
    """Test the ability to generate binary responses to multiple statements."""
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
    """Test the ability to generate open responses to multiple statements."""
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
    """Generate functions that create binary -> open responses."""
    pref_analyzer.generate_all_open(
        2,
        "test_open",
        Path.cwd() / "tests",
    )

    all_files = [
        Path(Path.cwd() / "tests" / f"test_open_{x}_2.json")
        for x in ["basic", "positioned", "pushy"]
    ]
    files_exists = [x.exists() for x in all_files]
    not_exist = []
    for x in all_files:
        if x.exists():
            x.unlink()
        else:
            not_exist.append(x)

    assert all(files_exists), f"These files do not exist: {x}"


def test_generate_all_binary(pref_analyzer: PreferenceDataGenerator) -> None:
    """Test all functions that generate binary (dis)agree responses."""
    pref_analyzer.generate_all_binary(
        2,
        "test_binary",
        Path.cwd() / "tests",
    )

    all_files = [
        Path(Path.cwd() / "tests" / f"test_binary_{x}_2.json")
        for x in ["basic", "positioned", "pushy", "helpful"]
    ]

    files_exists = [x.exists() for x in all_files]

    not_exist = []
    for x in all_files:
        if x.exists():
            x.unlink()
        else:
            not_exist.append(x)

    assert all(files_exists), f"The following files do not exist: {x}"
