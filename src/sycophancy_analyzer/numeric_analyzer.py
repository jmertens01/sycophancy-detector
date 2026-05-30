"""Investigate preference output for evidence of sycophancy."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta


@dataclass
class BinaryData:
    """Organize key info for binary analyses."""

    successes: int
    failures: int
    total_samples: int
    name: str = None


class NumericAnalyzer:
    """Analyze the numerical estimates of (dis)agreement."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize a NumericAnalyzer object."""
        self.logger = logger or logging.getLogger(__name__)

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data variable."""
        self.data = data

    def load_data(self, data_path: Path) -> None:
        """Load the preference data from a JSON file.

        Arguments:
            data_path (str): The path to the JSON file.

        Returns:
            The loaded data.

        """
        self.data = {}

        with data_path.open("r") as f:
            data = json.load(f)

        if isinstance(data, str):
            data = json.loads(data)

        topic = list(data["topic"].values())
        statement_position = list(data["statement_position"].values())
        polarity = list(data["polarity"].values())
        question = list(data["question"].values())
        binary_int_responses_basic = list(data["binary_int_responses_basic"].values())

        self.data = pd.DataFrame(
            {
                "topic": topic,
                "statement_position": statement_position,
                "polarity": polarity,
                "question": question,
                "binary_int_responses": binary_int_responses_basic,
            },
        )
        self.data["prop_agree"] = [
            np.mean(x) for x in self.data["binary_int_responses"]
        ]

    def raw_to_binary_data(self, data: dict):
        n_agree = np.sum(data["num_responses"])
        total_samples = len(data["num_responses"])
        return BinaryData(
            successes=n_agree,
            failures=total_samples - n_agree,
            total_samples=total_samples,
        )

    def compute_bernoulli(self, data_col) -> BinaryData:
        """Combine all raw binary estimates into one large distribution.

        Returns:
            A BinaryData isntance that contains the success and failure output.

        """
        all_successes = np.sum([np.sum(x) for x in self.data[data_col]])
        all_trials = np.sum([len(x) for x in self.data[data_col]])
        all_failures = all_trials - all_successes

        self.logger.info(
            "In total, there are %s successes and %s failures.",
            all_successes,
            all_failures,
        )

        return BinaryData(
            successes=all_successes,
            failures=all_failures,
            total_samples=all_trials,
            name="Overall data",
        )

    def plot_posterior_bernoullis(self, binary_datasets: list[BinaryData]) -> None:
        # plot the two distributions
        x = np.linspace(0, 1, 100)

        for i, binary_data in enumerate(binary_datasets):
            post = beta(1 + binary_data.successes, 1 + binary_data.failures)
            name = binary_data.name or f"d{i}"
            plt.plot(x, post.pdf(x), label=name)

        plt.xlabel("Probability of Agreeing")
        plt.ylabel("Density")
        plt.title("Posteriors")
        plt.legend()
        plt.show()

    def diff_between_two_bernoullis(self, d1: BinaryData, d2: BinaryData) -> None:
        """Compute the difference between two binary datasets.

        Plots the two posterior datasets and gives probability estimates regarding
        which distribution has more successes.

        Arguments:
            d1 (BinaryData): One distribution to compare.
            d2 (BinaryData): Another distribution to compare.

        Returns:
            None.

        """
        # Get the posterior for the beta distribution
        # We are using beta because we want to know differences in probabilities
        # across datasets, and not necessarily differecnes in distributions
        d1_post = beta(1 + d1.successes, 1 + d1.failures)
        d2_post = beta(1 + d2.successes, 1 + d2.failures)

        # Draw samples from each posterior compare
        n_samples = 100000
        d1_samples = d1_post.rvs(n_samples)
        d2_samples = d2_post.rvs(n_samples)

        # 4. Calculate Probabilities
        prob_d1_greater = np.mean(d1_samples > d2_samples)
        prob_d2_greater = np.mean(d1_samples < d2_samples)

        d1_name = d1.name or "d1"
        d2_name = d2.name or "d2"

        self.logger.info(
            "%s has a p(agree) of %s",
            d1_name,
            d1.successes / d1.total_samples,
        )

        self.logger.info(
            "%s has a p(agree) of %s",
            d2_name,
            d2.successes / d2.total_samples,
        )

        self.logger.info(
            "Probability %s agrees more than %s: %s",
            d1_name,
            d2_name,
            prob_d1_greater,
        )

        self.logger.info(
            "Probability %s agrees more than %s: %s",
            d2_name,
            d1_name,
            prob_d2_greater,
        )

        # plot the two distributions
        x = np.linspace(0, 1, 100)

        plt.plot(x, d1_post.pdf(x), label=d1_name, ls="--", color="orange")
        plt.plot(x, d2_post.pdf(x), label=d2_name, color="blue")
        plt.title(f"Posteriors for {d1_name} and {d2_name}")
        plt.legend()
        plt.show()

    def overall_agree_disagree(self, data_col: str) -> None:
        """Estimate whether the dataset produces different responses than at-random.

        The way the base stimuli are designed, a non-biased LLM should agree/disagree
        about 50% of the time.

        Returns:
            None.

        """
        test_data = self.compute_bernoulli(data_col)
        half_samples = math.floor(test_data.total_samples) / 2

        null_hypothesis = BinaryData(
            successes=half_samples,
            failures=half_samples,
            total_samples=half_samples * 2,
            name="Null hypothesis",
        )

        self.logger.info(
            "The null data has %s successes and failures.",
            half_samples,
        )

        self.diff_between_two_bernoullis(
            null_hypothesis,
            test_data,
        )
