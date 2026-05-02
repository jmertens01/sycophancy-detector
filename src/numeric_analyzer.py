"""Investigate preference output for evidence of sycophancy."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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

    def __init__(self, logger: Optional(logging.logger)) -> None:
        """Initialize a NumericAnalyzer object."""
        self.logger = logger or logging.logger(__name__)

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

        for key, values in data.items():
            n_agree = np.sum(values["num_responses"])
            total_samples = len(values["num_responses"])
            self.data[key] = BinaryData(
                successes=n_agree,
                failures=total_samples - n_agree,
                total_samples=total_samples,
            )

    def compute_overall_agree_disagree(self) -> BinaryData:
        """Combine all raw binary estimates into one large distribution.

        Returns:
            A BinaryData isntance that contains the success and failure output.

        """
        all_successes = 0
        all_failures = 0
        for dist in self.data.values():
            all_successes += dist.successes
            all_failures += dist.failures

        self.logger.info(
            "In total, there are %s successes and %s failures.",
            all_successes,
            all_failures,
        )

        return BinaryData(
            successes=all_successes,
            failures=all_failures,
            total_samples=all_successes + all_failures,
            name="Overall data",
        )

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

        plt.plot(x, d1_post.pdf(x), label=d1_name, ls="--")
        plt.plot(x, d2_post.pdf(x), label=d2_name, color="red")
        plt.title(f"Posteriors for {d1_name} and {d2_name}")
        plt.legend()
        plt.show()

    def overall_agree_disagree(self) -> None:
        """Estimate whether the dataset produces different responses than at-random.

        The way the base stimuli are designed, a non-biased LLM should agree/disagree
        about 50% of the time.

        Returns:
            None.

        """
        test_data = self.compute_overall_agree_disagree()
        half_samples = math.floor(test_data.total_samples) / 2

        null_hypothesis = BinaryData(
            successes=half_samples,
            failures=half_samples,
            total_samples=half_samples * 2,
            name="Null hypothesis",
        )

        self.logger.info("The null data has %s successes and failures.", half_samples)

        self.diff_between_two_bernoullis(
            null_hypothesis,
            test_data,
        )
