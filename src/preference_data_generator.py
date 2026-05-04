"""Prompt an LLM to generate responses that may elicit sycophancy."""

import json
import logging
from pathlib import Path

import numpy as np
from ollama import ChatResponse, chat
from tqdm import tqdm

from src.statement_to_query_prompts import prompt_func_dict


class PreferenceDataGenerator:
    """Prompt an LLM in ways that will highlight sycophancy."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize a preference data generator object."""
        self.logger = logger or logging.getLogger(__name__)

    def ask_ollama(self, prompt: str) -> ChatResponse:
        """Request a response from Ollama.

        Arguments:
            prompt (str): The prompt to send ollama.

        Returns:
            Ollama's response.

        """
        response: ChatResponse = chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response

    def retrieve_n_open_responses(
        self,
        statement: str,
        n: int,
        pressure: str,
        tries_per_query: int = 3,
    ) -> list[str]:
        """Sample a set of open-ended LLM responses to a statement.

        Currently, they start with "agree" or "disagree" in some format,
        and then an explanation.

        Arguments:
            statement (str): The statement that will be turned into a statement.
            n (int): The number of times to sample the LLMs response per
               statement.
            pressure (str): The amount of pressure to apply.
            tries_per_query (int): The number of times to try to query the LLM.

        Returns:
            A list of responses, in string format.

        """
        i = 0
        responses = []
        query = prompt_func_dict["bin_open"][pressure](statement)
        while i < n:
            response = ""
            attempts = 0
            bad_output = True
            while bad_output and attempts < tries_per_query:
                raw_response = self.ask_ollama(query)
                response = raw_response.message["content"].lower().strip()
                attempts += 1
                bad_output = "agree" not in response.split(" ", maxsplit=1)[0]

            response = "" if "agree" not in response.split(" ")[0] else response
            responses.append(response)
            i += 1
        return responses

    def retrieve_n_binary_responses(
        self,
        statement: str,
        n: int,
        pressure: str,
        tries_per_query: int = 3,
    ) -> list[str]:
        """Sample a set of binary LLM responses [(dis)agree] to a statement.

        Arguments:
            statement (str): The statement that will be turned into a statement.
            n (int): The number of times to sample the LLMs response per
               statement.
            pressure (str): The amount of pressure to apply.
            tries_per_query (int): The number of times to try to query the LLM.

        Returns:
            A list of responses, in string format.

        """
        i = 0
        responses = []
        query = prompt_func_dict["binary"][pressure](statement)
        while i < n:
            response = ""
            attempts = 0
            while response not in ["agree", "disagree"] and attempts < tries_per_query:
                raw_response = self.ask_ollama(query)
                response = raw_response.message["content"].lower().strip()
                attempts += 1

            response = "" if response not in ["agree", "disagree"] else response
            responses.append(response)
            i += 1
        return responses

    def binary_response_to_number(self, input_list: list[str]) -> list[int]:
        """Convert a list of binary string responses to 1 and 0.

        Agree becomes 1, disagree becomes 0.

        Arguments:
            input_list (list[str]): A list of agree/disagrees.

        Returns:
            A list of 1s and 0s.

        """
        convert_dict = {"agree": 1, "disagree": 0}
        return [convert_dict[x] for x in input_list if x in convert_dict]

    def open_response_to_number(self, input_list: list[str]) -> list[int]:
        """Convert a list of string responses to 1 and 0.

        If the first word contains "disagree", it's a 0. If it has "agree", it's a 1.

        Arguments:
            input_list (list[str]): A list of agree/disagrees.

        Returns:
            A list of 1s and 0s.

        """
        nums = []
        for x in input_list:
            stem = x.split(" ")[0]
            if "disagree" in stem.lower():
                nums.append(0)
            elif "agree" in stem.lower():
                nums.append(1)
            else:
                self.logger.info(
                    "Error processing %s",
                    x,
                )

        return nums

    def binary_for_all_qs(self, statements: list[str], n: int, pressure: str) -> dict:
        """Generate binary responses (dis/agree) for a list of statements.

        Arguments:
            statements (list): A list of statements to convert into queries
               to prompt the LLM.
            n (int): The number of times to query the LLM, per statement.
            pressure (str): The level of pressure to put on the LLM to agree,
               must align with the prompt_func_dict in statement_to_query_prompts.py.

        Returns:
            Dictionary of the LLM's responses to each statements.

        """
        question_dict = {}

        for question in tqdm(statements, "Retrieving Ollama responses"):
            str_responses = self.retrieve_n_binary_responses(
                question,
                n=n,
                pressure=pressure,
            )
            num_responses = self.binary_response_to_number(str_responses)
            question_dict[question] = {
                "str_responses": str_responses,
                "num_responses": num_responses,
                "prop_agree": np.average(num_responses),
            }

        return question_dict

    def open_for_all_qs(self, statements: list[str], n: int, pressure: str) -> dict:
        """Generate open-ended LLM responses to a list of statements.

        Arguments:
            statements (list): A list of statements to convert into queries
               to prompt the LLM.
            n (int): The number of times to query the LLM, per statement.
            pressure (str): The level of pressure to put on the LLM to agree,
               must align with the prompt_func_dict in statement_to_query_prompts.py.

        Returns:
            Dictionary of the LLM's responses to each statements.

        """
        question_dict = {}

        for question in tqdm(statements, "Retrieving Ollama responses"):
            str_responses = self.retrieve_n_open_responses(
                question,
                n=n,
                pressure=pressure,
            )
            num_responses = self.open_response_to_number(str_responses)
            question_dict[question] = {
                "str_responses": str_responses,
                "num_responses": num_responses,
                "prop_agree": np.average(num_responses),
            }

        return question_dict

    def generate_all_open(
        self,
        statements: list[str],
        n_samples: int,
        file_base: str,
    ) -> None:
        """Generate all open responses for all statements.

        Arguments:
            statements (list[str]): A list of statements to respond to.
            n_samples (int): The number of times to sample the LLMs response per
               statement.
            file_base (str): The start of the file to produce for each output.

        Returns:
            None.

        """
        for pressure_type in prompt_func_dict["binary"]:
            open_question_dict = self.open_for_all_qs(
                statements,
                n_samples,
                pressure_type,
            )
            with Path(f"{file_base}_{pressure_type}_{n_samples}.json").open("w") as f:
                json.dump(open_question_dict, f, indent=4)

        positioned_open_dict = self.open_for_all_qs(
            statements,
            n_samples,
            "positioned",
        )
        with Path(f"{file_base}_positioned_{n_samples}.json").open("w") as f:
            json.dump(positioned_open_dict, f, indent=4)

        pushy_open_dict = self.open_for_all_qs(
            statements,
            n_samples,
            "positioned",
        )
        with Path(f"{file_base}_pushy_{n_samples}.json").open("w") as f:
            json.dump(pushy_open_dict, f, indent=4)

    def generate_all_binary(
        self,
        statements: list[str],
        n_samples: int,
        file_base: str,
    ) -> None:
        """Generate all binary responses for all statements.

        Arguments:
            statements (list[str]): A list of statements to respond to.
            n_samples (int): The number of times to sample the LLMs response per
               statement.
            file_base (str): The start of the file to produce for each output.

        Returns:
            None.

        """
        for pressure_type in prompt_func_dict["binary"]:
            binary_question_dict = self.binary_for_all_qs(
                statements,
                n_samples,
                pressure_type,
            )
            with Path(f"{file_base}_{pressure_type}_{n_samples}.json").open("w") as f:
                json.dump(binary_question_dict, f, indent=4)


def main(statements_path: Path, stem: str | None = "") -> None:
    """Generate data for preference analysis.

    Saves the data locally.

    Arguments:
        statements_path (Path): Path to a json file that contains
           the statements to use to generate prompts.
        stem (str, Optional): A stem to add to output files.

    Returns:
        None.

    """
    preference_data_generator = PreferenceDataGenerator()
    with statements_path.open("r") as f:
        all_queries = json.load(f)

    statements = []
    for query in tqdm(all_queries, "Processing json file."):
        statements.extend(query["statements"])

    n_samples = 1

    # Binary responses
    preference_data_generator.generate_all_binary(
        statements,
        n_samples,
        f"{stem}binary",
    )

    # Open responses
    preference_data_generator.generate_all_open(
        statements,
        n_samples,
        f"{stem}open",
    )


if __name__ == "__main__":
    main(Path("preference_statements.json"))
