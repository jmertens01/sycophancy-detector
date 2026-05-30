"""Prompt an LLM to generate responses that may elicit sycophancy."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ollama import ChatResponse, chat
from tqdm import tqdm

from sycophancy_analyzer.statement_to_query_prompts import prompt_func_dict


class PreferenceDataGenerator:
    """Prompt an LLM in ways that will highlight sycophancy."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize a preference data generator object."""
        self.logger = logger or logging.getLogger(__name__)
        self.statements = pd.DataFrame()

    def ask_model(self, prompt: str, model: str = "llama3.1") -> ChatResponse:
        """Request a response from Ollama.

        Arguments:
            prompt (str): The prompt to send ollama.

        Returns:
            Ollama's response.

        """
        response: ChatResponse = chat(
            model=model,
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
                raw_response = self.ask_model(query)
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
        print(f"{pressure = }")
        query = prompt_func_dict["binary"][pressure](statement)
        while i < n:
            response = ""
            attempts = 0
            while response not in ["agree", "disagree"] and attempts < tries_per_query:
                raw_response = self.ask_model(query)
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

    def binary_for_all_qs(
        self,
        n: int,
        pressure: str,
    ) -> None:
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
        print(f"{pressure = }")
        all_str_responses = []
        all_bin_responses = []
        print(self.statements)
        for _, question_cat in tqdm(
            self.statements.iterrows(),
            "Retrieving Ollama responses",
        ):
            str_responses = self.retrieve_n_binary_responses(
                question_cat["question"],
                n=n,
                pressure=pressure,
            )
            all_str_responses.append(str_responses)

            num_responses = self.binary_response_to_number(str_responses)

            all_bin_responses.append(num_responses)

        self.statements[f"binary_str_responses_{pressure}"] = all_str_responses
        self.statements[f"binary_int_responses_{pressure}"] = all_bin_responses

    def open_for_all_qs(
        self,
        n: int,
        pressure: str,
    ) -> dict:
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

        for question in tqdm(self.statements, "Retrieving Ollama responses"):
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
        n_samples: int,
        file_base: str,
        output_dir: Path | None = None,
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
        output_dir = output_dir or Path.cwd()
        for pressure_type in prompt_func_dict["bin_open"]:
            open_question_dict = self.open_for_all_qs(
                n_samples,
                pressure_type,
            )
            with Path(
                output_dir / f"{file_base}_{pressure_type}_{n_samples}.json",
            ).open("w") as f:
                json.dump(open_question_dict, f, indent=4)

    def generate_all_binary(
        self,
        n_samples: int,
        output_dir: Path,
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
            self.binary_for_all_qs(
                n_samples,
                pressure_type,
            )
            intermed = self.statements.to_json()

            with Path(
                output_dir / f"binary_{n_samples}_{pressure_type}_intermed.json",
            ).open("w") as f:
                json.dump(intermed, f, indent=4)

    def json_to_statements(self, data) -> None:
        topic_list = []
        position_list = []
        q_type_list = []
        q_list = []

        for q_type in data:
            topic = q_type["topic"]
            positions = q_type["positions"]
            for position in positions:
                negative_q = positions[position]["negative"]
                topic_list.append(topic)
                position_list.append(position)
                q_type_list.append("negative")
                q_list.append(negative_q)

                positive_q = positions[position]["positive"]
                topic_list.append(topic)
                position_list.append(position)
                q_type_list.append("negative")
                q_list.append(positive_q)

        self.statements = pd.DataFrame(
            {
                "topic": topic_list,
                "statement_position": position_list,
                "polarity": q_type_list,
                "question": q_list,
            },
        )

    def read_json_statements(self, statements_path: Path) -> None:
        """Load and add statements from json file."""
        with statements_path.open("r") as f:
            file = json.load(f)
        self.json_to_statements(file)

    def compare_same_query(
        self,
        query: str,
        n_samples: int = 5,
    ) -> pd.DataFrame:
        direct = self.run_one_usr_query_bin(
            query=query, pressure="basic", samples=n_samples
        )
        direct_num = self.binary_response_to_number(direct)

        positioned = self.run_one_usr_query_bin(
            query=query, pressure="positioned", samples=n_samples
        )
        positioned_num = self.binary_response_to_number(positioned)

        pushy = self.run_one_usr_query_bin(
            query=query, pressure="pushy", samples=n_samples
        )
        pushy_num = self.binary_response_to_number(pushy)

        we = self.run_one_usr_query_bin(query=query, pressure="we", samples=n_samples)
        we_num = self.binary_response_to_number(we)

        return pd.DataFrame(
            {
                "type": ["direct", "positioned", "pushy", "we"],
                "proportion": [
                    np.mean(direct_num),
                    np.mean(positioned_num),
                    np.mean(pushy_num),
                    np.mean(we_num),
                ],
            },
        )

    def run_one_usr_query_bin(
        self,
        query: str,
        pressure: str,
        samples: int = 10,
    ):
        """Run binary response for one user query."""
        answers = self.retrieve_n_binary_responses(
            query,
            samples,
            pressure,
        )
        return answers


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
    preference_data_generator.read_json_statements(statements_path)

    n_samples = 30
    output_dir = Path.cwd() / "comp_data"

    # Binary responses
    preference_data_generator.generate_all_binary(
        n_samples,
        f"{stem}binary",
        output_dir,
    )

    json_data = preference_data_generator.statements.to_json()

    with Path(
        output_dir / f"all_binary_{n_samples}.json",
    ).open("w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main(Path("comp_data/preference_statements.json"))
