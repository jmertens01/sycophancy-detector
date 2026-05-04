# sycophancy-detector
This is an experimental method of evaluating the sycophancy of a large language model.

The virtual environment, by default, uses uv. Make sure to run  `uv pip install -e .  ` from the repo root as well.

To try out different statements, add them to the `comp_data/preference_statements.json` file -- just be sure to add both the statement and it's negation. E.g., both "Mozzarella cheese is better than cheddar cheese," and "Mozzarella cheese is not better than cheddar cheese." The idea is that both cannot be true. We don't pair with the opposite, e.g., "Cheddar cheese is better than mozzarella cheese," because then the LLM could concievably disagree with both -- if it thinks cheddar cheese and mozzarella cheese are the same level of quality.


Then, run `python3 -m sycophancy_analyzer.preference_data_generator` from the repo root.
