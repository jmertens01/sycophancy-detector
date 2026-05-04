from src.statement_to_query_prompts import prompt_func_dict


def test_prompt_func_dict():
    responses = []
    track_keys = []
    for key in prompt_func_dict.keys():
        for sub_key in prompt_func_dict[key].keys():
            response = prompt_func_dict[key][sub_key]("This is a good test.")
            track_keys.append(f"{key}_{sub_key}")
            responses.append(response)

    for answer, key in zip(responses, track_keys, strict=True):
        assert isinstance(answer, str), f"Issue with {key}"
        assert len(answer) > 0, f"Issue with {key}"
        assert any(x in answer.lower() for x in ["agree", "disagree"])
