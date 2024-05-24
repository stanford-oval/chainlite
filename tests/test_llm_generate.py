import pytest

from chainlite import llm_generation_chain, load_config_from_file
from chainlite.llm_generate import write_prompt_logs_to_file
from chainlite.utils import get_logger

logger = get_logger(__name__)


load_config_from_file("./chainlite_config.yaml")

@pytest.mark.asyncio(scope="session")
async def test_llm_generate():

    response = await llm_generation_chain(
        template_file="test.prompt", # prompt path relative to one of the paths specified in `prompt_dirs`
        engine="gpt-35-turbo",
        max_tokens=100,
    ).ainvoke({})
    logger.info(response)

    assert response is not None, "The response should not be None"
    assert isinstance(response, str), "The response should be a string"
    assert len(response) > 0, "The response should not be empty"


@pytest.mark.asyncio(scope="session")
async def test_readme_example():
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine="gpt-35-turbo",
        max_tokens=100,
    ).ainvoke({"topic": "Life as a PhD student"})
    print(response)

    write_prompt_logs_to_file("llm_input_outputs.jsonl")