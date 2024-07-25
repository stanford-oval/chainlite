from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from langchain_core.runnables import RunnableLambda

from chainlite import (
    get_logger,
    llm_generation_chain,
    load_config_from_file,
    write_prompt_logs_to_file,
    get_all_configured_engines,
    register_prompt_constants,
)
from chainlite.llm_config import GlobalVars

logger = get_logger(__name__)


# load_config_from_file("./llm_config.yaml")

chain_inputs = [
    {"topic": "Ice cream"},
    {"topic": "Cats"},
    {"topic": "Dogs"},
    {"topic": "Rabbits"},
]

test_engine = "gpt-4o-mini"


@pytest.mark.asyncio(scope="session")
async def test_llm_generate():
    logger.info("All registered engines: %s", str(get_all_configured_engines()))

    # Check that the config file has been loaded properly
    assert GlobalVars.all_llm_endpoints
    assert GlobalVars.prompt_dirs
    assert GlobalVars.prompt_log_file
    # assert GlobalVars.prompts_to_skip_for_debugging
    assert GlobalVars.local_engine_set

    response = await llm_generation_chain(
        template_file="test.prompt",  # prompt path relative to one of the paths specified in `prompt_dirs`
        engine=test_engine,
        max_tokens=100,
    ).ainvoke({})
    # logger.info(response)

    assert response is not None, "The response should not be None"
    assert isinstance(response, str), "The response should be a string"
    assert len(response) > 0, "The response should not be empty"


@pytest.mark.asyncio(scope="session")
async def test_readme_example():
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=test_engine,
        max_tokens=100,
        temperature=0.1,
        progress_bar_desc="test1",
        additional_postprocessing_runnable=RunnableLambda(lambda x: x[:5]),
    ).ainvoke({"topic": "Life as a PhD student"})


@pytest.mark.asyncio(scope="session")
async def test_constants():
    pacific_zone = ZoneInfo("America/Los_Angeles")
    today = datetime.now(pacific_zone).date().strftime("%B %d, %Y")  # e.g. May 30, 2024
    response = await llm_generation_chain(
        template_file="tests/constants.prompt",
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"question": "what is today's date?"})
    assert today in response

    # overwrite "today"
    register_prompt_constants({"today": "Thursday"})
    response = await llm_generation_chain(
        template_file="tests/constants.prompt",
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"question": "what day of the week is today?"})
    assert "thursday" in response.lower()


@pytest.mark.asyncio(scope="session")
async def test_batching():
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=test_engine,
        max_tokens=100,
        temperature=0.1,
        progress_bar_desc="test2",
    ).abatch(chain_inputs)
    assert len(response) == len(chain_inputs)

    write_prompt_logs_to_file("tests/llm_input_outputs.jsonl")


@pytest.mark.asyncio(scope="session")
async def test_mock_llm():
    c = llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=test_engine,
        max_tokens=100,
        temperature=0.1,
        progress_bar_desc="test2",
        mock=True,
    )
    output = await c.abatch(chain_inputs)
    assert output == [""] * len(chain_inputs)
    output = await c.ainvoke(chain_inputs[0])
    assert output == ""
    write_prompt_logs_to_file("tests/llm_input_outputs.jsonl")
