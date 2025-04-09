import asyncio
import random
import string
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from langchain_core.runnables import RunnableLambda

from chainlite import (
    get_all_configured_engines,
    get_logger,
    get_total_cost,
    llm_generation_chain,
    register_prompt_constants,
    write_prompt_logs_to_file,
)
from chainlite.llm_config import GlobalVars
from chainlite.utils import run_async_in_parallel
import time
import subprocess
import sys
import pytest

logger = get_logger(__name__)


test_engine = "gpt-4o-openai"


def test_chainlite_import_time():
    # Measure the import time of the chainlite module in a subprocess,
    # using Python's -X importtime flag to get detailed import timing.
    script = "import chainlite"
    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", script],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start

    # Check that the import was successful.
    assert result.returncode == 0, f"Chainlite import failed: {result.stderr}"

    # Set a threshold for the import time (e.g., 0.1 second).
    threshold = 1
    if elapsed >= threshold:
        print("Import took too long. Breakdown of import times:")
        print(result.stderr)
        # Write the detailed trace to disk.
        with open("chainlite_import_trace.log", "w") as f:
            f.write(result.stderr)

    assert (
        elapsed < threshold
    ), f"Importing chainlite took too long: {elapsed:.2f} seconds"
    logger.info(f"Importing chainlite took {elapsed:.2f} seconds")


@pytest.fixture(scope="session", autouse=True)
def run_after_all_tests():
    """
    This fixture will run after all tests in the session.
    """
    yield  # This ensures that the fixture runs after all tests are done.

    write_prompt_logs_to_file("tests/llm_input_outputs.jsonl")
    with open("tests/llm_input_outputs.jsonl", "r") as f:
        prompt_logs = f.read()

    print(prompt_logs)
    assert (
        "test.prompt" not in prompt_logs
    ), "test.prompt is in the prompts_to_skip and therefore should not be logged"
    logger.info(f"Total LLM cost: ${get_total_cost():.1f}")


@pytest.mark.asyncio(scope="session")
async def test_instruction_with_variable():
    # Test a prompt where the instruction block contains a variable.
    template_blocks = [
        ("instruction", "You are a chatbot named {{ name }}."),
        ("input", "{{ message }}"),
    ]
    # Invoke the chain with variables to be interpolated.
    variables = {"name": "ChainLite", "message": "what is your name?"}
    response = await llm_generation_chain(
        template_file="",
        template_blocks=template_blocks,
        engine=test_engine,
        max_tokens=50,
        temperature=0,
    ).ainvoke(variables)

    assert "chainlite" in response.lower()


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
        force_skip_cache=True,
    ).ainvoke({})
    # logger.info(response)

    assert response is not None, "The response should not be None"
    assert isinstance(response, str), "The response should be a string"
    assert len(response) > 0, "The response should not be empty"


@pytest.mark.asyncio(scope="session")
async def test_string_prompts():
    response = await llm_generation_chain(
        template_file="",
        template_blocks=[
            ("instruction", "X = 1, Y = 6."),
            ("input", "what is X?"),
            ("output", "The value of X is one"),
            ("input", "what is {{ variable }}?"),
        ],
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"variable": "Y"})
    assert "The value of Y is six" in response

    # Without instruction block
    response = await llm_generation_chain(
        template_file="",
        template_blocks=[
            ("input", "what is X?"),
            ("output", "The value of X is one"),
            ("input", "what is {{ variable }}?"),
        ],
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"variable": "Y"})


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["gpt-4o-openai", "gpt-4o-azure"])
async def test_llm_examples(engine):
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=engine,
        max_tokens=100,
        temperature=0.1,
        progress_bar_desc="test1",
        additional_postprocessing_runnable=RunnableLambda(lambda x: x[:5]),
    ).ainvoke({"topic": "Life as a PhD student"})

    assert isinstance(response, str)
    assert len(response) == 5


@pytest.mark.asyncio(scope="session")
async def test_constants():
    pacific_zone = ZoneInfo("America/Los_Angeles")
    today = datetime.now(pacific_zone).date().strftime("%B %d, %Y")  # e.g. May 30, 2024
    response = await llm_generation_chain(
        template_file="tests/constants.prompt",
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"question": "What is today's date?"})
    assert today in response

    # overwrite "today"
    register_prompt_constants({"today": "Thursday"})
    response = await llm_generation_chain(
        template_file="tests/constants.prompt",
        engine=test_engine,
        max_tokens=10,
        temperature=0,
    ).ainvoke({"question": "What day of the week is today?"})
    assert "thursday" in response.lower()


@pytest.mark.asyncio(scope="session")
async def test_batching():
    chain_inputs = [
        {"topic": "Ice cream"},
        {"topic": "Cats"},
        {"topic": "Dogs"},
        {"topic": "Rabbits"},
    ]
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=test_engine,
        max_tokens=10,
        temperature=0.1,
        progress_bar_desc="test2",
    ).abatch(chain_inputs)
    assert len(response) == len(chain_inputs)


@pytest.mark.asyncio(scope="session")
async def test_cached_batching():
    c = llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=test_engine,
        max_tokens=100,
        temperature=0.0,
        progress_bar_desc="test2",
    )
    await c.ainvoke({"topic": "Ice cream"})
    first_cost = get_total_cost()
    start_time = time.time()
    response = await c.abatch([{"topic": "Ice cream"}] * 200)
    elapsed_time = time.time() - start_time
    assert (
        elapsed_time < 1
    ), f"The batched LLM calls should be cached and therefore very fast, but took {elapsed_time:.2f} seconds"
    assert (
        get_total_cost() == first_cost
    ), "The cost should not change after a cached batched LLM call"


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["o1", "o3-mini", "o3-mini-azure"])
async def test_reasoning_models(engine):
    response = await llm_generation_chain(
        template_file="tests/joke.prompt",
        engine=engine,
        max_tokens=2000,
        temperature=0.01,
    ).ainvoke({"topic": "A strawberry."})
    assert response


@pytest.mark.asyncio(scope="session")
async def test_cache():
    c = llm_generation_chain(
        template_file="tests/copy.prompt",
        engine=test_engine,
        max_tokens=100,
        temperature=0.0,
    )
    # use random input so that the first call is not cached
    start_time = time.time()
    random_input = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    response1 = await c.ainvoke({"input": random_input})
    first_time = time.time() - start_time
    first_cost = get_total_cost()

    print("First call took {:.2f} seconds".format(first_time))
    print("Total cost after first call: ${:.10f}".format(first_cost))

    start_time = time.time()
    response2 = await c.ainvoke({"input": random_input})
    second_time = time.time() - start_time
    print("Second call took {:.2f} seconds".format(second_time))
    second_cost = get_total_cost()
    print("Total cost after second call: ${:.10f}".format(second_cost))

    assert response1 == response2
    assert (
        second_time < first_time * 0.5
    ), "The second (cached) LLM call should be much faster than the first call"
    assert first_cost > 0, "The cost should be greater than 0"
    assert (
        second_cost == first_cost
    ), "The cost should not change after a cached LLM call"


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["o3-mini", "o3-mini-azure"])
async def test_reasoning_effort_cache(engine: str):
    c1 = llm_generation_chain(
        template_file="tests/copy.prompt",
        engine=engine,
        max_tokens=1000,
        temperature=0.0,
        reasoning_effort="low",
    )

    c2 = llm_generation_chain(
        template_file="tests/copy.prompt",
        engine=engine,
        max_tokens=1000,
        temperature=0.0,
        reasoning_effort="medium",
    )
    # use random input so that the first call is not cached
    start_time = time.time()
    random_input = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    response1 = await c1.ainvoke({"input": random_input})
    first_time = time.time() - start_time
    first_cost = get_total_cost()

    print("First call took {:.2f} seconds".format(first_time))
    print("Total cost after first call: ${:.10f}".format(first_cost))

    start_time = time.time()
    response2 = await c2.ainvoke({"input": random_input})
    second_time = time.time() - start_time
    print("Second call took {:.2f} seconds".format(second_time))
    second_cost = get_total_cost()
    print("Total cost after second call: ${:.10f}".format(second_cost))

    assert response1 == response2
    assert (
        second_time > first_time * 0.5
    ), "The different reasoning efforts should not be cached"
    assert first_cost > 0, "The cost should be greater than 0"
    assert (
        second_cost > first_cost
    ), "The cost should increase after a different reasoning effort LLM call"

    # another call to c1 should be cached
    start_time = time.time()
    response3 = await c1.ainvoke({"input": random_input})
    third_time = time.time() - start_time
    third_cost = get_total_cost()
    print("Third call took {:.2f} seconds".format(third_time))
    print("Total cost after third call: ${:.10f}".format(third_cost))
    assert response1 == response3
    assert (
        third_time < first_time * 0.5
    ), "The third (cached) LLM call should be much faster than the first call"
    assert (
        third_cost == second_cost
    ), "The cost should not change after a cached LLM call"


@pytest.mark.asyncio(scope="session")
async def test_run_async_in_parallel():

    async def async_function(i, j):
        await asyncio.sleep(1)
        return i

    test_inputs1 = range(10)
    test_inputs2 = range(10, 20)
    max_concurrency = 5
    desc = "test"
    ret = await run_async_in_parallel(
        async_function,
        test_inputs1,
        test_inputs2,
        max_concurrency=max_concurrency,
        desc=desc,
    )
    assert ret == list(test_inputs1)


@pytest.mark.asyncio(scope="session")
async def test_o1_reasoning_effort():
    for reasoning_effort in ["low", "medium", "high"]:
        start_time = time.time()
        response = await llm_generation_chain(
            template_file="tests/reasoning.prompt",
            engine="o1",
            max_tokens=2000,
            force_skip_cache=True,
            reasoning_effort=reasoning_effort,
        ).ainvoke({})
        print(response)
        print(
            f"Reasoning effort: {reasoning_effort}, Time taken: {time.time() - start_time}"
        )
        assert response
