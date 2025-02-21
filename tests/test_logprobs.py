import random
import string
import time
import pytest

from chainlite import get_logger, llm_generation_chain, get_total_cost

logger = get_logger(__name__)


test_engine = "gpt-4o-openai"


@pytest.mark.asyncio(scope="session")
async def test_llm_generate_with_logprobs():
    response, logprobs = await llm_generation_chain(
        template_file="test.prompt",  # prompt path relative to one of the paths specified in `prompt_dirs`
        engine=test_engine,
        max_tokens=5,
        force_skip_cache=True,
        return_top_logprobs=10,
    ).ainvoke({})

    assert response is not None, "The response should not be None"
    assert isinstance(response, str), "The response should be a string"
    assert len(response) > 0, "The response should not be empty"

    assert len(logprobs) == 5
    for i in range(len(logprobs)):
        assert "top_logprobs" in logprobs[i]
        assert len(logprobs[i]["top_logprobs"]) == 10


@pytest.mark.asyncio(scope="session")
async def test_logprob_cache():
    c = llm_generation_chain(
        template_file="tests/copy.prompt",
        engine=test_engine,
        max_tokens=1,
        temperature=0.0,
        return_top_logprobs=20,
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
