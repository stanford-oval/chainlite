import pytest

from chainlite import get_logger, llm_generation_chain

logger = get_logger(__name__)


test_engine = "gpt-4o-openai"


@pytest.mark.asyncio(scope="session")
async def test_llm_generate():
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
