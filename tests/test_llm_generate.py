import pytest

from chainlite import llm_generation_chain, load_config_from_file
from chainlite.utils import get_logger

logger = get_logger(__name__)


@pytest.mark.asyncio(scope="session")
async def test_llm_generate():
    load_config_from_file("./chainlite_config.yaml")

    response = await llm_generation_chain(
        template_file="tests/test.prompt",
        engine="gpt-35-turbo",
        max_tokens=100,
    ).ainvoke({})
    logger.info(response)

    assert response is not None, "The response should not be None"
    assert isinstance(response, str), "The response should be a string"
    assert len(response) > 0, "The response should not be empty"
