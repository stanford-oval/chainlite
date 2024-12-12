import asyncio
import random
import string
import time
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

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

logger = get_logger(__name__)


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["gpt-4o-openai", "gpt-4o-azure"])
async def test_structured_output(engine: str):
    class Debate(BaseModel):
        """
        A Debate event
        """

        mention: str
        people: List[str]

    response = await llm_generation_chain(
        template_file="structured.prompt",
        engine=engine,
        max_tokens=1000,
        force_skip_cache=True,
        pydantic_class=Debate,
    ).ainvoke(
        {
            "text": "4 major candidates for California U.S. Senate seat clash in first debate"
        }
    )

    assert isinstance(response, Debate)
    assert response.mention
    assert response.people


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["gpt-4o-openai", "gpt-4o-azure"])
async def test_structured_output_engine(engine: str):
    class Debate(BaseModel):
        """
        A Debate event
        """

        mention: str
        people: List[str]

    response = await llm_generation_chain(
        template_file="structured.prompt",
        engine=engine,
        engine_for_structured_output=engine,
        max_tokens=1000,
        force_skip_cache=True,
        pydantic_class=Debate,
    ).ainvoke(
        {
            "text": "4 major candidates for California U.S. Senate seat clash in first debate"
        }
    )

    print(response)
    assert isinstance(response, Debate)
    assert response.mention
    assert response.people
