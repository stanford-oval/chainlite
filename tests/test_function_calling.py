import pytest

from chainlite import llm_generation_chain, write_prompt_logs_to_file
from chainlite.llm_generate import ToolOutput


def get_current_weather(location: str):
    """
    Get the current weather in a given location.

    Parameters
    ----------
    location : str
        The location for which to get the current weather.

    Returns
    -------
    str
        A string describing the current weather in the specified location.
    """

    if "boston" in location.lower():
        return "The weather is 12F"


def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["gpt-4o-openai", "gpt-4o-azure"])
async def test_function_calling(engine):
    test_tool_chain = llm_generation_chain(
        "tool.prompt",
        engine=engine,
        max_tokens=100,
        tools=[get_current_weather, add],
        force_skip_cache=True,
    )
    # No function calling done, just output text
    text_output, tool_outputs = await test_tool_chain.ainvoke(
        {"message": "What tools do you have available?"}
    )
    assert "weather" in text_output.lower()
    assert "add" in text_output.lower()
    assert tool_outputs == []

    # Function calling needed
    text_output, tool_outputs = await test_tool_chain.ainvoke(
        {"message": "What is the weather like in Boston  ?"}
    )

    assert text_output == ""
    assert str(tool_outputs) == "[get_current_weather(location='Boston')]"

    text_output, tool_outputs = await test_tool_chain.ainvoke(
        {"message": "What 1021 + 9573?"}
    )
    assert text_output == ""
    assert str(tool_outputs) == "[add(a=1021, b=9573)]"

    write_prompt_logs_to_file("tests/llm_input_outputs.jsonl")


@pytest.mark.asyncio(scope="session")
@pytest.mark.parametrize("engine", ["gpt-4o-openai", "gpt-4o-azure"])
async def test_forced_function_calling(engine):
    test_tool_chain = llm_generation_chain(
        "tool.prompt",
        engine=engine,
        max_tokens=100,
        tools=[get_current_weather, add],
        force_skip_cache=True,
        force_tool_calling=True,
    )

    # Forcing function call when it is already needed
    tool_outputs = await test_tool_chain.ainvoke(
        {"message": "What is the weather like in New York City?"}
    )

    assert isinstance(tool_outputs, list)
    assert str(tool_outputs) == "[get_current_weather(location='New York City')]"
    print(tool_outputs)

    # Forcing function call when it is not needed
    tool_outputs = await test_tool_chain.ainvoke({"message": "What is your name?"})
    print(tool_outputs)
    assert isinstance(tool_outputs, list)
    assert len(tool_outputs) > 0
    assert isinstance(tool_outputs[0], ToolOutput)

    write_prompt_logs_to_file("tests/llm_input_outputs.jsonl")
