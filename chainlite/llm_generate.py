import json
import os
import random
import re
from datetime import datetime
import litellm
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable, chain
from tqdm.auto import tqdm
from pydantic import BaseModel

from chainlite.llm_config import GlobalVars, load_config_from_file

from .chat_lite_llm import ChatLiteLLM
from .load_prompt import load_fewshot_prompt_template
from .utils import get_logger

logger = get_logger(__name__)

# This regex pattern aims to capture up through the last '.', '!', '?',
# and includes an optional ending quotation mark '"'.
# It uses a lookahead to ensure it captures until the last such punctuation
# possibly followed by a quotation mark.
partial_sentence_regex = re.compile(r'([\s\S]*?[.!?]"?)(?=(?:[^.!?]*$))')


def is_same_prompt(template_name_1: str, template_name_2: str) -> bool:
    return os.path.basename(template_name_1) == os.path.basename(template_name_2)


def write_prompt_logs_to_file(
    log_file: Optional[str] = None,
    append: bool = False,
    include_timestamp: bool = False,
):
    if not log_file:
        log_file = GlobalVars.prompt_log_file
    key_order = [
        "template_name",
        "instruction",
        "input",
        "output",
    ]  # specifies the sort order of keys in the output, for a better viewing experience
    if include_timestamp:
        key_order = ["datetime"] + key_order

    mode = "w"
    if append:
        mode = "a"
    with open(log_file, mode) as f:
        for item in GlobalVars.prompt_logs.values():
            should_skip = False
            for t in GlobalVars.prompts_to_skip_for_debugging:
                if is_same_prompt(t, item["template_name"]):
                    should_skip = True
                    break
            if should_skip:
                continue
            if "output" not in item:
                # happens if the code crashes in the middle of a an LLM call
                continue
            if include_timestamp:
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                item["datetime"] = datetime_str

            f.write(
                json.dumps(
                    {key: item[key] for key in key_order},
                    ensure_ascii=False,
                )
            )
            f.write("\n")


class ChainLogHandler(AsyncCallbackHandler):
    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        run_id = str(parent_run_id)
        distillation_instruction = (
            metadata["distillation_instruction"]
            if metadata["distillation_instruction"]
            else "<no distillation instruction is specified for this prompt>"
        )
        llm_input = messages[0][-1].content
        if messages[0][-1].type == "system":
            # it means the prompt did not have an `# input` block, and only has an instruction block
            llm_input = ""
        if run_id not in GlobalVars.prompt_logs:
            GlobalVars.prompt_logs[run_id] = {}
        GlobalVars.prompt_logs[run_id]["instruction"] = distillation_instruction
        GlobalVars.prompt_logs[run_id]["input"] = llm_input
        GlobalVars.prompt_logs[run_id]["template_name"] = metadata["template_name"]

    async def on_chain_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        run_id = str(run_id)
        if run_id in GlobalVars.prompt_logs:
            # this is the final response in the entire chain
            if (
                isinstance(response, tuple)
                and len(response) == 2
                and isinstance(response[1], ToolOutput)
            ):
                response = list(response)
                response[1] = str(response[1])
            elif isinstance(response, ToolOutput):
                response = str(response)
            if isinstance(response, tuple) and len(response) == 2:
                response = list(response)
                # if exactly one is not None/empty, then we want to log that one
                if response[0] and not response[1]:
                    response = response[0]
                elif not response[0] and response[1]:
                    response = response[1]
            GlobalVars.prompt_logs[run_id][
                "output"
            ] = (
                response.__repr__()
            )  # convert to str because output might be a Pydantic object (if `pydantic_class` is provided in `llm_generation_chain()`)


class ProgbarHandler(AsyncCallbackHandler):
    def __init__(self, desc: str):
        super().__init__()
        self.count = 0
        self.desc = desc

    # Override on_llm_end method. This is called after every response from LLM
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.count == 0:
            self.progress_bar = tqdm(
                total=None,
                desc=self.desc,
                unit=" LLM Calls",
                bar_format="{desc}: {n_fmt}{unit} ({rate_fmt})",
                mininterval=0,
                position=0,
                leave=True,
            )  # define a progress bar
        self.count += 1
        self.progress_bar.update(1)


@chain
def string_to_pydantic_object(llm_output: str, pydantic_class: BaseModel):
    try:
        return pydantic_class.model_validate(json.loads(llm_output))
    except Exception as e:
        logger.exception(
            f"Error decoding JSON: {e}. This might be resolved by increasing `max_tokens`"
        )
        logger.error(f"LLM output: {llm_output}")
        return None


def is_list(obj):
    return isinstance(obj, list)


def is_dict(obj):
    return isinstance(obj, dict)


def _ensure_strict_json_schema(
    json_schema: object,
    path: tuple[str, ...],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.

    Adapted from OpenAI's Python client code
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key)
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"))

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)))
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        json_schema["allOf"] = [
            _ensure_strict_json_schema(entry, path=(*path, "anyOf", str(i)))
            for i, entry in enumerate(all_of)
        ]

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name))

    return json_schema


class ToolOutput(BaseModel):
    function: Callable
    kwargs: dict

    def __repr__(self):
        return (
            f"{self.function.__name__}("
            + ", ".join([f"{k}={repr(v)}" for k, v in self.kwargs.items()])
            + ")"
        )


@chain
async def return_response_and_tool(
    llm_output, tools: list[Callable], force_tool_calling: bool
):
    response = await StrOutputParser().ainvoke(input=llm_output)
    tool_output_in_json_format = llm_output.tool_calls

    tool_outputs = []
    for t in tool_output_in_json_format:
        tool_name = t["name"]
        matching_tool = next(
            (tool for tool in tools if tool.__name__ == tool_name), None
        )
        if matching_tool:
            tool_outputs.append(ToolOutput(function=matching_tool, kwargs=t["args"]))
    if force_tool_calling:
        return tool_outputs
    return response, tool_outputs

@chain
async def return_response_and_logprobs(llm_output):
    response = await StrOutputParser().ainvoke(input=llm_output)
    return response, llm_output.response_metadata.get("logprobs")

def llm_generation_chain(
    template_file: str,
    engine: str,
    max_tokens: int,
    temperature: float = 0.0,
    stop_tokens: Optional[List[str]] = None,
    top_p: float = 0.9,
    output_json: bool = False,
    pydantic_class: BaseModel = None,
    template_blocks: list[tuple[str]] = None,
    keep_indentation: bool = False,
    progress_bar_desc: Optional[str] = None,
    additional_postprocessing_runnable: Runnable = None,
    tools: Optional[list[Callable]] = None,
    force_tool_calling: bool = False,
    return_top_logprobs: int = 0,
    bind_prompt_values: Dict = {},
    force_skip_cache: bool = False,
) -> Runnable:
    """
    Constructs a LangChain generation chain for LLM response utilizing LLM APIs prescribed in the ChainLite config file.

    Parameters:
        template_file (str): The path to the generation template file. Must be a .prompt file.
        engine (str): The language model engine to employ. An engine represents the left-hand value of an `engine_map` in the ChainLite config file.
        max_tokens (int): The upper limit of tokens the LLM can generate.
        temperature (float, optional): Dictates the randomness in the generation. Must be >= 0.0. Defaults to 0.0 (deterministic).
        stop_tokens (List[str], optional): The list of tokens causing the LLM to stop generating. Defaults to None.
        top_p (float, optional): The max cumulative probability for nucleus sampling, must be within 0.0 - 1.0. Defaults to 0.9.
        output_json (bool, optional): If True, asks the LLM API to output a JSON. This depends on the underlying model to support.
            For example, GPT-4, GPT-4o and newer GPT-3.5-Turbo models support it, but require the word "json" to be present in the input. Defaults to False.
        pydantic_class (BaseModel, optional): If provided, will parse the output to match this Pydantic class. Only models like gpt-4o-mini and gpt-4o-2024-08-06
            and newer are supported
        template_blocks: If provided, will use this instead of `template_file`. The format is [(role, string)] where role is one of "instruction", "input", "output"
        keep_indentation (bool, optional): If True, will keep indentations at the beginning of each line in the template_file. Defaults to False.
        progress_bar_name (str, optional): If provided, will display a `tqdm` progress bar using this name
        additional_postprocessing_runnable (Runnable, optional): If provided, will be applied to the output of LLM generation, and the final output will be logged
        tools (List[Callable], optional): If provided, will be made available to the underlying LLM, to optionally output it for function calling. Defaults to None.
        force_tool_calling (bool, optional): If True, will force the LLM to output the tools for function calling. Defaults to False.
        return_top_logprobs (int, optional): If > 0, will return the top logprobs for each token, so the output will be Tuple[str, dict]. Defaults to 0.
        bind_prompt_values (Dict, optional): A dictionary containing {Variable: str : Value}. Binds values to the prompt. Additional variables can be provided when the chain is called. Defaults to {}.
        force_skip_cache (bool, optional): If True, will force the LLM to skip the cache, and the new value won't be saved in cache either. Defaults to False.

    Returns:
        Runnable: The language model generation chain

    Raises:
        IndexError: Raised when no engine matches the provided string in the LLM APIs configured, or the API key is not found.
    """

    load_config_from_file()
    if not GlobalVars.all_llm_endpoints:
        raise ValueError(
            "No LLM API found. Make sure configuration and API_KEY files are set correctly, and that load_config_from_file() is called before using any other function."
        )

    # Decide which LLM resource to send this request to.
    potential_llm_resources = [
        resource
        for resource in GlobalVars.all_llm_endpoints
        if engine in resource["engine_map"]
    ]
    if len(potential_llm_resources) == 0:
        raise IndexError(
            f"Could not find any matching engines for {engine}. Please check that llm_config.yaml is configured correctly and that the API key is set in the terminal before running this script."
        )
    
    if (
        sum(
            [
                bool(pydantic_class),
                bool(output_json),
                bool(tools),
                return_top_logprobs > 0,
            ]
        )
        > 1
    ):
        raise ValueError(
            "At most one of `pydantic_class`, `output_json`, `return_top_logprobs` and `tools` can be used."
        )
    llm_resource = random.choice(potential_llm_resources)

    model = llm_resource["engine_map"][engine]

    # ChatLiteLLM expects these parameters in a separate dictionary for some reason
    model_kwargs = {}

    # TODO move these to ChatLiteLLM
    if engine in GlobalVars.local_engine_set:
        if temperature > 0:
            model_kwargs["do_sample"] = True
        else:
            model_kwargs["do_sample"] = False
        if top_p == 1:
            top_p = None

    if model.startswith("mistral/"):
        # Mistral API expects top_p to be 1 when greedy decoding
        if temperature == 0:
            top_p = 1

    is_distilled = (
        "prompt_format" in llm_resource and llm_resource["prompt_format"] == "distilled"
    )

    prompt, distillation_instruction = load_fewshot_prompt_template(
        template_file,
        template_blocks,
        is_distilled=is_distilled,
        keep_indentation=keep_indentation,
    )
    if output_json:
        model_kwargs["response_format"] = {"type": "json_object"}
    elif pydantic_class:
        model_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "schema": _ensure_strict_json_schema(
                    pydantic_class.model_json_schema(), path=()
                ),
                "name": pydantic_class.__name__,
                "strict": True,
            },
        }

    if return_top_logprobs > 0:
        model_kwargs["logprobs"] = True
        model_kwargs["top_logprobs"] = return_top_logprobs

    if tools:
        function_json = [
            {"type": "function", "function": litellm.utils.function_to_dict(t)}
            for t in tools
        ]
        model_kwargs["tools"] = function_json
        model_kwargs["tool_choice"] = "required" if force_tool_calling else "auto"

    callbacks = []
    if progress_bar_desc:
        cb = ProgbarHandler(progress_bar_desc)
        callbacks.append(cb)

    should_cache = (temperature == 0) and not force_skip_cache
    llm = ChatLiteLLM(
        model=model,
        api_base=llm_resource["api_base"] if "api_base" in llm_resource else None,
        api_key=llm_resource["api_key"] if "api_key" in llm_resource else None,
        api_version=(
            llm_resource["api_version"] if "api_version" in llm_resource else None
        ),
        cache=should_cache,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_tokens,
        callbacks=callbacks,
        model_kwargs=model_kwargs,
    )

    if len(bind_prompt_values) > 0:
        prompt = prompt.partial(**bind_prompt_values)

    llm_generation_chain = prompt | llm
    if tools:
        llm_generation_chain = llm_generation_chain | return_response_and_tool.bind(
            tools=tools, force_tool_calling=force_tool_calling
        )
    else:
        if return_top_logprobs > 0:
            llm_generation_chain = llm_generation_chain | return_response_and_logprobs
        else:
            llm_generation_chain = llm_generation_chain | StrOutputParser()

    if pydantic_class:
        llm_generation_chain = llm_generation_chain | string_to_pydantic_object.bind(
            pydantic_class=pydantic_class
        )
    

    if additional_postprocessing_runnable:
        llm_generation_chain = llm_generation_chain | additional_postprocessing_runnable
    return llm_generation_chain.with_config(
        callbacks=[ChainLogHandler()],
        metadata={
            "distillation_instruction": distillation_instruction,
            "template_name": os.path.basename(template_file),
        },
    )  # for logging to file
