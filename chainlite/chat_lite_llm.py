"""Wrapper around LiteLLM. Modified from https://python.langchain.com/api_reference/_modules/langchain_community/chat_models/litellm.html#ChatLiteLLM"""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import warnings
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolCallChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from tenacity import (
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from chainlite.llm_config import GlobalVars
from chainlite.llm_output import ToolOutput

logger = logging.getLogger(__name__)


class ChatLiteLLMException(Exception):
    """Error with the `LiteLLM I/O` library"""


def _create_retry_decorator(llm) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, configured to handle LLM exceptions"""
    import litellm

    errors = [
        litellm.Timeout,
        litellm.APIError,
        litellm.APIConnectionError,
        litellm.RateLimitError,
    ]
    retry_instance: retry_base = retry_if_exception_type(errors[0])
    for error in errors[1:]:
        retry_instance = retry_instance | retry_if_exception_type(error)
    return retry(
        # reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_instance,
        # before_sleep=_before_sleep,
    )


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""

        additional_kwargs = {}
        if _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(_dict["function_call"])

        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


async def acompletion_with_retry(
    llm: ChatLiteLLM,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        import litellm

        return await litellm.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        try:
            tool_call_chunks = [
                ToolCallChunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                for rtc in raw_tool_calls
            ]
        except KeyError:
            pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any] = {"content": message.content}
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
        message_dict["name"] = message.name
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatLiteLLM(BaseChatModel):
    """Chat model that uses the LiteLLM API."""

    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    streaming: bool = False
    temperature: Optional[float] = 0
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    template_file: Optional[str] = None
    instruction: Optional[str] = None

    max_retries: int = 6

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling LLM API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
            "temperature": self.temperature,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "api_version": self.api_version,
            **self.model_kwargs,
        }

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            import litellm

            return litellm.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key, python package exists, temperature, top_p, and top_k."""
        values["api_key"] = values.get("api_key", "")

        if values["temperature"] is not None and not 0 <= values["temperature"] <= 1:
            raise ValueError("temperature must be in the range [0.0, 1.0]")

        if values["top_p"] is not None and not 0 <= values["top_p"] <= 1:
            raise ValueError("top_p must be in the range [0.0, 1.0]")

        if values["top_k"] is not None and values["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        return values

    def _generate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(messages, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model": self.model}
        if (
            "logprobs" in response["choices"][0]
            and "content" in response["choices"][0]["logprobs"]
        ):
            llm_output["logprobs"] = (
                response["choices"][0].get("logprobs").get("content")
            )
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        async for chunk in await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            delta = chunk["choices"][0]["delta"]
            chunk = _convert_delta_to_message_chunk(delta, default_chunk_class)
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(message=chunk)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.content, chunk=cg_chunk)
            yield cg_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages=messages, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages)
        params = {**params, **kwargs}
        response = await acompletion_with_retry(
            self, messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        LiteLLM expects tools argument in OpenAI format.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters.
        These are used as the cache key for LLM calls."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.model_kwargs.get("reasoning_effort"),
            "max_tokens": self.max_tokens,
        }

    @property
    def _llm_type(self) -> str:
        return "litellm-chat"

    def _log_inputs_and_outputs(
        self, messages: List[BaseMessage], response: Any
    ) -> None:
        """Log the inputs and outputs of the model."""
        llm_input = messages[0][-1].content
        if messages[0][-1].type == "system":
            # it means the prompt did not have an `# input` block, and only has an instruction block
            llm_input = ""

        prompt_log = {
            "template_name": self.template_name,
            "instruction": self.instruction,
            "input": llm_input,
        }

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

        prompt_log["output"] = str(response)

        GlobalVars.prompt_logs.append(prompt_log)
