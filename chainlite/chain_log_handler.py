from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from chainlite.llm_config import GlobalVars
from chainlite.llm_output import ToolOutput


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

        # find the first system message
        system_message = [m for m in messages[0] if m.type == "system"]
        if len(system_message) == 0:
            # no system message
            distillation_instruction = ""
        else:
            distillation_instruction = system_message[0].content

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
            elif (
                isinstance(response, tuple)
                and len(response) == 2
                and isinstance(response[1], list)
            ):
                # the second element of the tuple is a list of ChatCompletionTokenLogprob (or its converted dict)
                response = response[0]

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
