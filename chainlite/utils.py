import asyncio
import logging
from typing import Any, Optional

from pydantic import validate_call
import rich
from langchain_core.runnables import chain
from tqdm import tqdm

logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Proxy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None):
    logger = logging.getLogger(name)
    # logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        # without this if statement, will have duplicate logs
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)-5s : %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


async def run_async_in_parallel(
    async_function, iterable, max_concurrency: int, desc: str = ""
):
    semaphore = asyncio.Semaphore(max_concurrency)  # Limit concurrent tasks

    async def async_function_with_semaphore(f, i, original_index) -> tuple:
        # Acquire the semaphore to limit the number of concurrent tasks
        async with semaphore:
            try:
                # Execute the asynchronous function and get the result
                result = await f(i)
                # Return the original index, result, and no error
                return original_index, result, None
            except Exception as e:
                # If an exception occurs, return the original index, no result, and the error message
                logger.exception(f"Task {original_index} failed with error: {e}")
                return original_index, None, str(e)

    tasks = []
    for original_index, item in enumerate(iterable):
        tasks.append(
            async_function_with_semaphore(async_function, item, original_index)
        )

    ret = [None] * len(tasks)
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        smoothing=0,
        desc=desc,
        mininterval=0.5,
        miniters=1,
    ):
        original_index, result, error = await future
        if error:
            # logger.error(f"Task {original_index} failed with error: {error}")
            ret[original_index] = None  # set it to some error indicator
        else:
            ret[original_index] = result

    return ret


@chain
def pprint_chain(_dict: Any) -> Any:
    """
    Print intermediate results for debugging
    """
    rich.print(_dict)
    return _dict


def validate_function():
    """A shortcut decorator"""
    return validate_call(
        validate_return=True, config=dict(arbitrary_types_allowed=True)
    )
