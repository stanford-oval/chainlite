import asyncio
import atexit
from contextlib import redirect_stdout
import io
from typing import Any, Optional

from langchain_community.cache import AsyncRedisCache
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.load.dump import dumps
import redis.asyncio as redis


SECONDS_IN_A_WEEK = 60 * 60 * 24 * 7


class CustomAsyncRedisCache(AsyncRedisCache):
    """This class fixes langchain>=0.2.*'s cache issue with LiteLLM
    The core of the problem is that LiteLLM's `Usage`, `ChatCompletionMessageToolCall` and `ChatCompletionTokenLogprob` 
        classes should inherit from LangChain's Serializable class, but don't.
    This class is the minimal fix to make it work.
    """

    @staticmethod
    def _configure_pipeline_for_update(
        key: str,
        pipe: Any,
        return_val: RETURN_VAL_TYPE,
        ttl: Optional[int] = SECONDS_IN_A_WEEK,
    ) -> None:
        for r in return_val:
            if (
                hasattr(r.message, "additional_kwargs")
                and "tool_calls" in r.message.additional_kwargs
            ):
                r.message.additional_kwargs["tool_calls"] = [
                    tool_call.dict()
                    for tool_call in r.message.additional_kwargs["tool_calls"]
                ]

            if (
                hasattr(r.message, "response_metadata")
                and "token_usage" in r.message.response_metadata
            ):
                r.message.response_metadata["token_usage"] = (
                    r.message.response_metadata["token_usage"].dict()
                )
            if (
                hasattr(r.message, "response_metadata")
                and "logprobs" in r.message.response_metadata
            ):
                r.message.response_metadata["logprobs"] = [
                    logprob.dict()
                    for logprob in r.message.response_metadata["logprobs"]
                ]
        pipe.hset(
            key,
            mapping={
                str(idx): dumps(generation) for idx, generation in enumerate(return_val)
            },
        )
        if ttl is not None:
            pipe.expire(key, ttl)


_global_redis_client = None


def init_redis_client() -> None:
    """Initialize the global Redis client."""
    # TODO move cache setting to the config file
    # We do not use LiteLLM's cache since it has a bug. We use LangChain's instead

    global _global_redis_client
    _global_redis_client = redis.Redis.from_url("redis://localhost:6379")
    redis_cache = CustomAsyncRedisCache(
        _global_redis_client,
    )
    from langchain.globals import set_llm_cache

    set_llm_cache(redis_cache)

    # Register the cleanup function so that it runs when Python exits
    atexit.register(_sync_close_redis_client)


async def _close_redis_client() -> None:
    """Asynchronously close the global Redis client."""
    global _global_redis_client
    if _global_redis_client:
        await _global_redis_client.close()


def _sync_close_redis_client() -> None:
    try:
        # Create a new event loop to avoid pitfalls with an already closed global loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Redirect stdout (or stderr) while closing the client to prevent the message from appearing.
        with redirect_stdout(io.StringIO()):
            loop.run_until_complete(_close_redis_client())
        loop.close()
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            pass
        else:
            print(f"Error during Redis client cleanup: {e}")
