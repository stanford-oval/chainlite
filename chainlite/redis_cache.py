from typing import Any, Optional

import redis.asyncio as redis
from langchain.globals import set_llm_cache
from langchain_community.cache import AsyncRedisCache
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.load.dump import dumps

# TODO move cache setting to the config file
# We do not use LiteLLM's cache since it has a bug. We use LangChain's instead
redis_client = redis.Redis.from_url("redis://localhost:6379")


class CustomAsyncRedisCache(AsyncRedisCache):
    """This class fixes langchain 0.2.*'s cache issue with LiteLLM
    The core of the problem is that LiteLLM's `Usage` and `ChatCompletionMessageToolCall` classes should inherit from LangChain's Serializable class, but don't.
    This class is the minimal fix to make it work.
    """

    @staticmethod
    def _configure_pipeline_for_update(
        key: str, pipe: Any, return_val: RETURN_VAL_TYPE, ttl: Optional[int] = None
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
        pipe.hset(
            key,
            mapping={
                str(idx): dumps(generation) for idx, generation in enumerate(return_val)
            },
        )
        if ttl is not None:
            pipe.expire(key, ttl)


SECONDS_IN_A_WEEK = 60 * 60 * 24 * 7
redis_cache = CustomAsyncRedisCache(
    redis_client, ttl=SECONDS_IN_A_WEEK
)  # TTL is in seconds
set_llm_cache(redis_cache)
