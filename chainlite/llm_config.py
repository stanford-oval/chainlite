import os
import threading
from typing import Any, Optional

import litellm
import redis.asyncio as redis
import yaml
from langchain.globals import set_llm_cache
from langchain_community.cache import AsyncRedisCache
from langchain_core.caches import RETURN_VAL_TYPE
from langchain_core.load.dump import dumps

from .load_prompt import initialize_jinja_environment

# TODO move cache setting to the config file
# We do not use LiteLLM's cache, use LangChain's instead
redis_client = redis.Redis.from_url("redis://localhost:6379")


class CustomAsyncRedisCache(AsyncRedisCache):
    """This class fixes langchain 0.2.*'s cache issue with LiteLLM
    The core of the problem is that LiteLLM's Usage class should inherit from LangChain's Serializable class, but doesn't.
    This class is the minimal fix to make it work.
    """

    @staticmethod
    def _configure_pipeline_for_update(
        key: str, pipe: Any, return_val: RETURN_VAL_TYPE, ttl: Optional[int] = None
    ) -> None:
        for r in return_val:
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

litellm.drop_params = (
    True  # Drops unsupported parameters for non-OpenAI APIs like TGI and Together.ai
)


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __delitem__(self, key):
        with self._lock:
            del self._dict[key]

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())


class GlobalVars:
    prompt_logs = ThreadSafeDict()
    all_llm_endpoints = None
    prompt_dirs = None
    prompt_log_file = None
    prompts_to_skip_for_debugging = None
    local_engine_set = None

    @classmethod
    def get_all_configured_engines(cls):
        all_engines = set()
        for endpoint in cls.all_llm_endpoints:
            all_engines.update(endpoint["engine_map"].keys())
        return all_engines


def get_all_configured_engines():
    return GlobalVars.get_all_configured_engines()


def load_config_from_file(config_file: str) -> None:
    with open(config_file, "r") as config_file:
        config = yaml.unsafe_load(config_file)

    # TODO raise errors if these values are not set, use pydantic v2
    GlobalVars.prompt_dirs = config.get("prompt_dirs", ["./"])
    GlobalVars.prompt_log_file = config.get("prompt_logging", {}).get(
        "log_file", "./prompt_logs.jsonl"
    )
    GlobalVars.prompts_to_skip_for_debugging = set(
        config.get("prompt_logging", {}).get("prompts_to_skip", [])
    )

    litellm.set_verbose = config.get("litellm_set_verbose", False)

    GlobalVars.all_llm_endpoints = config.get("llm_endpoints", [])
    for a in GlobalVars.all_llm_endpoints:
        if "api_key" in a:
            a["api_key"] = os.getenv(a["api_key"])

    GlobalVars.all_llm_endpoints = [
        a
        for a in GlobalVars.all_llm_endpoints
        if "api_key" not in a or (a["api_key"] is not None and len(a["api_key"]) > 0)
    ]  # remove resources for which we don't have a key

    # tell LiteLLM how we want to map the messages to a prompt string for these non-chat models
    for endpoint in GlobalVars.all_llm_endpoints:
        if "prompt_format" in endpoint:
            if endpoint["prompt_format"] == "distilled":
                # {instruction}\n\n{input}\n
                for engine, model in endpoint["engine_map"].items():
                    litellm.register_prompt_template(
                        model=model,
                        roles={
                            "system": {
                                "pre_message": "",
                                "post_message": "\n\n",
                            },
                            "user": {
                                "pre_message": "",
                                "post_message": "\n",
                            },
                            "assistant": {
                                "pre_message": "",
                                "post_message": "\n",
                            },  # this will be ignored since "distilled" formats only support one output turn
                        },
                        initial_prompt_value="",
                        final_prompt_value="",
                    )
            else:
                raise ValueError(
                    f"Unsupported prompt format: {endpoint['prompt_format']}"
                )
    GlobalVars.local_engine_set = set()

    for endpoint in GlobalVars.all_llm_endpoints:
        for engine, model in endpoint["engine_map"].items():
            if model.startswith("huggingface/"):
                GlobalVars.local_engine_set.add(engine)

    initialize_jinja_environment(GlobalVars.prompt_dirs)


# this code is NOT safe to use with multiprocessing, only multithreading
thread_lock = threading.Lock()

load_config_from_file("./llm_config.yaml")
total_cost = 0.0  # in USD


def add_to_total_cost(amount: float):
    global total_cost
    with thread_lock:
        total_cost += amount


def get_total_cost() -> float:
    """
    This function is used to get the total LLM cost accumulated so far

    Returns:
        float: The total cost accumulated so far in USD.
    """
    global total_cost
    return total_cost


async def track_cost_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    try:
        if kwargs["cache_hit"]:
            # no cost because of caching
            # TODO this doesn't work with streaming
            return

        response_cost = 0
        # check if we have collected an entire stream response
        if "complete_streaming_response" in kwargs:
            # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost
            completion_response = kwargs["complete_streaming_response"]
            input_text = kwargs["messages"]
            output_text = completion_response["choices"][0]["message"]["content"]
            response_cost = litellm.completion_cost(
                model=kwargs["model"], messages=input_text, completion=output_text
            )
        elif kwargs["stream"] != True:
            # for non streaming responses
            response_cost = litellm.completion_cost(
                completion_response=completion_response
            )
        if response_cost > 0:
            add_to_total_cost(response_cost)
    except:
        pass
        # This can happen for example because of local models


# Assign the cost callback function
litellm.success_callback = [track_cost_callback]
