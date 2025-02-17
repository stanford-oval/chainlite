import os

import redis
from invoke import task

from chainlite.utils import get_logger

logger = get_logger(__name__)

DEFAULT_REDIS_PORT = 6379


@task
def load_api_keys(c):
    try:
        with open("API_KEYS") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = tuple(line.split("=", 1))
                    key, value = key.strip(), value.strip()
                    os.environ[key] = value
                    logger.debug("Loaded API key named %s", key)

    except Exception as e:
        logger.error(
            "Error while loading API keys from API_KEY. Make sure this file exists, and has the correct format. %s",
            str(e),
        )


@task()
def start_redis(c, redis_port: int = DEFAULT_REDIS_PORT):
    """
    Start a Redis server if it is not already running.

    This task attempts to connect to a Redis server on the specified port.
    If the connection fails (indicating that the Redis server is not running),
    it starts a new Redis server on that port.

    Parameters:
    - c: Context, automatically passed by invoke.
    - redis_port (int): The port number on which to start the Redis server. Defaults to DEFAULT_REDIS_PORT.
    """
    try:
        r = redis.Redis(host="localhost", port=redis_port)
        r.ping()
    except redis.exceptions.ConnectionError:
        logger.info("Redis server not found, starting it now...")
        c.run(
            f"docker run --rm -d --name redis-stack -p {redis_port}:6379 -p 8001:8001 redis/redis-stack:latest"
        )
        return

    logger.debug("Redis server is already running.")


@task(pre=[load_api_keys, start_redis], aliases=["test"])
def tests(c, log_level="info", parallel=False, test_file: str = None):
    """Run tests using pytest"""

    if test_file:
        test_files = [f"./tests/{test_file}"]
    else:
        test_files = [
            "./tests/test_llm_generate.py",
            "./tests/test_llm_structured_output.py",
            "./tests/test_function_calling.py",
            "./tests/test_logprobs.py",
        ]

    pytest_command = (
        f"pytest "
        f"--log-cli-level={log_level} "
        "-rP "
        "--color=yes "
        # "--disable-warnings "
        "-x "  # Stop after first failure
    )

    if parallel:
        pytest_command += f"-n auto "

    pytest_command += " ".join(test_files)

    c.run(pytest_command, pty=True)


@task
def format_code(c):
    """Format code using black and isort"""
    c.run("isort --profile black .", pty=True)
    c.run("black .", pty=True)
