from invoke import task
import os
from chainlite.utils import get_logger
import redis

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
def start_redis(c, redis_port=DEFAULT_REDIS_PORT):
    try:
        r = redis.Redis(host="localhost", port=redis_port)
        r.ping()
    except redis.exceptions.ConnectionError:
        logger.info("Redis server not found, starting it now...")
        c.run(f"redis-server --port {redis_port} --daemonize yes")
        return

    logger.debug("Redis server is aleady running.")


@task(pre=[load_api_keys, start_redis], aliases=["test"])
def tests(c, log_level="info", parallel=False):
    """Run tests using pytest"""

    test_files = [
        "./tests/test_llm_generate.py",
    ]

    pytest_command = (
        f"pytest "
        f"--log-cli-level={log_level} "
        "-rP "
        "--color=yes "
        "--disable-warnings "
    )

    if parallel:
        pytest_command += f"-n auto "

    pytest_command += " ".join(test_files)

    c.run(pytest_command, pty=True)
