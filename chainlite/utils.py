import logging
from typing import Optional


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
