"""
Functionality to work with .prompt files in Jinja2 format.
"""

import re
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple
from zoneinfo import ZoneInfo  # Python 3.9 and later

from jinja2 import Environment, FileSystemLoader
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

jinja2_comment_pattern = re.compile(r"{#.*?#}", re.DOTALL)

# Initial setup for prompt_block_identifiers remains the same
prompt_block_identifiers = {
    "input": [
        "# input\n",
        "# Input\n",
        "# INPUT\n",
        "# user\n",
        "# User\n",
        "# USER\n",
        "# human\n",
        "# Human\n",
        "# HUMAN\n",
    ],
    "output": [
        "# output\n",
        "# Output\n",
        "# OUTPUT\n",
        "# assistant\n",
        "# Assistant\n",
        "# ASSISTANT\n",
        "# ai\n",
        "# Ai\n",
        "# AI\n",
    ],
    "instruction": [
        "# instruction\n",
        "# Instruction\n",
        "# INSTRUCTION\n",
        "# System\n",
        "# SYSTEM\n",
        "# system\n",
    ],
}

jinja_environment = None  # Global variable to hold the Jinja2 environment


def initialize_jinja_environment(loader_paths):
    global jinja_environment

    loader = FileSystemLoader(loader_paths)
    jinja_environment = Environment(
        loader=loader,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@lru_cache()
def load_template_file(template_file: str, keep_indentation: bool) -> str:
    """
    This function is here just so that we can cache the templates and not have to read from disk every time.
    Also removes comment blocks and white space at the beginning and end of each line. These are usually added to make prompt templates more readable.
    We remove comment blocks first, so that we can get rid of extra lines before or after them.
    """
    raw_template = jinja_environment.loader.get_source(
        jinja_environment, template_file
    )[0]
    raw_template = re.sub(jinja2_comment_pattern, "", raw_template)
    if not keep_indentation:
        raw_template = "\n".join([line.strip() for line in raw_template.split("\n")])
    else:
        raw_template = "\n".join([line.rstrip() for line in raw_template.split("\n")])
    raw_template = re.sub(
        r"%}\s*", "%}", raw_template
    )  # remove the white space after {% for ... %} tags etc.

    return raw_template


added_template_constants = {}


def register_prompt_constants(constant_name_to_value_map: dict) -> None:
    """
    Make constant values available to all prompt templates.
    By default, current_year, today and location are set, and you can overwrite them or add new constants using this method.

    Args:
        constant_name_to_value_map (dict): A dictionary where keys are constant names and values are the corresponding constant values.

    Returns:
        None
    """
    for k, v in constant_name_to_value_map.items():
        added_template_constants[k] = v


def add_constants_to_template(
    chat_prompt_template: ChatPromptTemplate,
) -> ChatPromptTemplate:
    # always make these useful constants available in a template
    # make a new function call each time since the date might change during a long-term server deployment
    pacific_zone = ZoneInfo("America/Los_Angeles")
    today = datetime.now(pacific_zone).date()

    template_constants = {
        "current_year": today.year,
        "today": today.strftime("%B %d, %Y"),  # e.g. May 30, 2024
        "location": "the U.S.",
    }
    for k, v in added_template_constants.items():
        template_constants[k] = v

    chat_prompt_template = chat_prompt_template.partial(**template_constants)

    return chat_prompt_template


def find_all_substrings(string, substring) -> List[str]:
    return [match.start() for match in re.finditer(re.escape(substring), string)]


def _split_prompt_to_blocks(prompt: str) -> List[Tuple[str, str]]:
    block_indices = []
    for identifier in prompt_block_identifiers:
        for alternative in prompt_block_identifiers[identifier]:
            for i in find_all_substrings(prompt, alternative):
                block_indices.append((i, identifier, alternative))

    block_indices = sorted(
        block_indices
    )  # sort according to the index they were found at

    # check the prompt format is correct
    assert (
        len([b for b in block_indices if b[1] == "instruction"]) <= 1
    ), "Prompts should contain at most one instruction block"

    num_inputs = len([b for b in block_indices if b[1] == "input"])
    num_outputs = len([b for b in block_indices if b[1] == "output"])
    fewshot_start = 1
    assert (num_inputs == num_outputs + 1) or (
        num_inputs == 0 and num_outputs == 0
    ), "The order of few-shot blocks in the prompt should be ((input -> output) * N) -> input"
    for i, b in enumerate(block_indices[fewshot_start:]):
        if i % 2 == 0:
            assert (
                b[1] == "input"
            ), "The order of few-shot blocks in the prompt should be ((input -> output) * N) -> input"
        else:
            assert (
                b[1] == "output"
            ), "The order of few-shot blocks in the prompt should be ((input -> output) * N) -> input"

    block_indices_with_end = block_indices + [(len(prompt), "end", "end")]
    blocks = []
    for i in range(len(block_indices)):
        block_content = prompt[
            block_indices_with_end[i][0]
            + len(block_indices_with_end[i][2]) : block_indices_with_end[i + 1][0]
        ].strip()

        blocks.append((block_indices_with_end[i][1], block_content))

    return blocks


def _prompt_blocks_to_chat_messages(
    blocks: List[Tuple[str, str]], is_distilled: bool
) -> Tuple[ChatPromptTemplate, str | None]:
    message_prompt_templates = []

    # Add an instruction block if it is not present
    if len([b for b in blocks if b[0] == "instruction"]) == 0:
        blocks = [("instruction", "")] + blocks

    for block_type, block in blocks:
        if block_type == "instruction":
            block_type = SystemMessagePromptTemplate
        elif block_type == "input":
            block_type = HumanMessagePromptTemplate
        elif block_type == "output":
            block_type = AIMessagePromptTemplate
        else:
            assert False
        message_prompt_templates.append(
            block_type.from_template(block, template_format="jinja2")
        )
    if is_distilled:
        # only keep the system message and the last input
        message_prompt_templates = [
            message_prompt_templates[0],
            message_prompt_templates[-1],
        ]
    chat_prompt_template = ChatPromptTemplate.from_messages(message_prompt_templates)
    chat_prompt_template = add_constants_to_template(chat_prompt_template)

    return chat_prompt_template


def load_fewshot_prompt_template(
    template_file: str,
    template_blocks: list[tuple[str]],
    is_distilled: bool,
    keep_indentation: bool,
) -> Tuple[ChatPromptTemplate, str | None]:
    if not template_blocks:
        fp = load_template_file(template_file, keep_indentation)
        template_blocks = _split_prompt_to_blocks(fp)
    chat_prompt_template = _prompt_blocks_to_chat_messages(
        template_blocks, is_distilled
    )

    return chat_prompt_template
