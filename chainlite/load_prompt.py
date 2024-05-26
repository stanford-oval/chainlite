from functools import lru_cache
from datetime import datetime
import re
from zoneinfo import ZoneInfo  # Python 3.9 and later
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from jinja2 import Environment, FileSystemLoader

jinja2_comment_pattern = re.compile(r"{#.*?#}", re.DOTALL)

# Initial setup for prompt_block_identifiers remains the same
prompt_block_identifiers = {
    "input": ["# input\n", "# Input\n", "# INPUT\n"],
    "output": ["# output\n", "# Output\n", "# OUTPUT\n"],
    "instruction": ["# instruction\n", "# Instruction\n", "# Instruction\n"],
    "distillation_instruction": [
        "# distillation instruction\n",
        "# distillation_instruction\n",
        "# Distillation Instruction\n",
        "# DISTILLATION INSTRUCTION\n",
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


def add_template_constants(
    chat_prompt_template: ChatPromptTemplate,
) -> ChatPromptTemplate:
    # always make these useful constants available in a template
    # make a new function call each time since the date might change during a long-term server deployment
    pacific_zone = ZoneInfo("America/Los_Angeles")
    today = datetime.now(pacific_zone).date()

    current_year = today.year
    today = today.strftime("%B %d, %Y")  # e.g. May 30, 2024
    location = "the U.S."
    chatbot_name = "WikiChat"
    chat_prompt_template = chat_prompt_template.partial(
        today=today,
        current_year=current_year,
        location=location,
        chatbot_name=chatbot_name,
    )

    return chat_prompt_template


def find_all_substrings(string, substring) -> list[str]:
    return [match.start() for match in re.finditer(re.escape(substring), string)]


def _split_prompt_to_blocks(prompt: str) -> list[tuple[str, str]]:
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
        len([b for b in block_indices if b[1] == "instruction"]) == 1
    ), "Prompts should contain exactly one instruction block"
    num_distillation_instruction = len(
        [b for b in block_indices if b[1] == "distillation_instruction"]
    )
    assert (
        num_distillation_instruction <= 1
    ), "Prompts should contain at most one distillation instruction block"
    num_inputs = len([b for b in block_indices if b[1] == "input"])
    num_outputs = len([b for b in block_indices if b[1] == "output"])
    fewshot_start = 1 if num_distillation_instruction == 0 else 2
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
        block_string = prompt[
            block_indices_with_end[i][0]
            + len(block_indices_with_end[i][2]) : block_indices_with_end[i + 1][0]
        ].strip()

        blocks.append((block_indices_with_end[i][1], block_string))

    return blocks


def _prompt_blocks_to_chat_messages(
    blocks: list[tuple[str, str]], is_distilled: bool
) -> tuple[ChatPromptTemplate, str | None]:
    message_prompt_templates = []
    distillation_instruction = None
    if is_distilled:
        assert "distillation_instruction" in [
            b[0] for b in blocks
        ], "When using a distilled model, prompts used need a distillation instruction block"

    for block_type, block in blocks:
        if block_type == "distillation_instruction":
            distillation_instruction = block
            continue
        elif block_type == "instruction":
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
        # only keep the distillation_instruction and the last input
        assert distillation_instruction is not None
        message_prompt_templates = [
            SystemMessagePromptTemplate.from_template(block, template_format="jinja2"),
            message_prompt_templates[-1],
        ]
    chat_prompt_template = ChatPromptTemplate.from_messages(message_prompt_templates)
    chat_prompt_template = add_template_constants(chat_prompt_template)
    if distillation_instruction is not None:
        distillation_instruction = (
            (
                add_template_constants(
                    ChatPromptTemplate.from_template(
                        distillation_instruction, template_format="jinja2"
                    )
                )
            )
            .invoke({})
            .messages[0]
            .content
        )  # distillation prompts should not contain any variables other than template constants like {{today}}
        # print("distillation_instruction = ", distillation_instruction)

    return chat_prompt_template, distillation_instruction


def load_fewshot_prompt_template(
    template_file: str, is_distilled: bool, keep_indentation: bool
) -> tuple[ChatPromptTemplate, str | None]:
    fp = load_template_file(template_file, keep_indentation)
    blocks = _split_prompt_to_blocks(fp)
    # pprint(blocks)
    chat_prompt_template, distillation_instruction = _prompt_blocks_to_chat_messages(
        blocks, is_distilled
    )

    return chat_prompt_template, distillation_instruction
