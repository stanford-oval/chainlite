import re
from chainlite import chain


@chain
def extract_tag_from_llm_output(
    llm_output: str, tags: str | list[str]
) -> str | list[str]:
    """
    Extracts content enclosed within <tag></tag> tags from a given LLM output string.

    Args:
        llm_output (str): The output string from which to extract content.
        tags (str | list[str]): A single tag or a list of tags to search for in the output string.

    Returns:
        str | list[str]: The extracted content for the specified tag(s). If a single tag is provided,
                         a single string is returned. If a list of tags is provided, a list of strings
                         is returned, each corresponding to the content of the respective tag.
    """
    is_list = isinstance(tags, list)
    if not is_list:
        assert isinstance(tags, str)
        tags = [tags]
    all_extracted_tags = []
    for tag in tags:
        extracted_tag = ""
        tag_start = llm_output.find(f"<{tag}>") + len(f"<{tag}>")
        tag_end = llm_output.find(f"</{tag}>", tag_start)
        if tag_start >= 0 and tag_end >= 0:
            extracted_tag = llm_output[tag_start:tag_end].strip()
        if tag_start >= 0 and tag_end < 0:
            extracted_tag = llm_output[tag_start:].strip()
        if extracted_tag.startswith("-"):
            extracted_tag = extracted_tag[1:].strip()
        all_extracted_tags.append(extracted_tag.strip())

    if not is_list:
        return all_extracted_tags[0]
    return all_extracted_tags


@chain
def lines_to_list(llm_output: str) -> list[str]:
    """
    Convert a string of lines into a list of strings, processing each line.

    This function processes each line of the input string `llm_output` by:
    - Splitting the input string by newline characters.
    - Ignoring empty lines.
    - Removing leading hyphens and trimming whitespace.
    - Removing starting item numbers (e.g., "1.", "2.", etc.).

    Args:
        llm_output (str): The input string containing lines to be processed.

    Returns:
        list[str]: A list of processed strings.
    """
    ret = []
    for r in llm_output.split("\n"):
        if not r.strip():
            continue
        if r.startswith("-"):
            r = r[1:].strip()
        # remove starting item number
        r = re.split(r"^\d+\.", r)[-1]
        ret.append(r)

    return ret


@chain
def string_to_indices(llm_output: str, llm_output_start_index: int) -> list[int]:
    """
    Convert a comma-separated string of n-indexed integers into a 0-indexed list of integers.

    This function takes a string containing integers separated by commas,
    removes any surrounding square brackets, and converts each integer
    into 0-indexed by subtracting the specified start index.

    Args:
        llm_output (str): The string containing comma-separated integers.
        llm_output_start_index (int): Whether the llm_output_start_index is 1-indexed or 0-indexed.

    Returns:
        list[int]: A list of indices derived from the input string.
    """
    # Remove square brackets, if any
    cleaned_output = llm_output.strip("[]")

    # Split the string by commas and convert each element to an integer
    result = []
    for item in cleaned_output.split(","):
        item = item.strip()
        if item.isdigit():
            result.append(int(item) - llm_output_start_index)
    return result
