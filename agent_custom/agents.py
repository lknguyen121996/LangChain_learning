from langchain.agents import tool


@tool
def get_text_length(data: dict) -> int:
    """
    Returns the length of the text provided in the input dictionary.

    This utility function calculates and returns the number of characters
    in the 'text' field of the provided dictionary. It can be used to
    determine the size of a text input for various applications, such as
    text analysis or validation.

    Args:
        data (dict): A dictionary containing the key 'text' whose value
                     is the input string to be measured.

    Returns:
        int: The length of the input string in the 'text' field.
    """
    text = data.get("text", "")
    print("hey u are using my function to get text length")
    return len(text)

@tool
def replace_text(data: dict) -> str:
    """
    Replaces occurrences of a substring in the text with a new substring.

    This utility function searches for all occurrences of the specified
    substring (old) in the input text and replaces them with another
    substring (new).

    Args:
        data (dict): A dictionary containing the keys:
                     - 'text' (str): The input string where replacements will be made.
                     - 'old' (str): The substring to be replaced.
                     - 'new' (str): The substring to replace with.

    Returns:
        str: The modified string with replacements made.
    """
    text = data.get("text", "")
    old = data.get("old", "")
    new = data.get("new", "")
    print("hey u are using my function to remove text")
    return text.replace(old, new)