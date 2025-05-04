from langchain.agents import tool


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text"""
    print("hey u are using me")
    return len(text)
