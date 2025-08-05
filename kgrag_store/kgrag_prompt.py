from typing import Optional


PARSER_PROMPT: str = """
You are a precise graph relationship extractor.
Extract all relationships from the text and format
them as a JSON object with this exact structure:
{
    "graph": [
        {
            "node": "Person/Entity",
            "target_node": "Related Entity",
            "relationship": "Type of Relationship"
        },
        ...more relationships...
    ]
}
Include ALL relationships mentioned in the text, including
implicit ones. Be thorough and precise.
"""

AGENT_PROMPT: str = (
    "You are an intelligent assistant with access to the "
    "following knowledge graph:\n\n"
    "Nodes: \"{nodes_str}\"\n\n"
    "Edges: \"{edges_str}\"\n\n"
    "Using this graph, Answer the following question:\n\n"
    "User Query: \"{user_query}\""
)


def create_prompt_parser(prompt: Optional[str] = None) -> str:
    """
    Create a prompt for the LLM parser.
    This prompt is used to instruct the LLM on how to parse
    and extract relationships from the input text.
    Args:
        prompt_user (Optional[str]): Custom user prompt
        to include in the parser.
    If None, the default parser prompt is used.
    Returns:
        str: Formatted prompt string.
    """

    if prompt is None:
        prompt = PARSER_PROMPT
    elif not isinstance(prompt, str):
        raise ValueError("prompt_user must be a string or None.")
    elif not prompt.strip():
        raise ValueError("prompt_user cannot be an empty string.")
    else:
        prompt = (
            f"{PARSER_PROMPT}\n"
            f"{prompt}\n"
        )

    return prompt
