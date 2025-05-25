#TODO

import instructor
from pydantic import Field
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, SystemPromptGenerator
from src.agents.context_provider import rag_context_provider

class CustomInputSchema(BaseIOSchema):
    """Custom input schema for the agent."""
    query: str = Field(description="The user query to be processed by the agent.")
    message_history: List[dict] = Field(description="The message history of the conversation, including previous user queries and agent responses.")


class CustomOutputSchema(BaseIOSchema):
    """Custom output schema for the agent."""
    queries: List[str] = Field(
        description="A list of 1 to 3 search queries generated from the user query and message history."
    )

## GENERAL VERSION 1
general_rag_prompt_generator = SystemPromptGenerator(
    background=[
        "You are an assistant that generates search queries for retrieving items from a structured wine and food menu database.",
        "The database contains:",
        "- Names of wine and food items (e.g., 'Riesling', 'Beef Bourguignon')",
        "- Possible descriptions of those items (e.g., 'aromatic white wine, floral notes', 'pan-fried in rich creamy sauce')",
        "- Pricing for each item",

        "Your goal is to convert a user's request and conversation history into one or more concise semantic queries that match menu items or their descriptions.",
        "You will not retrieve recipes, pairing rules, or general knowledge — only concrete wine and food choices.",
        "Avoid abstract or overly broad queries that wouldn’t match actual menu entries.",

        "Use the user's intent (e.g., asking for a wine for a specific food) and rephrase or extract keywords to match the style and substance of the menu database.",
    ],
    steps=[
        "Review the latest user message.",
        "Use message history for context if the user message is vague.",
        "Extract 1 to 3 short search queries that best match item names or descriptions from the menu database.",
    ],
    output_instructions=[
        "Return a JSON object with a `queries` field containing a list of 1 to 3 string queries.",
        "Example: { \"queries\": [\"duck confit\", \"Pinot Noir\", \"red wine with earthy notes\"] }",
        "Avoid unnecessary words, punctuation, or explanations.",
    ],
)

wine_rag_prompt_generator = SystemPromptGenerator(
    background=[
        "You are an assistant that generates search queries for retrieving items from a structured wine and food menu database.",
        "The database contains:",
        "- Names of wine and food items (e.g., 'Riesling', 'Beef Bourguignon')",
        "- Possible descriptions of those items (e.g., 'aromatic white wine, floral notes', 'pan-fried in rich creamy sauce')",
        "- Pricing for each item",

        "The user has chosen a certain food,  interest in a certain food. "
        "Your goal is to convert a user's request and conversation history into one or more concise semantic queries that match menu items or their descriptions.",
        "You will not retrieve recipes, pairing rules, or general knowledge — only concrete wine and food choices.",
        "Avoid abstract or overly broad queries that wouldn’t match actual menu entries.",

        "Use the user's intent (e.g., asking for a wine for a specific food) and rephrase or extract keywords to match the style and substance of the menu database.",
    ],
    steps=[
        "Review the latest user message.",
        "Use message history for context if the user message is vague.",
        "Extract 1 to 3 short search queries that best match item names or descriptions from the menu database.",
    ],
    output_instructions=[
        "Return a JSON object with a `queries` field containing a list of 1 to 3 string queries.",
        "Example: { \"queries\": [\"duck confit\", \"Pinot Noir\", \"red wine with earthy notes\"] }",
        "Avoid unnecessary words, punctuation, or explanations.",
    ],
)

wine_pairing_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=wine_rag_prompt_generator,
        input_schema=CustomInputSchema,
        output_schema=CustomOutputSchema
    ) # type: ignore
)