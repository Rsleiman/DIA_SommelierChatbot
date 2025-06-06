import instructor
from pydantic import Field
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, SystemPromptGenerator
from src.agents.router_agent import CustomInputSchema


class GeneralRAGComposerOutputSchema(BaseIOSchema):
    """Custom output schema for the agent."""
    queries: List[str] = Field(
        description="A list of 1 to 3 search queries generated from the user query and message history."
    )

general_rag_prompt_generator = SystemPromptGenerator(
    background=[
        "You are an assistant that generates search queries for retrieving items from a structured wine and food menu database.",
        "The database contains:",
        "- Names of wine and food items (e.g., 'Riesling', 'Beef Bourguignon')",
        "- Item characteristics (e.g., “white, medium-bodied, dry, floral”, or “poultry, rich, creamy, spicy”)",
        "- Pricing",

        "The user has made a general inquiry. It could be about a wine or food, or it could be more general.",
        "Your goal is to convert a user's input and message history into semantic queries that will match menu items that may help a sommelier understand the user's question.",
    ],
    steps=[
        "Identify the user's request from the user message.",
        "If there is no clear request, use the message history to infer the user's request.",
        "Determine whether a wine or food menu context is needed to reply just to the user's request.",
        "If so, generate 1 to 3 short semantic queries that can be used to retrieve relevant items from the menu database to answer just the user's request.",
        "If not, return an empty list.",
    ],
    output_instructions=[
        "Return a JSON object with a `queries` field containing a list of 1 to 3 string queries.",
        "Example: { \"queries\": [\"meaty dish\", \"beef\", \"earthy\"] }",
        "or { \"queries\": [\" red wine, oaky, full-bodied\"] }",
        "Avoid unnecessary words, punctuation, or explanations.",
    ],
)


general_rag_prompt_generator_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=general_rag_prompt_generator,
        input_schema=CustomInputSchema,
        output_schema=GeneralRAGComposerOutputSchema
    ) # type: ignore
)