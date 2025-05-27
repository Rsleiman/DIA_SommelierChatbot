import instructor
from pydantic import Field
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, SystemPromptGenerator

class RAGComposerInputSchema(BaseIOSchema):
    """Custom input schema for the agent."""
    query: str = Field(description="The user query to be processed by the agent.")
    message_history: List[dict] = Field(description="The message history of the conversation, including previous user queries and agent responses.")

class RAGComposerOutputSchema(BaseIOSchema):
    """Custom output schema for the agent."""
    food_item: str = Field(
        description="The identified food item that the user has intended to order."
    )
    queries: List[str] = Field(
        description="A list of 1 to 2 search queries generated from the user query and message history."
    )

wine_rag_prompt_generator = SystemPromptGenerator(
    background=[
        "You generate search queries to retrieve wine items from a structured menu database. The database includes:",
        "Names of wines and foods (e.g., “Riesling”, “Beef Bourguignon”)",
        "Their characteristics (e.g., “white, medium-bodied, dry, floral”, or “poultry, rich, creamy, spicy”)",
        "Pricing",
        "The user has chosen a food and wants a well-paired wine.",
        "Your job is to identify the user's chosen food item and generate 1-2 concise queries describing ideal wine characteristics that would pair well with the food.",
        "Queries must only use wine-related keywords (e.g., body, tannin, acidity, sweetness, colour, flavour)."
    ],
    steps=[
        "From the user message and history, identify the food item.",
        "Infer the food's characteristics. Think of the possibe characteristics of well-paired wines.",
        "Describe 1–2 ideal wine profiles using only wine-related terms."
    ],
    output_instructions=[
        "Return a JSON object with the following format and example fields:",
        "{ \"food_item\": \"Seabream Fillet\", \"queries\": [\"white, light-bodied, citrusy, low tannin\", \"white, dry, crisp, mineral, earthy\"]}",
        "Each query = one wine profile using wine characteristics only."
    ]
)


wine_pairing_rag_composer_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=wine_rag_prompt_generator,
        input_schema=RAGComposerInputSchema,
        output_schema=RAGComposerOutputSchema
    ) # type: ignore
)