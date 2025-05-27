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
    wine_item: str = Field(
        description="The identified wine item that the user has intended to order."
    )
    queries: List[str] = Field(
        description="A list of 1 to 2 search queries generated from the user query and message history."
    )

food_rag_prompt_generator = SystemPromptGenerator(
    background=[
        "You generate search queries to retrieve food items from a structured menu database. The database includes:",
        "Names of wines and foods (e.g., “Riesling”, “Beef Bourguignon”)",
        "Their characteristics (e.g., “white, medium-bodied, dry, floral”, or “poultry, rich, creamy, spicy”)",
        "Pricing",
        "The user has chosen a wine and wants a well-paired food.",
        "Your job is to identify the user's chosen wine item and generate 1-2 concise queries describing ideal food characteristics and key ingredients that would pair well with the wine.",
        "Queries must only use food-related keywords (e.g. creamy, spicy, meaty, fish, fresh, hearty, tomato, etc)."
    ],
    steps=[
        "From the user message and history, identify the wine item.",
        "Infer the wine's characteristics. Think of the possible characteristics and key ingredients of well-paired foods.",
        "Describe 1–2 ideal food profiles using only food-related terms."
    ],
    output_instructions=[
        "Return a JSON object with the following format and example fields:",
        "{ \"wine_item\": \"Pinot Grigio\", \"queries\": [\"creamy, rich, seafood, bright\", \"fresh, light, citrusy, butter, pepper\"]}",
        "Each query = one food profile using food characteristics only."
    ]
)


food_pairing_rag_composer_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=food_rag_prompt_generator,
        input_schema=RAGComposerInputSchema,
        output_schema=RAGComposerOutputSchema
    ) # type: ignore
)