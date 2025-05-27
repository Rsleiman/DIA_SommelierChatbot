import instructor
from pydantic import Field
from enum import Enum
from typing import List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, SystemPromptGenerator

class Intent(str, Enum):
    """Enum for different intents of the user."""
    WINE_PAIRING = "wine_pairing"
    FOOD_PAIRING = "food_pairing"
    GENERAL_INQUIRY = "general_inquiry"


class CustomInputSchema(BaseIOSchema):
    """Custom input schema for the agent."""
    query: str = Field(description="The user query to be processed by the agent.")
    message_history: List[dict] = Field(description="The message history of the conversation, including previous user queries and agent responses.")

class IntentOutputSchema(BaseIOSchema):
    """Custom output schema for the agent."""
    intent: Intent = Field(description="Correctly assign one of the predefined intents of the user.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of your intent classification, between 0 and 1.")

# Version 1
# router_system_prompt_generator = SystemPromptGenerator(
#     background=[
#         "You classify user queries into one of three intents: 'wine_pairing', 'food_pairing', or 'general_inquiry'.",

#         "'wine_pairing': The user mentions a specific food and likely wants a wine to match it.",
#         "Examples: 'I'm having steak', 'What wine goes with lasagna?', 'This chicken looks good', 'I want sushi tonight'.",

#         "'food_pairing': The user mentions a specific wine and likely wants a food to match it.",
#         "Examples: 'I feel like drinking Merlot', 'I’ve got a Riesling', 'What food goes well with Pinot Noir?'.",

#         "'general_inquiry': General questions not about pairing, e.g., wine facts, food prep, or unrelated topics.",
#         "Examples: 'What are tannins?', 'Tell me about rosé', 'How is wine made?'.",

#         "Your job: Identify the correct intent and return it along with a confidence score (0–1).",
#     ],
#     steps=[
#         "Read the latest query (and message history if needed).",
#         "Match it to the most fitting intent.",
#         "Estimate confidence from 0 (no confidence) to 1 (certain).",
#     ],
#     output_instructions=[
#         "### Intent:",
#         "Return EXACTLY one: wine_pairing, food_pairing, general_inquiry.",
#         "If unsure, default to general_inquiry.",
#         "No extra text or punctuation.",

#         "### Confidence:",
#         "Return only a number between 0 and 1.",
#     ],
# )

# Version 2
router_system_prompt_generator = SystemPromptGenerator(
    background=[
        "You classify user queries into one of three intents: 'wine_pairing', 'food_pairing', or 'general_inquiry'.",

        "'wine_pairing': The user mentions a **specific dish or food** and likely wants a wine to pair with it.",
        "Examples: 'I'm having the steak', 'What wine goes with lasagna?', 'This chicken looks good'",

        "'food_pairing': The user mentions a **specific wine** and likely wants a food to pair with it.",
        "Examples: 'I feel like drinking Merlot', 'I’ve got a Riesling', 'What food goes well with Pinot Noir?'.",

        "'general_inquiry': General questions, or vague mentions of food/wine categories without specific items.",
        "Examples: 'What are tannins?', 'Tell me about rosé', 'I want a white wine', 'I want a fish dish'",

        "Important: If the user mentions only a wine or food **category** (like 'red wine', 'fish', or 'cheese'), classify it as 'general_inquiry'.",
        "Only classify as 'wine_pairing' or 'food_pairing' if a clearly identifiable specific food or wine is mentioned.",

        "Your job: Identify the correct intent and return it along with a confidence score (0–1).",
    ],
    steps=[
        "Read the latest query and message history.",
        "Match it to the most fitting intent based on specificity.",
        "Estimate confidence from 0 (no confidence) to 1 (certain).",
    ],
    output_instructions=[
        "### Intent:",
        "Return EXACTLY one: wine_pairing, food_pairing, general_inquiry.",
        "If not confident, default to general_inquiry.",

        "### Confidence:",
        "Return only a number between 0 and 1.",
    ],
)

router_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=router_system_prompt_generator,
        input_schema=CustomInputSchema,
        output_schema=IntentOutputSchema
    ) # type: ignore
)