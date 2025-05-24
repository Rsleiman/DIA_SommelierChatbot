import instructor
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, TypedDict
import sys
from pathlib import Path
# from atomic.tools

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.client import llm
from atomic_agents.agents.base_agent import BaseIOSchema, BaseAgent, BaseAgentConfig, BaseAgentInputSchema, AgentMemory, SystemPromptGenerator, SystemPromptContextProviderBase

router_memory = AgentMemory() #TODO: This memory should be the output agent's memory, not the router's memory. Find a way to pass the output agent's memory to the router agent.
                              # Also find a way to not append the router's output to any agent's memory, since it is not relevant to conversation history.


class Intent(str, Enum):
    """Enum for different intents of the user."""
    WINE_PAIRING = "wine_pairing"
    FOOD_PAIRING = "food_pairing"
    GENERAL_INQUIRY = "general_inquiry"


class CustomInputSchema(BaseIOSchema):
    """Custom input schema for the agent."""
    query: str = Field(description="The user query to be processed by the agent.")

class IntentOutputSchema(BaseIOSchema):
    """Custom output schema for the agent."""
    intent: Intent = Field(description="Correctly assign one of the predefined intents of the user.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of your intent classification, between 0 and 1.")


router_system_prompt_generator = SystemPromptGenerator(
    background= [ #TODO: Find a different format. "A [wine] sounds good!" incorrectly results in 'wine_pairing' intent.
        "You are an agent that classifies user queries into predefined intents.",
        "The predefined intents are: 'wine_pairing', 'food_pairing', and 'general_inquiry'.",
        "'wine_pairing' is assumed when the user has expressed their intention on ordering a certain food", #needs to be more clear
        "'food_pairing' is assumed when the user has expressed their intention on ordering a certain wine.",#needs to be more clear (examples?)
        "'general_inquiry' is typically for questions that don't explicitly ask for wine or food pairings, such as general questions about wine or food, or others.",
        "Your task is to analyze the user query and return the intent that best matches the query, and a confidence score for how confident you are in your decision.",
    ],
    steps= [
        "Analyze the user query to determine its intent.",
        "When unsure, refer to the message history to gather some context to inform your decision.", 
        "Determine how confident you are in your decision, and return a confidence score between 0 and 1.",
        "Consider both the message history and the ambiguity of the current query and how well they match the predefined intents when creating your score.",       
    ],  # TODO: Message history should refer to the output agent's memory, not the router's memory. Find a way to pass the output agent's memory to the router agent.
    output_instructions=[
        "### For the intent:"
        "Return the intent of the user query as EXACTLY one of the predefined intents: wine_pairing, food_pairing, general_inquiry.",
        "If the query does not match any of the predefined intents, return general_inquiry.",
        "Do NOT return any additional text or explanations, just the intent without any punctuation or quotes.",
        "### For the confidence score:",
        "Return a confidence score between 0 and 1, where 0 means you are not confident at all and 1 means you are completely confident.",
        "Do NOT return any additional text or explanations, just the score as a number.",
    ],
)

router_agent = BaseAgent(
    config=BaseAgentConfig(
        client=instructor.from_openai(llm),
        model="gpt-4o-mini",
        system_prompt_generator=router_system_prompt_generator,
        memory=router_memory, #TODO: Memory needs to be the output agent's memory OR remove entirely and inject memory into custom router input schema.
        input_schema=CustomInputSchema,
        output_schema=IntentOutputSchema
    ) # type: ignore
)



def route(user_input: str, intent_output: IntentOutputSchema):
    """
    Argument: Intent to route the query to appropriate agent.
    if intent == wine_pairing:
        send user query to wine pairing agent
    if intent == food_pairing:
        send user query to food pairing agent
    if intent == general_inquiry:
        send user query to general inquiry agent    
    """
    if intent_output.confidence < 0.5: # Low confidence -> general inquiry
        # general_agent.run(message=user_input)
        return 
    elif intent_output.intent == Intent.WINE_PAIRING:
        # wine_pairing_agent.run(message=user_input)
        return
    elif intent_output.intent == Intent.FOOD_PAIRING:
        # food_pairing_agent.run(message=user_input)
        return
    elif intent_output.intent == Intent.GENERAL_INQUIRY:
        # general_agent.run(message=user_input)
        return