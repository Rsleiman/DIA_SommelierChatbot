### To run, type:
# streamlit run src/main_router_rag_composer.py --server.fileWatcherType none
from typing import cast
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st

from src.agents.router_agent import router_agent, IntentOutputSchema, CustomInputSchema, Intent
from src.agents.wine_pairing_agent import wine_pairing_agent, rag_context_provider
from src.agents.food_pairing_agent import food_pairing_agent
from src.agents.general_inquiry_agent import general_inquiry_agent
from src.agents.rag_composers.wine_pairing_rag_composer import wine_pairing_rag_composer_agent, WinePairingRAGComposerOutputSchema
from src.agents.rag_composers.food_pairing_rag_composer import food_pairing_rag_composer_agent, FoodPairingRAGComposerOutputSchema
from src.agents.rag_composers.general_inquiry_rag_composer import general_rag_prompt_generator_agent, GeneralRAGComposerOutputSchema
from rag.query_chroma import get_retriever
from agents.rag_composers.set_chunks import composer_set_chunks

st.title("💬 Sommelier Chatbot")
st.caption("🚀 Your AI wine expert")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome to our restaurant! Can I interest you in any wine to pair with your meal?"}
    ]

for msg in st.session_state.messages:   
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    input_schema = CustomInputSchema(
        query=prompt,
        message_history=st.session_state.messages[-3:]) # Return last 3 messages (previous assistant response, user reply, most recent assistant reply)
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Intent classification
    response = router_agent.run(input_schema)
    response = cast(IntentOutputSchema, response)

    retriever = get_retriever(".chroma_enriched")
        
    if response.intent == Intent.WINE_PAIRING:
        composer_set_chunks(retriever, input_schema, rag_context_provider, wine_pairing_rag_composer_agent)
        msg = wine_pairing_agent.run(input_schema)

    elif response.intent == Intent.FOOD_PAIRING:
        composer_set_chunks(retriever, input_schema, rag_context_provider, food_pairing_rag_composer_agent)
        msg = food_pairing_agent.run(input_schema)
    else:
        composer_set_chunks(retriever, input_schema, rag_context_provider, general_rag_prompt_generator_agent)
        msg = general_inquiry_agent.run(input_schema)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)