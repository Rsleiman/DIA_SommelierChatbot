### To run, type:
# streamlit run src/main_router.py --server.fileWatcherType none
from typing import cast
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st

from src.agents.router_agent import router_agent, IntentOutputSchema, CustomInputSchema, Intent
from src.agents.wine_pairing_agent import wine_pairing_agent, rag_context_provider
from src.agents.food_pairing_agent import food_pairing_agent
from src.agents.general_inquiry_agent import general_inquiry_agent
from rag.query_chroma import get_retriever

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
        message_history=st.session_state.messages[-3:]) # Return last 3 messages
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # RAG retrieval
    retriever = get_retriever(".chroma_basic")
    context = retriever.retrieve(prompt)
    if context:
        rag_context_provider.set_chunks(context)

    # Intent classification
    router_response = router_agent.run(input_schema)
    router_response = cast(IntentOutputSchema, router_response)
        
    if router_response.intent == Intent.WINE_PAIRING:
        agent_response = wine_pairing_agent.run(input_schema)
        msg = agent_response.response # type: ignore
    elif router_response.intent == Intent.FOOD_PAIRING:
        agent_response = food_pairing_agent.run(input_schema)
        msg = agent_response.response # type: ignore
    else:
        agent_response = general_inquiry_agent.run(input_schema)
        msg = agent_response.response # type: ignore

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)