### To run, type: streamlit run src/main_route.py --server.fileWatcherType none

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
# from experimenting.query_chroma import get_retriever
from src.agents.router_agent import router_agent, IntentOutputSchema, CustomInputSchema

st.title("💬 Sommelier Chatbot")
st.caption("🚀 Your AI wine expert")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome to our restaurant! Can I interest you in any wine to pair with your meal?"}
    ]

for msg in st.session_state.messages:   
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # Add user message to session
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Intent classification
    input_schema = CustomInputSchema(query=prompt) #TODO: Suggestion: Create customrouterinputschema that has a message history field instead of configuring a new AgentMemory() object.

    response = router_agent.run(input_schema) # No need to update memory
    # If response is a pydantic object, get the string
    if hasattr(response, "response"):
        msg = response.response
    else:
        msg = str(response)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)