### To run, type:
# streamlit run src/main_rag.py --server.fileWatcherType none

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
from RAG.query_chroma import get_retriever
from src.agents.sommelier_agent import sommelier_agent, rag_context_provider, CustomOutputSchema, CustomInputSchema

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

    # RAG retrieval
    retriever = get_retriever(".chroma_basic")
    context = retriever.retrieve(prompt)
    if context:
        rag_context_provider.set_chunks(context)

    # Agent response
    input_schema = CustomInputSchema(query=prompt)
    response = sommelier_agent.run(input_schema)
    # If response is a pydantic object, get the string
    if hasattr(response, "response"):
        print("Response has attribute response. is a pydantic object, getting the string")
        msg = response.response # type: ignore
    else:
        print("Response is not a pydantic object, using str()")
        msg = str(response)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)