

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from experimenting.query_chroma import get_retriever
from atomic_agents.agents.base_agent import BaseAgentInputSchema
from src.agents.sommelier_agent import sommelier_agent, rag_context_provider

print("import done")
console = Console()

while True:
    # Prompt the user for input
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break
        
    # Query the RAG with the user's input
    retriever = get_retriever() # -> BaseRetriever
    context = retriever.retrieve(user_input) # -> List[NodeWithScore]

    if context:
        rag_context_provider.set_chunks(context)

    # Process the user's input through the agent
    input_schema = BaseAgentInputSchema(chat_message=user_input)
    response = sommelier_agent.run(input_schema)

    # Display the agent's response
    console.print("[bold green]Agent:[/bold green] ", response)
