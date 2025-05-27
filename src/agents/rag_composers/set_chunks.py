from llama_index.core.retrievers import BaseRetriever
from atomic_agents.agents.base_agent import BaseAgent
from src.agents.router_agent import CustomInputSchema
from src.agents.context_provider import RAGContextProvider

def composer_set_chunks(
        retriever: BaseRetriever,
        input_schema: CustomInputSchema,
        rag_context_provider: RAGContextProvider,
        agent: BaseAgent):
    # Get wine pairing queries from RAG composer
    rag_compose_response = agent.run(input_schema)
    queries = rag_compose_response.queries # type: ignore

    # Query chromadb for wine pairings
    if not queries:
        return
    context = retriever.retrieve("\n".join(queries))  #TODO: Handle multiple queries
    if context:
        rag_context_provider.set_chunks(context)
