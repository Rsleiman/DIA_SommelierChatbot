from llama_index.core.retrievers import BaseRetriever
from atomic_agents.agents.base_agent import BaseAgent
from src.agents.router_agent import CustomInputSchema
from src.agents.context_provider import RAGContextProvider
from typing import List
from llama_index.core.schema import NodeWithScore


def deduplicate_nodes(nodes_list: List[NodeWithScore]):
    seen_content = set()
    unique_nodes = []
    # print(f"Deduplicating {len(nodes_list)} nodes...")  # For debugging purposes
    for nws in nodes_list:
        content = nws.node.get_content()
        if content not in seen_content:
            unique_nodes.append(nws)
            seen_content.add(content)
    return unique_nodes

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
        rag_context_provider.set_chunks([])  # Clear context if no queries
        return
    context = retriever.retrieve("\n".join(queries))
    # print(f"composer_set_chunks len: {len(context)}") # For debugging purposes
    if context:
        context = deduplicate_nodes(context) # Remove duplicate nodes
        rag_context_provider.set_chunks(context)
