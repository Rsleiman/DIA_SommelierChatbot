from atomic_agents.agents.base_agent import SystemPromptContextProviderBase

class RAGContextProvider(SystemPromptContextProviderBase):
    """Context provider for RAG (Retrieval-Augmented Generation)."""
    def __init__(self, title: str):
        super().__init__(title)
        self.chunks = []

    def set_chunks(self, chunks):
        self.chunks = chunks

    def get_info(self) -> str:
        if not self.chunks:
            return "No relevant information found."
        context_info = ""
        for node in self.chunks:
            context_info += f"- {node.get_content()}\n"
            # print(f"Context chunk: {node.get_content()}") # For debugging purposes
        # print(f"RAGContextProvider: {len(self.chunks)} chunks retrieved") # For debugging purposes
        return context_info

rag_context_provider = RAGContextProvider(title="Food & Wine Menu RAG Retrieval")