from pathlib import Path
from pprint import pprint
from typing import List
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()
# Recreate the embedding model (must match what you used to create the DB)
hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_retriever() -> BaseRetriever:
    # Path to ChromaDB
    script_dir = Path(__file__).parent.resolve()
    chroma_path = script_dir.parent / "src" / ".chroma"

    # Initialize the ChromaDB client
    db = chromadb.PersistentClient(path=str(chroma_path))

    # Get the collection
    chroma_collection = db.get_or_create_collection("quickstart") #TODO: Change name to "menus"

    # Now you can use chroma_collection or wrap it in ChromaVectorStore for querying
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Recreate the index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=hf_embeddings
    )

    # Use the retriever to get relevant chunks for your query
    retriever = index.as_retriever()
    return retriever

if __name__ == "__main__":
    retriever = get_retriever()
    query = "What wine would you recommend with the slow-cooked duck?"
    results = retriever.retrieve(query)

    print("\n\n\n")
    print("Raw Results:")
    pprint(results)
    print("\n\n\n")

    print("Relevant Chunks (Formatted):")
    for i, node in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(node.get_content())

