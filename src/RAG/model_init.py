from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Load the embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents into memory
reader = SimpleDirectoryReader("data", exclude="data/extra menus")
docs = reader.load_data(show_progress=True)

# Initialise chromadb client
client = chromadb.PersistentClient("./chroma")

# Create a ChromaDB collection
collection = client.get_or_create_collection("food_wine_menu_pairings")

# Get and assign chroma vector store to storage context
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents=docs,
    storage_context=storage_context,
    embed_model=embed_model
)

# Save the index to disk
index.save_to_disk("index.json")
