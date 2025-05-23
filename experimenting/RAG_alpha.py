import nest_asyncio
from llm.client import get_llm
nest_asyncio.apply()
llm = get_llm()



from pprint import pprint
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb

# Load the embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents into memory
reader = SimpleDirectoryReader("data", exclude=["/data/extra menus"])
docs = reader.load_data(show_progress=True)

# Create a ChromaDB collection
client = chromadb.PersistentClient("chroma_db")
collection = client.get_or_create_collection("food_wine_menu_pairings")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents=docs,
    storage_context=storage_context,
    embed_model=embed_model
)




# Create query engine
query_engine = index.as_query_engine(
    llm=llm
    )

response = query_engine.query("I want the Muscadet, what main dish do you recommend I eat it with and why?")

print(f"RESPONSE: \n{response.__dict__}\n")