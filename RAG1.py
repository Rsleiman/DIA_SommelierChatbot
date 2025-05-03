import nest_asyncio
nest_asyncio.apply()


from pprint import pprint
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb


# Load the embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents into memory
reader = SimpleDirectoryReader("data", exclude="data/extra menus")
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


from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
import os
load_dotenv()
llm = OpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Create query engine

query_engine = index.as_query_engine(
    llm=llm
    )

response = query_engine.query("I want the Muscadet, what main dish do you recommend I eat it with and why?")

print(f"RESPONSE: \n{response.__dict__}\n")