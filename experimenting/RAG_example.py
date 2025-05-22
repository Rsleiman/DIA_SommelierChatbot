from pprint import pprint
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


docs = SimpleDirectoryReader("data").load_data()

# pprint(docs[0].__dict__)

# print("\n\n\nBEFORE")
# print(docs[0].get_content(metadata_mode=MetadataMode.LLM))

# for doc in docs:
#     doc.text_template = "Metadata:\n{metadata_str}\n------\nContent:\n{content}"

#     ## These exclude keys determine what metadata is passed into the embeddings and LLM. Any potentially useful metadata should be included.
#     if "page_label" not in doc.excluded_embed_metadata_keys:
#         doc.excluded_embed_metadata_keys.append("page_label")

#     if "file_name" in doc.excluded_llm_metadata_keys:
#         doc.excluded_llm_metadata_keys.remove("file_name")

# print("\n\n\nAFTER")
# print(docs[0].get_content(metadata_mode=MetadataMode.LLM))


import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
llm_transformer = OpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(
    separator=" ", chunk_size=1000, chunk_overlap=100
)
title_extractor = TitleExtractor(llm=llm_transformer, nodes=3)
qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformer, questions=3)

from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        title_extractor,
        qa_extractor
    ]
)

nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True
)

print(f"Node length: {len(nodes)}")
pprint(nodes[0].__dict__)
print(nodes[0].get_content(metadata_mode=MetadataMode.LLM))


hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
test_embed = hf_embeddings.get_text_embedding("Hello world")
print("\n\n\n")
print(test_embed)
print("\n\n\n")

index = VectorStoreIndex(nodes, embed_model=hf_embeddings)

## Query the index with a query engine ##
query_engine = index.as_query_engine(
    llm=llm_transformer,
)

response = query_engine.query("How do I get a gift card?")
print(f"RESPONSE: \n{response.__dict__}\n\n\n\n")

## Store the index to persistant storage ##
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize client, setting path to save database
db = chromadb.PersistentClient(path="./chroma_db")

# Create colection
chroma_collection = db.get_or_create_collection("quickstart")

# Assign chrome as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex(nodes, storage_context=storage_context)