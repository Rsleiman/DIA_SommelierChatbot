from pprint import pprint
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
data_path = script_dir.parent / "src" / "data"

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

docs = SimpleDirectoryReader(data_path, exclude=[data_path / "extra menus"]).load_data()

for doc in docs:
    # Exclusions from embeddings
    excluded_keys = [
        'page_label', 'file_name', 'file_path', 'file_type',
        'file_size', 'creation_date', 'last_modified_date'
    ]
    doc.excluded_embed_metadata_keys = excluded_keys
    doc.excluded_llm_metadata_keys = excluded_keys

    doc.text_template = "{content}"


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
    separator="\n", chunk_size=80, chunk_overlap=20
)
# title_extractor = TitleExtractor(llm=llm_transformer, nodes=3)
# qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformer, questions=3)

from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        # title_extractor,
        # qa_extractor
    ]
)

nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True
)

print(f"Node length: {len(nodes)}")
pprint(nodes[0].__dict__)
# print(nodes[0].get_content(metadata_mode=MetadataMode.LLM))

## TEMPORARY VECTOR STORE ##
# hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# test_embed = hf_embeddings.get_text_embedding("Hello world")
# print("\n\n\n")
# print(test_embed)
# print("\n\n\n")

# index = VectorStoreIndex(nodes, embed_model=hf_embeddings)

# ## Query the index with a query engine ##
# query_engine = index.as_query_engine(
#     llm=llm_transformer,
# )

# response = query_engine.query("How do I get a gift card?")
# print(f"RESPONSE: \n{response.__dict__}\n\n\n\n")




## PERSISTANT VECTOR STORE ##
hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize client, setting path to save database
db = chromadb.PersistentClient(path=(data_path.parent / ".chroma").__str__())

# Create colection
chroma_collection = db.get_or_create_collection("quickstart") #TODO: Change name to "menus"

# Assign chrome as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=hf_embeddings)