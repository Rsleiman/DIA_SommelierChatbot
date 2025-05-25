from pathlib import Path
import sys
script_dir = Path(__file__).parent.resolve()
src_path = script_dir.parent
sys.path.append(str(script_dir.parent))

import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI
llm_transformer = OpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from RAG.wine_food_characteristics_extractor import WineFoodCharacteristicsExtractor

# Extract data from pdfs
data_path = src_path / "data"
docs = SimpleDirectoryReader(data_path, exclude=[data_path / "extra menus"]).load_data()

# Set matadata settings
for doc in docs:
    # Exclusions from embeddings
    excluded_keys = [
        'page_label', 'file_name', 'file_path', 'file_type',
        'file_size', 'creation_date', 'last_modified_date'
    ]
    doc.excluded_embed_metadata_keys = excluded_keys
    doc.excluded_llm_metadata_keys = excluded_keys

    doc.text_template = "{content}"

# Split docs into chunks
text_splitter = SentenceSplitter(
    separator="\n", chunk_size=80, chunk_overlap=20
)

# Extract Wine and Food characteristics
wf_characteristics_extractor = WineFoodCharacteristicsExtractor(llm=llm_transformer)

 
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        wf_characteristics_extractor,
    ]
)

nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True
)

## Debugging ##
# print(f"Node length: {len(nodes)}")
# pprint(nodes[10].__dict__)

# w_chars = [node.metadata["wine_characteristics"] for node in nodes]
# f_chars = [node.metadata["food_characteristics"] for node in nodes]
# for w in w_chars:
#     print(f"WINE: {w}")
#     print("\n", "-"*50, "\n")


## PERSISTANT VECTOR STORE ##
hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Initialize client, setting path to save database
db = chromadb.PersistentClient(path=(src_path / ".chroma_enriched").__str__())

# Create collection
chroma_collection = db.get_or_create_collection("food_wine_menus")

# Assign chrome as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=hf_embeddings)