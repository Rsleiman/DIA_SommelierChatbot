from pprint import pprint
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode

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
    model="gpt-3.5-turbo",
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