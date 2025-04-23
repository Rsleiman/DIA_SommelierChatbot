import getpass
import os

# os.environ["DIA_API_KEY"] = getpass.getpass("OpenAI API Key: ")


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

