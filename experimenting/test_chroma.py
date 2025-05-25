from pprint import pprint
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
import sys

script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
sys.path.append(str(Path(__file__).parent.parent))

from src.RAG.query_chroma import get_retriever

retriever = get_retriever(".chroma_enriched")
chunks = retriever.retrieve("white, medium-bodied, high acidity, sweet")

texts = [chunk.text for chunk in chunks]

for text in texts:
    print(text)
    print("\n" + "="*80 + "\n")