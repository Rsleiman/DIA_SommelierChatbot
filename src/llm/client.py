from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
import logfire



api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found.")


def get_llm() -> OpenAI:
    llm = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    logfire.info("LLM initialized")
    return llm

llm = get_llm()    
