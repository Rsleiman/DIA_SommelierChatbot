from typing import Any, Dict, List, Optional, Sequence, cast

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, run_jobs
from llama_index.core.bridge.pydantic import (
    Field,
    SerializeAsAny,
)
from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.settings import Settings

DEFAULT_WINE_FOOD_CHAR_TMPL = """
Here is the context:
{context_str}


Identify any wine or food items mentioned in the context.

For each wine, extract its characteristics.
You are encouraged to use the following vocabulary for wine characteristics:
"body": "light-bodied", "medium-bodied", "full-bodied",
"acidity": "low acidity", "medium acidity", "high acidity",
"tannin": "low tannin", "medium tannin", "high tannin",
"sweetness": "dry", "sweet",
"colour": "white", "red", "rosé",
"flavour": "fruity", "floral", "herbaceous", "spicy", "earthy", "oaky", "toasty", "nutty", "smoky", "vegetal", "citrus",

For each food, extract its characteristics (such as meat type, creaminess, heaviness, or anything that is associated with the dish.).
You are encouraged to infer characteristics from the dish name and description (e.g., 'duck confit with a gravy' is fatty, rich, savoury)

Return your answer as a JSON object with two keys: "wine_characteristics" and "food_characteristics". 
Each key should map to a list of strings, where each string describes the wine or food item with its characteristics.

Example:
{
  "wine_characteristics": [
    "Chardonnay - white, medium-bodied, medium acidity, ...",
    ...
  ],
  "food_characteristics": [
    "duck confit with cream sauce - poultry, rich, creamy, ...",
    ...
  ]
}

If you don't identify any obvious wine or food items in the context, return empty lists for the key's values.
NOTE: The context may begin or end mid-entry due to text splitting, and might not include full names or descriptions of wines or dishes. Ignore incomplete fragments and only extract information when items can be confidently identified.
"""

class WineFoodCharacteristicsExtractor(BaseExtractor):
    """
    Extracts wine and food characteristics from a node's content.
    Adds 'wine_characteristics' and 'food_characteristics' metadata fields.
    """
    llm: SerializeAsAny[LLM] = Field(description="The LLM to use for extraction.")
    prompt_template: str = Field(
        default=DEFAULT_WINE_FOOD_CHAR_TMPL,
        description="Prompt template for extracting wine and food characteristics.",
    )
    embedding_only: bool = Field(
        default=True, description="Whether to use metadata for embeddings only."
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        prompt_template: str = DEFAULT_WINE_FOOD_CHAR_TMPL,
        embedding_only: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            llm=llm or Settings.llm, # type: ignore
            prompt_template=prompt_template, # type: ignore
            embedding_only=embedding_only, # type: ignore
            num_workers=num_workers,
            **kwargs,
        ) 

    @classmethod
    def class_name(cls) -> str:
        return "WineFoodCharacteristicsExtractor"

    async def _aextract_characteristics_from_node(self, node: BaseNode) -> Dict[str, Any]:
        if self.is_text_node_only and not isinstance(node, TextNode):
            return {}
        context_str = node.get_content(metadata_mode=self.metadata_mode)
        prompt = PromptTemplate(template=self.prompt_template)
        response = await self.llm.apredict(prompt, context_str=context_str)
        # Try to parse the response as JSON
        import json
        cleaned = response.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):].strip()

        # Load JSON
        parsed = json.loads(cleaned)

        wine_text = "\n".join(parsed["wine_characteristics"])
        food_text = "\n".join(parsed["food_characteristics"])  # Will be empty in your case

        characteristics = {"wine_characteristics": wine_text, "food_characteristics": food_text}

        return characteristics

    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        jobs = [self._aextract_characteristics_from_node(node) for node in nodes]
        metadata_list: List[Dict] = await run_jobs(
            jobs, show_progress=self.show_progress, workers=self.num_workers
        )
        return metadata_list
