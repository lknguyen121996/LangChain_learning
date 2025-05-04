from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, List, Any
from langchain.schema import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print("LLM started")
        print(prompts[0])
        print("--------------------------------")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print("LLM ended")
        print(response.generations[0][0].text)
        print("--------------------------------")
