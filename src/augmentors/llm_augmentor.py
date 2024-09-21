from src.augmentors.base_augmentor import BaseAugmentor
from src.augmentors.llm_api.base_llm_api import BaseLLMApi


class LLMAugmentor(BaseAugmentor):
    def __init__(self, llm_api: BaseLLMApi) -> None:
        self._llm_api = llm_api

    def _augment(self, data: list[str]) -> list[str]:
        return self._llm_api.prompt(data)
