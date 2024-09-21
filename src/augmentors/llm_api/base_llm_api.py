from abc import ABC, abstractmethod


class BaseLLMApi(ABC):
    def prompt(self, data: list[str]) -> list[str]:
        return self._prompt(data)

    @abstractmethod
    def _prompt(self, data: list[str]) -> list[str]:
        pass
