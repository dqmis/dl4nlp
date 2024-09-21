from abc import ABC, abstractmethod


class BaseAugmentor(ABC):
    def __call__(self, data: list[str]) -> list[str]:
        return self._augment(data)

    @abstractmethod
    def _augment(self, data: list[str]) -> list[str]:
        pass
