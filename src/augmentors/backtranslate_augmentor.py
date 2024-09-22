from transformers import pipeline

from src.augmentors.base_augmentor import BaseAugmentor


class BacktranslateAugmentor(BaseAugmentor):
    def __init__(self, lang_from: str, lang_to: str, from_model: str, to_model: str) -> None:
        self._task_from = f"translation_{lang_from}_to_{lang_to}"
        self._task_to = f"translation_{lang_to}_to_{lang_from}"

        self._translator_from = pipeline(self._task_from, model=from_model)
        self._translator_to = pipeline(self._task_to, model=to_model)

    def _augment(self, data: list[str]) -> list[str]:
        translated_text = [i["translation_text"] for i in self._translator_from(data)]
        backtranslated_text = [i["translation_text"] for i in self._translator_to(translated_text)]

        return backtranslated_text
