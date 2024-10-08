from tqdm import tqdm
from transformers import pipeline

from src.augmentors.base_augmentor import BaseAugmentor


class BacktranslateAugmentor(BaseAugmentor):
    def __init__(self, lang_from: str, lang_to: str, from_model: str, to_model: str) -> None:
        self._task_from = f"translation_{lang_from}_to_{lang_to}"
        self._task_to = f"translation_{lang_to}_to_{lang_from}"

        self._translator_from = pipeline(self._task_from, model=from_model, device=0, batch_size=32)
        self._translator_to = pipeline(self._task_to, model=to_model, device=0, batch_size=32)

    def _augment(self, data: list[str]) -> list[str]:
        translated_texts = [
            result["translation_text"] for result in self._translator_from(data)
        ]
        backtranslated_texts = [
            result["translation_text"] for result in self._translator_to(translated_texts)
        ]
        return backtranslated_texts
