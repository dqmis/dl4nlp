import re

import google.generativeai as genai

from src.augmentors.llm_api.base_llm_api import BaseLLMApi


class GeminiAPI(BaseLLMApi):
    def __init__(self, api_key: str, model: str, base_prompt: str, output_regexp: str) -> None:
        genai.configure(api_key=api_key)

        self._model = genai.GenerativeModel(model)
        self._base_prompt = base_prompt
        self._output_regexp = re.compile(output_regexp, re.DOTALL)

    def _format_prompt(self, prompt: str) -> str:
        return self._base_prompt + " " + prompt

    def _prompt(self, data: list[str]) -> list[str]:
        output: list[str] = []

        for sample in data:
            generated_text = self._model.generate_content(self._format_prompt(sample)).text
            matches = self._output_regexp.findall(generated_text)
            for match in matches:
                output.append(match + "\n")

        return output
