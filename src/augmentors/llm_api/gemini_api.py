import re
import time
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory

from src.augmentors.llm_api.base_llm_api import BaseLLMApi


class GeminiAPI(BaseLLMApi):
    def __init__(self, api_key: str, model: str, base_prompt: str, output_regexp: str) -> None:
        genai.configure(api_key=api_key)

        self._model = genai.GenerativeModel(model)
        self._base_prompt = base_prompt
        self._output_regexp = re.compile(output_regexp, re.DOTALL)

    def _format_prompt(self, prompt: str) -> str:
        return self._base_prompt + " " + prompt

    def _generate_for_sample(
        self, sample: str, max_retries: int = 3, delay: float = 1.0
    ) -> list[str]:
        attempt = 0
        while attempt < max_retries:
            try:
                generated_text = self._model.generate_content(
                    self._format_prompt(sample),
                    safety_settings={
                        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    },
                ).text

                matches = self._output_regexp.findall(generated_text)
                if len(matches) > 1:
                    matches = matches[:1]
                return [match.strip() + "\n" for match in matches]
            except Exception as e:
                attempt += 1
                print(
                    f"Attempt {attempt}: Error occurred with message '{e}'. Retrying after {delay} seconds..."
                )
                time.sleep(delay)

        print(f"Failed to process sample after {max_retries} attempts. Returning original text.")
        return [sample.strip() + "\n"]  # Return the original text as a fallback

    def _prompt(self, data: list[str]) -> list[str]:
        output: list[str] = []

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._generate_for_sample, data))

        for result in results:
            output.extend(result)

        return output
