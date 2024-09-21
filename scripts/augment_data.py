import os

from src.augmentors.llm_api.gemini_api import GeminiAPI
from src.augmentors.llm_augmentor import LLMAugmentor


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    llm_api = GeminiAPI(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash-latest",
        base_prompt="I will give you text in Lithuanian. Please slightly rephrase it in Lithuanian, so that words would be different, but the meaning would be the same. Output only one variant in block []. The text is: ",  # noqa
        output_regexp=r"\[(.*?)\]",
    )
    augmentor = LLMAugmentor(llm_api)
    augmented_data = augmentor(dataset)

    with open(output_path, "w") as f:
        f.writelines(augmented_data)


if __name__ == "__main__":
    main("data/samples/lithuanian_sample.txt", "out/augmented_dataset.txt")
