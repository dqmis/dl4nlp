import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.augmentors.llm_api.gemini_api import GeminiAPI
from src.augmentors.llm_augmentor import LLMAugmentor


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def load_flores_dataset(source_lang: str, target_lang: str) -> dict:
    dataset = load_dataset("facebook/flores", f"{source_lang}_Latn-{target_lang}_Latn")
    return dataset["devtest"]["sentence_eng_Latn"]


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    llm_api = GeminiAPI(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash-latest",
        base_prompt="I will give you text in English. Please translate it to French. Output only one variant in block | |. The text is: ",  # noqa
        output_regexp=r"\|(.*?)\|",
    )

    augmentor = LLMAugmentor(llm_api)

    for batch_idx, batch in enumerate(tqdm(batch_data(dataset, 100))):
        augmented_data = augmentor(batch)

        write_path = Path().resolve() / output_path / f"{batch_idx}.txt"
        write_path.parent.mkdir(parents=True, exist_ok=True)

        with open(write_path, "w") as f:
            f.writelines(augmented_data)


if __name__ == "__main__":
    main("data/en-fr.txt/NLLB.en-fr.en", "out/llm-tr-NLLB.en-fr.fr")
