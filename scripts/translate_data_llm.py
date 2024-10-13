import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from src.augmentors.llm_api.gemini_api import GeminiAPI
from src.augmentors.llm_augmentor import LLMAugmentor

load_dotenv()


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def load_flores_dataset(source_lang: str, target_lang: str) -> dict:
    dataset = load_dataset(
        "facebook/flores",
        f"{source_lang}_Latn-{target_lang}_Latn",
        trust_remote_code=True,
    )
    return dataset["devtest"]["sentence_eng_Latn"]


def main(dataset_path: str, output_path: str, lang_to: str) -> None:
    print("Loading dataset...")

    with open(dataset_path) as f:
        dataset = f.readlines()

    dataset_size = len(dataset)

    # Take about 250_000 of the dataset
    dataset = dataset[:250_000]

    print(
        f"Dataset size: {len(dataset)}. Old dataset size: {dataset_size}. Fraction: {len(dataset) / dataset_size}"
    )

    llm_api = GeminiAPI(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash-latest",
        base_prompt=f"I will give you text in English. Please translate it to {lang_to}. Output only one variant in block | |. The text is: ",  # noqa
        output_regexp=r"\|(.*?)\|",
    )

    augmentor = LLMAugmentor(llm_api)
    print("Translating the dataset...")

    os.makedirs(output_path, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(batch_data(dataset, 100))):
        augmented_data = augmentor(batch)

        write_path = Path().resolve() / output_path / f"{batch_idx}.txt"
        write_path.parent.mkdir(parents=True, exist_ok=True)

        with open(write_path, "w") as f:
            f.writelines(augmented_data)

    print("Finished translating the dataset!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lang_to", type=str, required=True)

    args = parser.parse_args()
    main(args.dataset_path, args.output_path, args.lang_to)
