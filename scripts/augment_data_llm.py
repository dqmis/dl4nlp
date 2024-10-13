import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from src.augmentors.llm_api.gemini_api import GeminiAPI
from src.augmentors.llm_augmentor import LLMAugmentor

load_dotenv()


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main(dataset_path: str, output_path: str, language_from: str) -> None:
    print("Starting data augmentation for LLM ...")
    with open(dataset_path) as f:
        dataset = f.readlines()

    print(f"Dataset size: {len(dataset)}")
    # Take size of 250_000
    dataset = dataset[:250_000]

    dataset = batch_data(dataset, 100)
    print(f"Number of batches: {len(dataset)}")

    llm_api = GeminiAPI(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash-latest",
        base_prompt=f"I will give you text in {language_from}. Please slightly rephrase it in {language_from}, so that words would be different, but the meaning would be the same. Output only one variant in block | |. The text is: ",  # noqa
        output_regexp=r"\|(.*?)\|",
    )
    augmentor = LLMAugmentor(llm_api)

    for batch_idx, batch in enumerate(tqdm(dataset)):
        augmented_data = augmentor(batch)

        write_path = Path().resolve() / output_path / f"{batch_idx}.txt"
        write_path.parent.mkdir(parents=True, exist_ok=True)

        with open(write_path, "w") as f:
            f.writelines(augmented_data)

        # also write original data
        write_path = Path().resolve() / output_path / f"{batch_idx}_original.txt"
        write_path.parent.mkdir(parents=True, exist_ok=True)

        with open(write_path, "w") as f:
            f.writelines(batch)

    print("Data augmentation for LLM done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lang_from", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.lang_from)
