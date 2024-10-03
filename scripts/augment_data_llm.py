import os
from pathlib import Path

from tqdm import tqdm

from src.augmentors.llm_api.gemini_api import GeminiAPI
from src.augmentors.llm_augmentor import LLMAugmentor


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    print(f"Dataset size: {len(dataset)}")
    dataset = dataset[: int(len(dataset) * 0.01)]

    dataset = batch_data(dataset, 100)
    print(f"Number of batches: {len(dataset)}")

    llm_api = GeminiAPI(
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-1.5-flash-latest",
        base_prompt="I will give you text in Lithuanian. Please slightly rephrase it in Lithuanian, so that words would be different, but the meaning would be the same. Output only one variant in block | |. The text is: ",  # noqa
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


if __name__ == "__main__":
    main("data/en-lt.txt/NLLB.en-lt.lt", "out/NLLB.en-lt.lt")
