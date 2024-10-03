from tqdm import tqdm

from src.augmentors.backtranslate_augmentor import BacktranslateAugmentor


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    dataset = dataset[:100]
    print(f"Data loaded with {len(dataset)} lines")

    augmentor = BacktranslateAugmentor(
        lang_from="lt",
        lang_to="en",
        from_model="Helsinki-NLP/opus-mt-fr-en",
        to_model="Helsinki-NLP/opus-mt-en-fr",
    )

    augmented_data = []

    for batch in tqdm(batch_data(dataset, 10)):
        augmented_batch = augmentor(dataset)
        augmented_data.extend(augmented_batch)

    with open(output_path, "w") as f:
        f.writelines(s + "\n" for s in augmented_data)


if __name__ == "__main__":
    main("data/en-lt.txt/NLLB.en-lt.lt", "out/augmented_dataset.txt")
