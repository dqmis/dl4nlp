from tqdm import tqdm

from src.augmentors.backtranslate_augmentor import BacktranslateAugmentor


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    print(f"Data loaded with {len(dataset)} lines")

    augmentor = BacktranslateAugmentor(
        lang_from="lt",
        lang_to="en",
        from_model="Helsinki-NLP/opus-mt-fr-en",
        to_model="Helsinki-NLP/opus-mt-en-fr",
    )

    for idx, batch in enumerate(tqdm(batch_data(dataset, 512))):
        augmented_batch = augmentor(batch)

        with open(f"{output_path}/{idx}.txt", "w") as f:
            f.writelines(s + "\n" for s in augmented_batch)


if __name__ == "__main__":
    main("data/NLLB.en-fr.en-sampled", "out/backtranslated-fr")
