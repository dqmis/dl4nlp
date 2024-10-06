from src.augmentors.backtranslate_augmentor import BacktranslateAugmentor


def main(dataset_path: str, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = f.readlines()

    augmentor = BacktranslateAugmentor(
        lang_from="lt",
        lang_to="en",
        from_model="Helsinki-NLP/opus-mt-fr-en",
        to_model="Helsinki-NLP/opus-mt-en-fr",
    )
    augmented_data = augmentor(dataset)

    with open(output_path, "w") as f:
        f.writelines(s + "\n" for s in augmented_data)


if __name__ == "__main__":
    main("data/samples/NLLB.en-fr.en-sampled", "out/augmented_dataset.txt")
