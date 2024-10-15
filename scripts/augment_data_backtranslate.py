import os
from tqdm import tqdm

from src.augmentors.backtranslate_augmentor import BacktranslateAugmentor


def batch_data(data: list[str], batch_size: int) -> list[list[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def main(dataset_path: str, output_path: str, lang_from: str, lang_to: str) -> None:
    print(f"Backtranslating {dataset_path} to {output_path} from {lang_from} to {lang_to}")

    with open(dataset_path) as f:
        dataset = f.readlines()

    dataset_size = len(dataset)

    # Take size of 250_000
    dataset = dataset[:250_000]

    print(
        f"Data loaded size {len(dataset)} lines, old dataset size: {dataset_size}. Fraction: {len(dataset) / dataset_size}"
    )

    os.makedirs(output_path, exist_ok=True)

    augmentor = BacktranslateAugmentor(
        lang_from=f"{lang_from}",
        lang_to=f"{lang_to}",
        from_model=f"Helsinki-NLP/opus-mt-{lang_to}-{lang_from}",
        to_model=f"Helsinki-NLP/opus-mt-{lang_from}-{lang_to}",
    )

    for idx, batch in enumerate(tqdm(batch_data(dataset, 512))):
        augmented_batch = augmentor(batch)

        with open(f"{output_path}/{idx}.txt", "w") as f:
            f.writelines(s + "\n" for s in augmented_batch)

    print("Backtranslation done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lang_from", type=str, required=True)
    parser.add_argument("--lang_to", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.lang_from, args.lang_to)
