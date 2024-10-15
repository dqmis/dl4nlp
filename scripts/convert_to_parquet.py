import os
import glob
import pandas as pd


def main(
    data_dir: str, original_data_dir: str, output_parquet_file: str, original_lang: str
) -> None:
    print(f"Reading data from: {data_dir}")

    files = glob.glob(f"{data_dir}/*")
    files.sort(key=lambda x: int(x.split(".txt")[0].split("/")[-1]))

    all_data = []

    for idx, file in enumerate(files):
        with open(file, "r") as f:
            data = f.readlines()

        if len(data) != 100:
            with open(f"{data_dir}/{idx}.txt", "r") as f:
                org_data = f.readlines()

            all_data.extend(org_data)
            continue

        all_data.extend(data)

    # Read the original sentences
    with open(original_data_dir, "r") as f:
        original_data = f.readlines()

    # Create a DataFrame with the backtranslated and original data
    bt_df = pd.DataFrame(all_data, columns=["en"])
    orig_df = pd.DataFrame(original_data, columns=[original_lang]).iloc[: bt_df.shape[0]]

    assert bt_df.shape == orig_df.shape, "Data shapes do not match"

    print(f"Backtranslated data shape: {bt_df.shape}")
    print(f"Original data shape: {orig_df.shape}")

    # Concatenate the two dataframes into one
    df = pd.concat([orig_df, bt_df], axis=1)

    # Create directory if it does not exist
    os.makedirs(os.path.dirname(output_parquet_file), exist_ok=True)

    df.to_parquet(output_parquet_file)

    print(f"Parquet file saved at: {output_parquet_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--original_data_dir", type=str, required=True)
    parser.add_argument("--output_parquet_file", type=str, required=True)
    parser.add_argument("--orig_lang", type=str, required=True)

    args = parser.parse_args()

    main(args.data_dir, args.original_data_dir, args.output_parquet_file, args.orig_lang)
