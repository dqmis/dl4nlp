import glob
import pandas as pd

if __name__ == "__main__":
    backtranslation_directory = "out/backtranslated-ee"
    output_file_name = "out/nllb-ee-backtranslated.txt"
    parquet_file_name = "data/bt-opus.nllb.en-ee/nllb-ee-backtranslated.parquet"

    files = glob.glob(f"{backtranslation_directory}/*")
    files.sort(key=lambda x: int(x.split(".txt")[0].split("/")[-1]))

    all_data = []

    for idx, file in enumerate(files):
        with open(file, "r") as f:
            data = f.readlines()

        if len(data) != 100:
            with open(f"{backtranslation_directory}/{idx}.txt", "r") as f:
                org_data = f.readlines()

            all_data.extend(org_data)
            continue

        all_data.extend(data)

    # Write the data to a text file
    with open(output_file_name, "w") as f:
        f.writelines(all_data)

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(all_data, columns=["text"])

    # Save the DataFrame to a Parquet file
    df.to_parquet(parquet_file_name)

    print(f"Parquet file saved as {parquet_file_name}")
