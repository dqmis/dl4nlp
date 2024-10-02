import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

PATH = Path(__file__).parent.parent / "data/opus.nllb.en-lt"
print(PATH)
os.makedirs(PATH, exist_ok=True)

# download the dataset from the OPUS website
URL = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-lt.txt.zip"

ZIP_PATH = f"{PATH}/en-lt.txt.zip"
if not os.path.exists(ZIP_PATH):
    print("Downloading the dataset...")
    os.system(f"wget {URL} -O {ZIP_PATH}")
else:
    print("Dataset already downloaded.")

UNZIP_PATH = f"{PATH}/en-lt.txt"
if not os.path.exists(UNZIP_PATH):
    print("Unzipping the dataset...")
    os.system(f"unzip {ZIP_PATH} -d {UNZIP_PATH}")
else:
    print("Dataset already unzipped.")

# read first three lines of the dataset
FILE_PREFIX = "NLLB.en-lt"
for suff in ["en", "lt", "scores"]:
    with open(f"{UNZIP_PATH}/{FILE_PREFIX}.{suff}") as f:
        print(f"First three lines of '{UNZIP_PATH}/en-lt.txt.{suff}':")
        for i in range(3):
            print(f.readline().strip())
        print()

PARQUET_FILE = f"{PATH}/en-lt.parquet"
if os.path.exists(PARQUET_FILE):
    print(f"Dataset already saved to a parquet file ({PARQUET_FILE}).")
    if input("Do you want to overwrite it? (y/N): ").lower() != "y":
        exit()

# read the dataset into a pandas DataFrame
data = {}
for suff in ["en", "lt", "scores"]:
    print(f"Reading '{UNZIP_PATH}/{FILE_PREFIX}.{suff}'...")
    with open(f"{UNZIP_PATH}/{FILE_PREFIX}.{suff}") as f:
        data[suff] = tuple(line.strip() for line in f.readlines())
data["scores"] = tuple(float(score) for score in data["scores"])

print("Creating a pandas DataFrame...")
df = pd.DataFrame(data)
df["scores"] = df["scores"].astype("float16")  # Ensure scores are float16
print(df)
del data

# save the dataset to a parquet file
print(f"Saving the dataset to '{PARQUET_FILE}'...")

schema = pa.schema(
    [
        pa.field("en", pa.string()),
        pa.field("lt", pa.string()),
        pa.field("scores", pa.float16()),
    ]
)
writer = pq.ParquetWriter(PARQUET_FILE, schema)

# save it in chunks to avoid memory errors
CHUNK_SIZE = 10**5
for i in tqdm.tqdm(range(0, len(df), CHUNK_SIZE), desc="Writing chunks"):
    table = pa.Table.from_pandas(df[i : i + CHUNK_SIZE])
    writer.write_table(table)

writer.close()
