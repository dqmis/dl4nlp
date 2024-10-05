import os
from pathlib import Path
import requests

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

default = "en"
SOURCE_LANG = input(f"Enter the source language [{default}]: ") or default
default = "lt"
TARGET_LANG = input(f"Enter the target language [{default}]: ") or default

PATH = Path(__file__).parent.parent / f"data/opus.nllb.{SOURCE_LANG}-{TARGET_LANG}"
print(PATH)
os.makedirs(PATH, exist_ok=True)

# download the dataset from the OPUS website
URL_FORMAT = "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{}-{}.txt.zip"
URL = None
for url in [
    URL_FORMAT.format(SOURCE_LANG, TARGET_LANG),
    URL_FORMAT.format(TARGET_LANG, SOURCE_LANG),
]:
    try:
        print(f"Checking if '{url}' exists...")
        resp = requests.head(url)
        resp.raise_for_status()
        URL = url
        print(f"Found the dataset at '{url}'.")
        break
    except requests.HTTPError:
        print(f"URL '{url}' does not exist.")

if URL is None:
    print("Could not find the dataset. Please check the languages.")
    exit()

ZIP_PATH = f"{PATH}/{SOURCE_LANG}-{TARGET_LANG}.txt.zip"
if not os.path.exists(ZIP_PATH):
    print("Downloading the dataset...")
    resp = os.system(f"wget {URL} -O {ZIP_PATH}")
    if resp != 0:
        print(f"Failed to download the dataset. Please check the URL. Response code: {resp}")
        os.remove(ZIP_PATH)
        exit()
else:
    print("Dataset already downloaded.")

UNZIP_PATH = f"{PATH}/{SOURCE_LANG}-{TARGET_LANG}.txt"
if not os.path.exists(UNZIP_PATH):
    print("Unzipping the dataset...")
    os.system(f"unzip {ZIP_PATH} -d {UNZIP_PATH}")
else:
    print("Dataset already unzipped.")

# read first three lines of the dataset
files = {}
for file_prefix in (f"NLLB.{SOURCE_LANG}-{TARGET_LANG}", f"NLLB.{TARGET_LANG}-{SOURCE_LANG}"):
    for suff in [SOURCE_LANG, TARGET_LANG, "scores"]:
        file = f"{UNZIP_PATH}/{file_prefix}.{suff}"
        if not os.path.exists(file):
            print(f"Could not find '{file}'.")
            continue
        files[suff] = file
        with open(file) as f:
            print(f"First three lines of '{file}':")
            for i in range(3):
                print(f.readline().strip())
            print()
assert all(
    suff in files for suff in [SOURCE_LANG, TARGET_LANG, "scores"]
), "Some files are missing."

PARQUET_FILE = f"{PATH}/{SOURCE_LANG}-{TARGET_LANG}.parquet"
if os.path.exists(PARQUET_FILE):
    print(f"Dataset already saved to a parquet file ({PARQUET_FILE}).")
    if input("Do you want to overwrite it? (y/N): ").lower() != "y":
        exit()

# read the dataset into a pandas DataFrame
data = {}
for suff, file in files.items():
    print(f"Reading '{file}'...")
    with open(f"{file}") as f:
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
        pa.field(SOURCE_LANG, pa.string()),
        pa.field(TARGET_LANG, pa.string()),
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
