import glob


def augmentation_valid(lines: list[str]) -> bool:
    # check if there are any empty lines
    if "" in lines:
        return False
    return True


def main(files_path_root: str) -> None:
    all_files_in_path = glob.glob(files_path_root + "/*")
    augmented_files = sorted(
        [filename for filename in all_files_in_path if "original" not in filename]
    )
    augmented_files.sort(key=lambda x: int(x.split(".txt")[0].split("/")[-1]))

    original_files = [filename for filename in all_files_in_path if "original" in filename]
    original_files.sort(key=lambda x: int(x.split("_original")[0].split("/")[-1]))

    output_files = []

    # check if all files are valid
    for idx in range(len(augmented_files)):
        with open(augmented_files[idx]) as f:
            augmented_lines = f.readlines()

        with open(original_files[idx]) as f:
            original_lines = f.readlines()

        if not len(augmented_lines) == len(original_lines):
            output_files.extend(original_lines)
            continue

        if not augmentation_valid(augmented_lines):
            output_files.extend(original_lines)
            continue

        output_files.extend(augmented_lines)

    with open("out/llm-NLLB.en-fr.fr.txt", "w") as f:
        f.writelines(output_files)


if __name__ == "__main__":
    main("out/llm-NLLB.en-fr.fr")
