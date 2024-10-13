import os
import glob


def augmentation_valid(lines: list[str]) -> bool:
    # check if there are any empty lines
    if "" in lines:
        return False
    return True


def main(dataset_path: str, outut_file_path: str) -> None:
    print("Starting to join files!")

    all_files_in_path = glob.glob(dataset_path + "/*")
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

    os.makedirs(os.path.dirname(outut_file_path), exist_ok=True)

    with open(outut_file_path, "w") as f:
        f.writelines(output_files)

    print("Finished joining files!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset_path, args.output_file_path)
