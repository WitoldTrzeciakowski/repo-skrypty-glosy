import os
import shutil
import random
import re

SOURCE_DIRECTORY = 'spectrograms'
DESTINATION_DIRECTORY = './clean_data'
LOCATORS_SPEAKERS_LIST = ["f1", "f7", "f8", "m3", "m6", "m8"]
SPLIT_RATIOS = {"train": 0.6, "test": 0.2, "eval": 0.2}

def move_files_with_test_split(src_dir, dest_dir):
    # Create dataset directories
    for subset in SPLIT_RATIOS.keys():
        base_dest = os.path.join(dest_dir, subset)
        for speaker in LOCATORS_SPEAKERS_LIST + ["wrong"]:
            os.makedirs(os.path.join(base_dest, speaker), exist_ok=True)

    # Collect files by speaker
    speaker_files = {speaker: [] for speaker in LOCATORS_SPEAKERS_LIST}
    wrong_files = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            match = next((spk for spk in LOCATORS_SPEAKERS_LIST if f"{spk}_" in file), None)
            if match:
                speaker_files[match].append((root, file))
            elif re.match(r'([mf]\d+)_', file):  # Any other speaker prefix
                wrong_files.append((root, file))

    # Function to split and move files
    def split_and_move(files, speaker):
        random.shuffle(files)
        total_files = len(files)
        train_end = int(total_files * SPLIT_RATIOS["train"])
        test_end = train_end + int(total_files * SPLIT_RATIOS["test"])

        splits = {
            "train": files[:train_end],
            "test": files[train_end:test_end],
            "eval": files[test_end:]
        }

        for subset, file_list in splits.items():
            for root, file in file_list:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, subset, speaker, file)
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

    # Process LOCATORS_SPEAKERS_LIST
    for speaker, files in speaker_files.items():
        split_and_move(files, speaker)

    # Process "wrong" category
    split_and_move(wrong_files, "wrong")

# Run the function
move_files_with_test_split(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)
