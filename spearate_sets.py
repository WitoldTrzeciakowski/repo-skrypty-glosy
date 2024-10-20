import os
import shutil
import re

SOURCE_DIRECTORY = 'spectrograms'
DESTINATION_DIRECTORY = './clean_data'
LOCATORS_SPEAKERS_LIST = ["f1", "f7", "f8", "m3", "m6", "m8"]

def move_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    speaker_files = {speaker: [] for speaker in LOCATORS_SPEAKERS_LIST}
    other_speakers_files = {}

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            res = [ele for ele in LOCATORS_SPEAKERS_LIST if ele + "_" in file]
            if res:
                res = ''.join(res)
                speaker_files[res].append((root, file))
            else:
                match = re.match(r'([mf]\d+)_', file)
                if match:
                    prefix = match.group(1)
                    if prefix not in LOCATORS_SPEAKERS_LIST:
                        other_speakers_files.setdefault(prefix, []).append((root, file))

    for speaker, files in speaker_files.items():
        split_index = int(len(files) * 0.8)
        for i, (root, file) in enumerate(files):
            subset = 'test' if i < split_index else 'eval'
            src_file_path = os.path.join(root, file)
            temp_dest = os.path.join(dest_dir, speaker, subset)
            if not os.path.exists(temp_dest):
                os.makedirs(temp_dest)
            dest_file_path = os.path.join(temp_dest, file)
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved: {src_file_path} to {dest_file_path}")

    for speaker, files in other_speakers_files.items():
        split_index = int(len(files) * 0.8)
        for i, (root, file) in enumerate(files):
            subset = 'test' if i < split_index else 'eval'
            src_file_path = os.path.join(root, file)
            temp_dest = os.path.join(dest_dir, 'wrong', speaker, subset)
            if not os.path.exists(temp_dest):
                os.makedirs(temp_dest)
            dest_file_path = os.path.join(temp_dest, file)
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved: {src_file_path} to {dest_file_path}")

move_files(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)
