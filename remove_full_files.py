import os
from concurrent.futures import ThreadPoolExecutor

SOURCE_DIRECTORIES = ['daps'] 

def remove_non_segment_files(directory):
    for root, dirs, files in os.walk(directory):
        if not dirs:
            for file in files:
                if file.endswith(".wav") and "segment" not in file:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")

def find_final_directories(src_dirs):
    final_dirs = []
    for directory in src_dirs:
        for root, dirs, _ in os.walk(directory):
            if not dirs:
                final_dirs.append(root)
    return final_dirs

def process_directory(directory):
    remove_non_segment_files(directory)

def recursively_remove_files(src_dirs):
    final_dirs = find_final_directories(src_dirs)
    with ThreadPoolExecutor() as executor:
        executor.map(process_directory, final_dirs)

if __name__ == "__main__":
    recursively_remove_files(SOURCE_DIRECTORIES)
