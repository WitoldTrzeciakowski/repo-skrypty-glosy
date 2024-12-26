import os
import glob

def remove_files_in_directory(directory):
    # Get all files in the directory
    files = glob.glob(os.path.join(directory, "*"))
    for file in files:
        try:
            os.remove(file)
            print(f"Removed: {file}")
        except OSError as e:
            print(f"Error deleting {file}: {e}")

# Remove files from the specified directories
remove_files_in_directory("NewAudio")
remove_files_in_directory("NewSpec")