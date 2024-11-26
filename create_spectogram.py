import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
from concurrent.futures import ThreadPoolExecutor

SOURCE_DIRECTORIES = ['stash']
SPECTROGRAM_DIR = 'spectrograms'

def save_spectrogram(audio, sr, output_path):
    """
    Save the spectrogram of the given audio signal to the output path.
    """
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved to: {output_path}")
    return output_path

def process_audio_file(audio_path, dest_root):
    """
    Process a single audio file: generate and save its spectrogram.
    """
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=None)

        # Define the output path, mirroring the source directory structure
        relative_path = os.path.relpath(audio_path, start=SOURCE_DIRECTORIES[0])
        output_dir = os.path.join(dest_root, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_spectrogram.png"
        output_path = os.path.join(output_dir, output_filename)

        # Save the spectrogram
        return save_spectrogram(audio, sr, output_path)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def process_final_directories(src_dir, dest_dir):
    """
    Process final directories containing audio files.
    """
    for root, dirs, files in os.walk(src_dir):
        if not dirs:  # Process only final directories (no subdirectories)
            for file in files:
                if file.endswith(".wav") and not file.startswith('.'):  
                    audio_path = os.path.join(root, file)
                    process_audio_file(audio_path, dest_dir)

def recursively_generate_spectrograms(src_dirs, dest_dir):
    """
    Generate spectrograms for all `.wav` files in the source directories.
    """
    with ThreadPoolExecutor() as executor:
        for src_dir in src_dirs:
            executor.submit(process_final_directories, src_dir, dest_dir)

if __name__ == "__main__":
    recursively_generate_spectrograms(SOURCE_DIRECTORIES, SPECTROGRAM_DIR)
