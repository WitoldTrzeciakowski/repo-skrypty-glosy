import librosa
import os
import matplotlib.pyplot as plt
import librosa.display
import concurrent.futures

SOURCE_DIRECTORIES = ['daps']
SPECTROGRAM_DIR = 'spectrograms' 

def create_spectrogram_directory():
    if not os.path.exists(SPECTROGRAM_DIR):
        os.makedirs(SPECTROGRAM_DIR)

def save_spectrogram(audio, sr, output_path):
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.axis('off') 
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Spectrogram saved to: {output_path}")

def process_audio_file(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    output_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_spectrogram.png"
    output_path = os.path.join(SPECTROGRAM_DIR, output_filename)
    save_spectrogram(audio, sr, output_path)

def process_final_directories(src_dirs):
    for directory in src_dirs:
        for root, dirs, files in os.walk(directory):
            # Process only the final directories
            if not dirs:  # Check if it's a final directory (no subdirectories)
                for file in files:
                    if file.endswith(".wav") and not file.startswith('.'):  
                        audio_path = os.path.join(root, file)
                        process_audio_file(audio_path)

def recursively_generate_spectrograms(src_dirs):
    create_spectrogram_directory()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for directory in src_dirs:
            futures.append(executor.submit(process_final_directories, [directory]))
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    recursively_generate_spectrograms(SOURCE_DIRECTORIES)
