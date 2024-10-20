import librosa
import soundfile as sf
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SOURCE_DIRECTORIES = ['daps']
MIN_SEGMENT_LENGTH = 12  

def split_and_save_segments(audio, sr, non_silent_intervals, audio_path):
    segment_count = 0
    current_segment = []
    current_duration = 0

    for start, end in non_silent_intervals:
        segment = audio[start:end]
        segment_duration = (end - start) / sr
        if current_duration + segment_duration < MIN_SEGMENT_LENGTH:
            current_segment.append(segment)
            current_duration += segment_duration
        else:
            if current_segment:
                combined_segment = np.concatenate(current_segment)
                segment_count += 1
                output_filename = f"{os.path.splitext(audio_path)[0]}_segment_{segment_count}.wav"
                sf.write(output_filename, combined_segment, sr)
                print(f"Saved segment to: {output_filename}")
            current_segment = [segment]
            current_duration = segment_duration
    if current_segment and current_duration >= MIN_SEGMENT_LENGTH:
        combined_segment = np.concatenate(current_segment)
        segment_count += 1
        output_filename = f"{os.path.splitext(audio_path)[0]}_segment_{segment_count}.wav"
        sf.write(output_filename, combined_segment, sr)
        print(f"Saved segment to: {output_filename}")

def process_audio_file(audio_path, top_db=60):
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        non_silent_intervals = librosa.effects.split(trimmed_audio, top_db=top_db)
        split_and_save_segments(trimmed_audio, sr, non_silent_intervals, audio_path)
    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")

def process_final_directory(directory):
    print("processing" + directory)
    for file in os.listdir(directory):
        if file.endswith(".wav") and not file.startswith('.'):
            audio_path = os.path.join(directory, file)
            process_audio_file(audio_path)

def recursively_process_final_directories(src_dirs):
    final_directories = []
    for directory in src_dirs:
        for root, dirs, files in os.walk(directory):
            if not dirs:
                final_directories.append(root)
    with ThreadPoolExecutor() as executor:
        executor.map(process_final_directory, final_directories)

recursively_process_final_directories(SOURCE_DIRECTORIES)
