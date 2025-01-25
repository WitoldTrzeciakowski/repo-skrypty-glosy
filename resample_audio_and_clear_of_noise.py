import librosa
import soundfile as sf
import wave
import os
from df.enhance import enhance, init_df, load_audio, save_audio

SOURCE_DIRECTORIES = ['NewAudio']

def is_valid_wav_file(file_path):
    """Check if the file is a valid WAV file."""
    try:
        with wave.open(file_path, "rb") as wave_file:
            return True
    except wave.Error as e:
        print(f"Error: {e}")  # Print the error message
        return False

def re_sample_audio(audio_path):
    print(f"Resampling audio: {audio_path}")
    if not is_valid_wav_file(audio_path):
        print(f"Skipping invalid WAV file: {audio_path}")
        return 
    with wave.open(audio_path, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        audio, _ = librosa.load(audio_path, sr=frame_rate)
        audio_resampled = librosa.resample(audio, target_sr=48000, orig_sr=frame_rate)
        sf.write(audio_path, audio_resampled, 48000)
        return audio_path
    print(f"Resampling done on file: {audio_path}")

def recursively_resample(src_dirs):
    for directory in src_dirs:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".wav"):
                    audio_path = os.path.join(root, file) 
                    re_sample_audio(audio_path)

def recursively_delete_noise(src_dirs, model, df_state):
    for directory in src_dirs:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".wav"):  
                    
                    audio_path = os.path.join(root, file)  
                    if not is_valid_wav_file(audio_path):
                        print(f"Skipping invalid WAV file: {audio_path}")
                        continue 
                    audio, _ = load_audio(audio_path, sr=df_state.sr())
                    enhanced = enhance(model, df_state, audio)
                    enhanced_audio_path = audio_path.replace('.wav', '_enhanced.wav') 
                    save_audio(enhanced_audio_path, enhanced, df_state.sr())
                    os.remove(audio_path)
                    print(f"Processed and deleted: {audio_path}")

if __name__ == "__main__":
    recursively_resample(SOURCE_DIRECTORIES)
    model, df_state, _ = init_df()
    recursively_delete_noise(SOURCE_DIRECTORIES, model, df_state)
