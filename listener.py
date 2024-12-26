import sounddevice as sd
import numpy as np
import librosa
import threading
import os
from datetime import datetime
import matplotlib.pyplot as plt
import soundfile as sf

AUDIO_DIR = 'NewAudio'

# Parameters
sr = 44100
channels = 1
dtype = 'float32'
is_recording = True


def stop_recording():
    """Function to stop the recording when a key is pressed."""
    global is_recording
    input("Press Enter to stop recording...\n")
    is_recording = False


def record_audio():
    """Function to record audio until stopped."""
    global is_recording
    print("Recording... Press Enter to stop.")
    audio_buffer = []
    def callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        if is_recording:
            audio_buffer.append(indata.copy())
        else:
            raise sd.CallbackStop()
    with sd.InputStream(samplerate=sr, channels=channels, dtype=dtype, callback=callback):
        while is_recording:
            sd.sleep(100)
    print("Recording stopped!")
    audio = np.concatenate(audio_buffer, axis=0).flatten()
    return audio


def process_audio_file(audio, sr):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{timestamp}_audio.wav"
    output_path = os.path.join(AUDIO_DIR, output_filename)
    sf.write(output_path, audio, sr)
    print(f"Audio file saved to: {output_path}")



stop_thread = threading.Thread(target=stop_recording)
stop_thread.start()


audio = record_audio()


audio = librosa.util.normalize(audio)

print(f"Audio recorded: {audio.shape}, Sampling rate: {sr}")

process_audio_file(audio, sr)

