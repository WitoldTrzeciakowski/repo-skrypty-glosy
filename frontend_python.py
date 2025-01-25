import os
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from tkinter import Tk, Label, Button, filedialog
from test import process_file, model_df, df_state
import numpy as np

# Sampling frequency and recording duration
freq = 44100  # 44.1kHz
duration = 5  # Record for 5 seconds

class AudioRecorder:
    def __init__(self, duration=5, rate=44100, channels=1):
        self.duration = duration
        self.rate = rate
        self.channels = channels
        self.recording = None

    def start_recording(self):
        """Start recording audio."""
        self.recording = sd.rec(int(self.duration * self.rate),
                                samplerate=self.rate,
                                channels=self.channels)
        print("Recording started...")

    def stop_recording(self, file_path):
        """Stop recording and save the audio to a file."""
        sd.wait()  # Wait for the recording to finish
        print("Recording finished.")
        
        # Convert the file path to absolute
        absolute_file_path = os.path.abspath(file_path)

        # Convert to 16-bit (normalize to the range of 16-bit integers)
        scaled_recording = np.int16(self.recording * 32767)  # Scaling to 16-bit range

        # Save as 16-bit WAV using scipy
        write(absolute_file_path, self.rate, scaled_recording)

        # Optionally, save as 16-bit WAV using wavio
        wv.write(absolute_file_path.replace(".wav", "_wavio.wav"), scaled_recording, self.rate, sampwidth=2)
        print(f"Recording saved to {absolute_file_path}.")

        return absolute_file_path

    def get_recording(self):
        return self.recording

# Initialize the recorder
recorder = AudioRecorder()

def start_record():
    recorder.start_recording()

def stop_and_save():
    """Stop recording and save the file."""
    output_file = "recorded_audio.wav"
    output_file = recorder.stop_recording(output_file)
    result_label.config(text=f"Audio saved to {output_file}. Ready for processing.", fg="blue")
    return output_file

def process_saved_file():
    """Process the saved audio file."""
    output_file = "recorded_audio.wav"
    if os.path.exists(output_file):
        process_and_display_result(output_file)
    else:
        result_label.config(text="No recorded file to process.", fg="red")

def process_and_display_result(file_path):
    """Process the audio file and display the result."""
    print("Processing the recorded audio...")
    result = process_file(file_path, model_df, df_state)

    if result:
        result_label.config(text="The audio is classified as GOOD.", fg="green")
    else:
        result_label.config(text="The audio is classified as BAD.", fg="red")

def select_and_process():
    """Select an audio file and process it."""
    file_path = filedialog.askopenfilename(filetypes=[["WAV files", "*.wav"]])
    if file_path:
        absolute_file_path = os.path.abspath(file_path)
        process_and_display_result(absolute_file_path)
    print("FINISHED")

# Create the GUI
root = Tk()
root.title("Audio Classifier")

Label(root, text="Audio Classification Tool", font=("Arial", 16)).pack(pady=10)

start_button = Button(root, text="Start Recording", font=("Arial", 14), command=start_record)
start_button.pack(pady=5)

stop_button = Button(root, text="Stop Recording", font=("Arial", 14), command=stop_and_save)
stop_button.pack(pady=5)

classify_button = Button(root, text="Classify Recording", font=("Arial", 14), command=process_saved_file)
classify_button.pack(pady=5)

select_button = Button(root, text="Select and Classify", font=("Arial", 14), command=select_and_process)
select_button.pack(pady=5)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
