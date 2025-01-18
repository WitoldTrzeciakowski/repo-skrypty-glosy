import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Path to the extracted audio files
extract_path = "C:/Users/66985/Downloads/daps/daps/cleanraw"
workspace_path = r"C:\Warsaw University of Technology\Sem5\IMLProject\venv"

# Step 1: Load the audio files
audio_files = []
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(".wav") and not file.startswith("._"):
            audio_files.append(os.path.join(root, file))

print(f"Found {len(audio_files)} audio files.")

# Step 2: Analyze the sampling rate characteristics
sampling_rates = []
for audio_file in audio_files:
    try:
        data, samplerate = sf.read(audio_file)
        sampling_rates.append(samplerate)
    except RuntimeError as e:
        print(f"Error reading {audio_file}: {e}")

# Step 3: Generate and save the sampling rate histogram
plt.figure(figsize=(10, 6))
plt.hist(sampling_rates, bins=30, edgecolor='black')
plt.title('Sampling Rate Characteristics of Audio Files')
plt.xlabel('Sampling Rate (Hz)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(r"C:\Warsaw University of Technology\Sem5\IMLProject\venv\sampling_rate_characteristics.png")
plt.show()
print("Sampling rate histogram saved.")

# Step 4: Visualize the Power Spectral Density (PSD) for the first audio file
if audio_files:
    for audio_file in audio_files:
        try:
            data, samplerate = sf.read(audio_file)
            break
        except RuntimeError as e:
            print(f"Error reading {audio_file}: {e}")
            continue
    
    # Power Spectral Density
    plt.figure(figsize=(10, 6))
    plt.psd(data, NFFT=2048, Fs=samplerate, scale_by_freq=True)
    plt.title('Power Spectral Density of the Audio File')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.savefig(r"C:\Warsaw University of Technology\Sem5\IMLProject\venv\power_spectral_density.png")
    plt.show()
    print("Power Spectral Density visualization saved.")