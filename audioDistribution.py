import os
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

extract_path = "C:/Users/66985/Downloads/daps/daps/cleanraw"
workspace_path = r"C:\Warsaw University of Technology\Sem5\IMLProject\venv"

audio_files = []
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(".wav") and not file.startswith("._"):
            audio_files.append(os.path.join(root, file))

print(f"Found {len(audio_files)} audio files.")

volumes = []
segment_counts = []

for audio_file in audio_files:
    try:
        data, samplerate = sf.read(audio_file)
        volume = np.sqrt(np.mean(data**2))
        volumes.append(volume)
        
        # Calculate number of 3-second segments
        segment_count = len(data) // (3 * samplerate)
        segment_counts.append(segment_count)
    except RuntimeError as e:
        print(f"Error reading {audio_file}: {e}")

plt.figure(figsize=(10, 6))
plt.hist(volumes, bins=30, edgecolor='black')
plt.title('Volume Distribution of Audio Files')
plt.xlabel('Volume (RMS)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(r"C:\Warsaw University of Technology\Sem5\IMLProject\venv\volume_distribution.png")
plt.show()
print("Volume distribution histogram saved.")

plt.figure(figsize=(10, 6))
plt.hist(segment_counts, bins=30, edgecolor='black')
plt.title('3-Second Segment Count of Audio Files')
plt.xlabel('Number of 3-Second Segments')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(r"C:\Warsaw University of Technology\Sem5\IMLProject\venv\segment_count_distribution.png")
plt.show()
print("3-second segment count histogram saved.")