from listener import *
from test import *
import glob

AUDIO_DIR = 'NewAudio'

files = glob.glob(f'./{AUDIO_DIR}/*')

# Remove each file
for file in files:
    try:
        os.remove(file)
        print("")
    except Exception as e:
        print(f"")

sr = 44100
channels = 1
dtype = 'float32'
is_recording = True

stop_thread = threading.Thread(target=stop_recording)
stop_thread.start()


audio = record_audio()


#audio = librosa.util.normalize(audio)

print(f"Audio recorded: {audio.shape}, Sampling rate: {sr}")

audio_path = save_audio_file(audio, sr)

print(f"Audio saved: {audio_path}")

if (process_file(audio_path,model_df,df_state)):
    print("returned TRUE")
else:
    print("returned FALSE")