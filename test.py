import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from PIL import Image
from IPython.display import Audio, display
from torchvision import transforms
from resample_audio_and_clear_of_noise import re_sample_audio, is_valid_wav_file
from torchvision.models import resnet18
from silence_removal import process_audio_file
from resample_audio_and_clear_of_noise import re_sample_audio
from create_spectogram import process_audio_file as specotgram_process
from df.enhance import enhance, init_df, load_audio, save_audio

LOCATORS_SPEAKERS_LIST = ["f1", "f7", "f8", "m3", "m6", "m8"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'trained_model9.pth'

# Initialize the noise reduction model
model_df, df_state, _ = init_df()

# Load the ResNet18 model
def initialize_model():
    model = resnet18()  # Load ResNet18
    model.fc = nn.Linear(model.fc.in_features, len(LOCATORS_SPEAKERS_LIST) + 1)  # Adjust output classes
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

classification_model = initialize_model()

# Transform for spectrogram processing
spec_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet normalization
])

def delete_noise_for_file(audio_path, model, df_state):
    """Process and remove noise from a single audio file."""
    try:
        if not is_valid_wav_file(audio_path):
            print(f"Skipping invalid WAV file: {audio_path}")
            return
        
        # Load and enhance the audio
        audio, _ = load_audio(audio_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        
        # Save the enhanced audio
        enhanced_audio_path = audio_path.replace('.wav', '_enhanced.wav') 
        save_audio(enhanced_audio_path, enhanced, df_state.sr())
        
        # Delete the original audio file
       # os.remove(audio_path)
        print(f"Processed and deleted: {audio_path}")
        return enhanced_audio_path
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

def generate_spectrogram(segment):
    """Generate a normalized spectrogram tensor from an audio segment."""
    spec = librosa.stft(segment)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    spec_db = np.clip((spec_db - spec_db.min()) / (spec_db.max() - spec_db.min()), 0, 1)  # Normalize to [0, 1]
    spec_image = (spec_db * 255).astype(np.uint8)
    # Convert to 3 channels (RGB) by duplicating the single channel
    spec_image_rgb = np.stack([spec_image] * 3, axis=-1)
    return spec_image_rgb

def classify_segment_from_path(spectrogram_path, model):
    """Classify a spectrogram image from a file path using the ResNet model."""
    try:
        # Load the spectrogram image from the file path
        spec_image = Image.open(spectrogram_path).convert('RGB')  # Ensure the image is in RGB format
        
        # Convert the spectrogram to a tensor and apply the transformation
        spec_tensor = spec_transform(spec_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Perform inference on the spectrogram
            output = model(spec_tensor)
            
            # Apply softmax to get class probabilities
            probs = torch.nn.functional.softmax(output, dim=1)
            
            # Get the predicted class index and the associated confidence
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item() * 100  # Confidence in percentage
        
        return pred_idx, confidence
    
    except Exception as e:
        print(f"Error during classification: {e}")
        return None, None

def process_file(file_path,model_df,df_state):
    """Process an audio file: enhance, split, and classify."""
    resampled_path = re_sample_audio(file_path)
    enhanced_path = delete_noise_for_file(resampled_path, model_df, df_state)
    audio_paths = process_audio_file(enhanced_path)
    bad = 0
    good = 0
    for audio_path in audio_paths:
        specotogram_path = specotgram_process(audio_path,"temp")
        predicted_class, confidence = classify_segment_from_path(specotogram_path,classification_model )
        print(f"class {predicted_class}, confidence: {confidence}")
        if predicted_class < 6:
            good+= 1
        else:
            bad+= 1
    if good > bad:
        print("Welcome in :)")
    else:
        print("Bye Bye robber ")





# Example usage
process_file(r"G:\glosy_model\stash\daps\ipadflat_office1\f1_script1_ipadflat_office1_enhanced.wav",model_df,df_state)
