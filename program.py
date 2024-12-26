from torchvision.models import resnet18
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

# Load the model
model = resnet18()
model.fc = torch.nn.Linear(in_features=512, out_features=20)
state_dict = torch.load("trained_model3.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# Directory containing spectrograms
image_dir = "./NewSpec"

# Get all relevant files
files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('enhanced_spectrogram.png')]

if not files:
    raise FileNotFoundError("No spectrograms found in the directory.")

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# List to store all probabilities
all_probabilities = []

# Analyze each spectrogram
for file_path in files:
    print(f"Processing file: {file_path}")

    # Open and preprocess the image using a `with` statement
    with Image.open(file_path) as image:
        input_tensor = preprocess(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        all_probabilities.append(probabilities)

# Stack all probabilities and compute the mean
all_probabilities = torch.cat(all_probabilities, dim=0)  # Combine into one tensor
final_probabilities = all_probabilities.mean(dim=0)

# Define class categories
class_1 = ['f1', 'f7', 'f8', 'm3', 'm6', 'm8']
current_class = ""
class_1_ = 0

print("Probabilities for each class:")
for i, prob in enumerate(final_probabilities):
    if i < 10:
        current_class = f"f{i+1}"
        print(f"{current_class}: {prob:.4f}")
        if current_class in class_1:
            class_1_ += prob
    else:
        current_class = f"m{i-9}"
        print(f"{current_class}: {prob:.4f}")
        if current_class in class_1:
            class_1_ += prob

print(f"Probability that the audio belongs to class 1 is: {class_1_:.4f}")
