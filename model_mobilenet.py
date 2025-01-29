import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
PATH_TO_VOICES = r'G:\glosy_model\clean_data'
def evaluate_model_per_epoch(model, test_loader, selected_classes, device, epoch):
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Non-normalized outputs (logits)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_logits.extend(outputs.cpu().numpy().tolist())  # Convert logits to a list
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=selected_classes, output_dict=True)
    recall_per_class = [report[cls]['recall'] for cls in selected_classes]
    
    top_5_best_recall = sorted(zip(selected_classes, recall_per_class), key=lambda x: x[1], reverse=True)[:5]
    top_5_worst_recall = sorted(zip(selected_classes, recall_per_class), key=lambda x: x[1])[:5]
    overall_accuracy = 100 * correct / total

    # Save predictions to JSON
    predictions = []
    num =0
    for actual, predicted, logits in zip(all_labels, all_preds, all_logits):
        num+=1
        predictions.append({
            "file": selected_classes[actual] + " " + str(num),
            "predicted": selected_classes[predicted],
            "logits": logits  # Non-normalized predictions
        })

    predictions_file = f'predictions_epoch_{epoch}.json'

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "top_5_best_recall": top_5_best_recall,
        "top_5_worst_recall": top_5_worst_recall,
        "overall_accuracy": overall_accuracy,
    }
           
def train_and_evaluate(gamma=0.1, step_size=7, weight_decay=5e-4, momentum=0.9, lr=0.001, num_epochs=10, batch_size=16, num_classes=3):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of the model
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize for single-channel
    ])

    train_path = os.path.join(PATH_TO_VOICES, 'train')
    test_path = os.path.join(PATH_TO_VOICES, 'eval')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Train directory not found: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Test directory not found: {test_path}')

    train_dataset = ImageFolder(root=train_path, transform=transform)
    test_dataset = ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    selected_classes = next(os.walk(train_path))[1]
    num_classes = len(selected_classes)

    # Load ConvNeXt model
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    metrics_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        epoch_metrics = evaluate_model_per_epoch(model, test_loader, selected_classes, device, epoch + 1)
        epoch_metrics["epoch"] = epoch + 1
        epoch_metrics["loss"] = running_loss / len(train_loader)

        metrics_per_epoch.append(epoch_metrics)
        #torch.save(model.state_dict(), f'trained_model_epoch_{epoch + 1}.pth')
        print(f"Model saved to 'trained_model_epoch_{epoch + 1}.pth'")

    return metrics_per_epoch
def main():
    metrics = train_and_evaluate(num_epochs=10)

    for epoch_data in metrics:
        print(f"\nEpoch {epoch_data['epoch']} Metrics:")
        print(f"Loss: {epoch_data['loss']:.4f}")
        print(f"Overall Accuracy: {epoch_data['overall_accuracy']:.2f}%")
        print(f"Confusion Matrix:\n{epoch_data['confusion_matrix']}")
        print(f"Classification Report: {epoch_data['classification_report']}")
        print("Top 5 classes with the best recall:")
        for cls, recall in epoch_data['top_5_best_recall']:
            print(f"Class {cls}: {recall:.2f}")

        print("\nTop 5 classes with the worst recall:")
        for cls, recall in epoch_data['top_5_worst_recall']:
            print(f"Class {cls}: {recall:.2f}")

    with open('metrics_data.pkl', 'wb') as handle:
        pickle.dump(metrics, handle)

if __name__ == '__main__':
    main()
