import matplotlib.pyplot as plt
import pandas as pd

data_resnet = pd.read_pickle('metrics_data.pkl')
data_squeeze = pd.read_pickle('metrics_data_squeeze.pkl')
data_mobilenet = pd.read_pickle('metrics_data_mobilenet.pkl')
data_cifar = pd.read_pickle('cifar_final.pkl')

data_resnet = pd.DataFrame(data_resnet)
data_squeeze = pd.DataFrame(data_squeeze)
data_mobilenet = pd.DataFrame(data_mobilenet)

accuracies_resnet = list(data_resnet['overall_accuracy'])
accuracies_squeeze = list(data_squeeze['overall_accuracy'])
accuracies_mobilenet = list(data_mobilenet['overall_accuracy'])
accuracies_cifar = []

for line in data_cifar:
    accuracy = float(line.split('Validation Accuracy: ')[1].replace('%\n', ''))
    accuracies_cifar.append(accuracy)

epochs = list(range(1, len(accuracies_resnet) + 1))


df = pd.DataFrame({
    'Epoch': epochs,
    '2 layer CNN Accuracy': accuracies_cifar,
    'ResNet Accuracy': accuracies_resnet,
    'SqueezeNet Accuracy': accuracies_squeeze,
    'MobileNet Accuracy': accuracies_mobilenet
})


plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['2 layer CNN Accuracy'], label='2 layer CNN', marker='o', linestyle='-')
plt.plot(df['Epoch'], df['ResNet Accuracy'], label='ResNet', marker='s', linestyle='--')
plt.plot(df['Epoch'], df['SqueezeNet Accuracy'], label='SqueezeNet', marker='^', linestyle='-.')
plt.plot(df['Epoch'], df['MobileNet Accuracy'], label='MobileNet', marker='d', linestyle=':')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy vs Epoch')
plt.legend()
plt.grid(True)

plt.savefig('./accuracy_plot.png')

plt.show()
