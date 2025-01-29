
import matplotlib.pyplot as plt
import pandas as pd

data_resnet = pd.read_pickle('metrics_data.pkl')

data_resnet = pd.DataFrame(data_resnet)

data_cifar = pd.read_pickle('cifar_final.pkl')

accuracies_cifar = []

accuracies_restnet = []

for index, row in data_resnet.iterrows():
    accuracies_restnet.append(row['overall_accuracy'])

for line in data_cifar:
    accuracy = float(line.split('Validation Accuracy: ')[1].replace('%\n', ''))
    accuracies_cifar.append(accuracy)

print()
data = {
    'Epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '2 layer CNN Accuracy': accuracies_cifar,
    'Resnet Accuracy': accuracies_restnet
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['2 layer CNN Accuracy'], label='2 layer CNN Accuracy', marker='o')
plt.plot(df['Epoch'], df['Resnet Accuracy'], label='Resnet Accuracy', marker='o', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)

plt.savefig('./check_png/accuracy_plot.png')


