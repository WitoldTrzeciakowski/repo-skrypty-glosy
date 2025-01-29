
import matplotlib.pyplot as plt
import pandas as pd

data_resnet = pd.read_pickle('metrics_data.pkl')

data_resnet = pd.DataFrame(data_resnet)

data_cifar = pd.read_pickle('cifar_final.pkl')

losses_cifar = []

losses_restnet = []

for index, row in data_resnet.iterrows():
    losses_restnet.append(row['loss'])

for line in data_cifar:
    loss = float((line.split('Validation Accuracy: ')[0]).split('Validation Loss: ')[1].replace(',', ''))
    losses_cifar.append(loss)

print()
data = {
    'Epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    '2 layer CNN Loss': losses_cifar,
    'Resnet Loss': losses_restnet
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['2 layer CNN Loss'], label='2 layer CNN Loss', marker='o')
plt.plot(df['Epoch'], df['Resnet Loss'], label='Resnet Loss', marker='o', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss (%)')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

plt.savefig('./check_png/loss_plot.png')


