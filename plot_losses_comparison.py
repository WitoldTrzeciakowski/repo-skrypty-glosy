import matplotlib.pyplot as plt
import pandas as pd

data_resnet = pd.read_pickle('metrics_data.pkl')
data_squeeze = pd.read_pickle('metrics_data_squeeze.pkl')
data_mobilenet = pd.read_pickle('metrics_data_mobilenet.pkl')
data_cifar = pd.read_pickle('cifar_final.pkl')


data_resnet = pd.DataFrame(data_resnet)
data_squeeze = pd.DataFrame(data_squeeze)
data_mobilenet = pd.DataFrame(data_mobilenet)

losses_resnet = list(data_resnet['loss'])
losses_squeeze = list(data_squeeze['loss'])
losses_mobilenet = list(data_mobilenet['loss'])
losses_cifar = []

for line in data_cifar:
    loss = float((line.split('Validation Accuracy: ')[0]).split('Validation Loss: ')[1].replace(',', ''))
    losses_cifar.append(loss)

epochs = list(range(1, len(losses_resnet) + 1))

df = pd.DataFrame({
    'Epoch': epochs,
    '2 layer CNN Loss': losses_cifar,
    'ResNet Loss': losses_resnet,
    'SqueezeNet Loss': losses_squeeze,
    'MobileNet Loss': losses_mobilenet
})

plt.figure(figsize=(12, 8))
plt.plot(df['Epoch'], df['2 layer CNN Loss'], label='2 layer CNN Loss', marker='o', linestyle='-')
plt.plot(df['Epoch'], df['ResNet Loss'], label='ResNet Loss', marker='s', linestyle='--')
plt.plot(df['Epoch'], df['SqueezeNet Loss'], label='SqueezeNet Loss', marker='^', linestyle='-.')
plt.plot(df['Epoch'], df['MobileNet Loss'], label='MobileNet Loss', marker='d', linestyle=':')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss vs Epoch for All Models')
plt.legend()
plt.grid(True)

plt.savefig('./loss_plot_all_models.png')

plt.show()
