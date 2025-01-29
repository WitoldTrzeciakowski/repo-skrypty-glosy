import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_pickle('metrics_data.pkl')

data = pd.DataFrame(data)

confusion_matrix = data['confusion_matrix'][9]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.savefig('./check_png/LAST_CM.png')


