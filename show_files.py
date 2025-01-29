import pickle

file1 = 'cifar_final.pkl'
file2 = 'metrics_data.pkl'
file3 = 'metrics_data_mobilenet.pkl'

def load_and_print_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(f"Contents of {file_path}:")
        print(data)
        print("\n" + "="*50 + "\n")

load_and_print_pickle(file2)
#load_and_print_pickle(file2)
