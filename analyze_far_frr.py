import json
import os
import matplotlib.pyplot as plt
import numpy as np

# List of allowed speakers (Class 1)
ALLOWED_SPEAKERS = ["f1", "f7", "f8", "m3", "m6", "m8"]

def analyze_predictions(json_file):
    """
    Analyzes prediction file and calculates FAR and FRR metrics.
    
    FAR = number of incorrectly accepted Class 0 samples / total number of Class 0 samples
    FRR = number of incorrectly rejected Class 1 samples / total number of Class 1 samples
    """
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    # Counters for FAR and FRR calculations
    class0_total = 0  # Total number of Class 0 samples
    class1_total = 0  # Total number of Class 1 samples
    false_accepts = 0  # Number of incorrectly accepted Class 0 samples
    false_rejects = 0  # Number of incorrectly rejected Class 1 samples
    
    for pred in predictions:
        # Extract true speaker from filename
        true_speaker = pred["file"].split()[0].lower()
        predicted_speaker = pred["predicted"].lower()
        
        # Determine if it's Class 1 (allowed) or Class 0 (not allowed)
        is_class1 = true_speaker in ALLOWED_SPEAKERS
        predicted_class1 = predicted_speaker in ALLOWED_SPEAKERS
        
        if is_class1:
            class1_total += 1
            if not predicted_class1:
                false_rejects += 1
        else:
            class0_total += 1
            if predicted_class1:
                false_accepts += 1
    
    # Calculate FAR and FRR
    far = false_accepts / class0_total if class0_total > 0 else 0
    frr = false_rejects / class1_total if class1_total > 0 else 0
    
    return {
        'far': far,
        'frr': frr,
        'false_accepts': false_accepts,
        'false_rejects': false_rejects,
        'class0_total': class0_total,
        'class1_total': class1_total
    }

def plot_metrics(results):
    """Creates separate plots for FAR/FRR trends and their difference."""
    epochs = list(results.keys())
    fars = [results[e]['far'] for e in epochs]
    frrs = [results[e]['frr'] for e in epochs]

    # Plot 1: FAR and FRR trends
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fars, 'b-', marker='o', label='FAR', linewidth=2)
    plt.plot(epochs, frrs, 'r-', marker='o', label='FRR', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('FAR and FRR across Training Epochs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('far_frr_trends.png')
    plt.close()
    
    # Plot 2: Difference between FAR and FRR
    plt.figure(figsize=(10, 6))
    differences = np.array(fars) - np.array(frrs)
    plt.bar(epochs, differences, color='purple', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('FAR - FRR')
    plt.title('Difference between FAR and FRR')
    plt.grid(True)
    
    # Add values on the bar plot
    for i, v in enumerate(differences):
        plt.text(i + 1, v, f'{v:.3f}', 
                horizontalalignment='center',
                verticalalignment='bottom' if v > 0 else 'top')

    plt.tight_layout()
    plt.savefig('far_frr_diff.png')
    plt.close()

def analyze_all_epochs():
    """Analyzes all predictions_epoch_i.json files and creates plots."""
    results = {}
    
    for i in range(1, 11):  # for epochs 1-10
        filename = f'predictions_epoch_{i}.json'
        if os.path.exists(filename):
            results[i] = analyze_predictions(filename)
            print(f"\nEpoch {i}:")
            print(f"FAR: {results[i]['far']:.4f} ({results[i]['false_accepts']}/{results[i]['class0_total']})")
            print(f"FRR: {results[i]['frr']:.4f} ({results[i]['false_rejects']}/{results[i]['class1_total']})")
    
    if results:
        plot_metrics(results)
        print("\nPlots have been saved to 'far_frr_trends.png' and 'far_frr_diff.png'")
    
    return results

if __name__ == "__main__":
    results = analyze_all_epochs()