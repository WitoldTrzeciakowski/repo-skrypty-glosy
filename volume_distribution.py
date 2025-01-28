import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

def analyze_single_file(file_path):
    """
    Analyze volume distribution in a single audio file.
    Returns normalized RMS values in dB scale.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        
        # Calculate RMS energy in 30ms frames with 50% overlap
        frame_length = int(sr * 0.03)
        hop_length = int(frame_length / 2)
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)[0]
        
        # Interpolate to fixed length for averaging
        target_length = 1000
        x = np.linspace(0, len(rms_db), len(rms_db))
        x_new = np.linspace(0, len(rms_db), target_length)
        rms_db_interp = np.interp(x_new, x, rms_db)
        
        return rms_db_interp
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def find_wav_files(data_dir):
    """Find all WAV files in the specified directories."""
    target_dirs = [
        'clean', 'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 
        'ipad_confroom1', 'ipad_confroom2', 'ipadflat_confroom1', 
        'ipadflat_office1', 'ipad_livingroom1', 'ipad_office1', 
        'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 
        'iphone_livingroom1', 'produced'
    ]
    
    wav_files = []
    for root, _, files in os.walk(data_dir):
        if any(target_dir in root for target_dir in target_dirs):
            for file in files:
                if file.endswith('.wav') and not file.startswith('._'):
                    wav_files.append(os.path.join(root, file))
    return wav_files

def plot_average_distribution(volume_data, output_dir, input_dir_name):
    """
    Plot average volume distribution with standard deviation bands.
    
    Args:
        volume_data: List of volume distributions for each file
        output_dir: Directory to save the plot
        input_dir_name: Name of input directory for plot title
    """
    volume_array = np.vstack(volume_data)
    mean_volume = np.mean(volume_array, axis=0)
    std_volume = np.std(volume_array, axis=0)
    
    # Create normalized time axis (0-100%)
    time_normalized = np.linspace(0, 100, len(mean_volume))
    
    plt.figure(figsize=(15, 10))
    
    # Plot mean and standard deviation
    plt.plot(time_normalized, mean_volume, 'b-', label='Mean Volume', linewidth=2)
    plt.fill_between(time_normalized, 
                     mean_volume - std_volume,
                     mean_volume + std_volume,
                     color='b', alpha=0.2,
                     label='Standard Deviation')
    
    plt.xlabel('Normalized Recording Time (%)')
    plt.ylabel('Volume (dB)')
    plt.title(f'Average Volume Distribution\nDirectory: {input_dir_name}')
    plt.grid(True)
    plt.legend()
    
    # Add statistics box
    stats_text = (
        f'Mean Volume: {np.mean(mean_volume):.1f} dB\n'
        f'Median Volume: {np.median(mean_volume):.1f} dB\n'
        f'Avg Std Dev: {np.mean(std_volume):.1f} dB\n'
        f'Peak: {np.max(mean_volume):.1f} dB\n'
        f'Min: {np.min(mean_volume):.1f} dB'
    )
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'average_volume_distribution_{input_dir_name}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze volume distribution in WAV files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory containing WAV files')
    parser.add_argument('--output_dir', type=str, default='volume_analysis',
                      help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    input_dir_name = os.path.basename(os.path.normpath(args.input_dir))
    
    # Find and analyze WAV files
    wav_files = find_wav_files(args.input_dir)
    print(f"Found {len(wav_files)} WAV files in {args.input_dir}")
    
    volume_data = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_single_file, wav_files), 
                          total=len(wav_files),
                          desc="Processing files"))
        volume_data = [r for r in results if r is not None]
    
    if volume_data:
        output_path = plot_average_distribution(volume_data, args.output_dir, input_dir_name)
        print(f"Analysis saved to: {output_path}")
    else:
        print("No files were successfully analyzed")

if __name__ == "__main__":
    main()