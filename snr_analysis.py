import os
import numpy as np
import soundfile as sf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
from scipy import stats

def calculate_snr(signal):
   """Calculate Signal-to-Noise Ratio in dB with improved signal/noise separation."""
   frame_length = 1024
   frames = np.array_split(signal, len(signal)//frame_length)
   frame_energies = [np.sum(frame**2) for frame in frames]
   
   threshold = np.percentile(frame_energies, 70)
   signal_frames = np.concatenate([frame for frame, energy in zip(frames, frame_energies) if energy > threshold])
   noise_frames = np.concatenate([frame for frame, energy in zip(frames, frame_energies) if energy <= threshold])
   
   signal_power = np.mean(signal_frames**2)
   noise_power = np.mean(noise_frames**2)
   
   return 10 * np.log10(signal_power/noise_power) if noise_power > 0 else 0

def process_audio_file(file_path):
   """Process a single audio file to extract SNR and metadata."""
   try:
       y, sr = sf.read(file_path)
       if len(y.shape) > 1:
           y = y.mean(axis=1)
       
       y = y / np.max(np.abs(y))
       snr = calculate_snr(y)
       
       filename = os.path.basename(file_path)
       parts = filename.split('_')
       path_parts = str(file_path).split(os.sep)
       environment = path_parts[-2] if len(path_parts) > 1 else 'unknown'
       
       device = 'unknown'
       if 'ipad' in environment:
           device = 'ipad'
       elif 'iphone' in environment:
           device = 'iphone'
       elif any(x in environment for x in ['clean', 'produced']):
           device = 'studio'
           
       return {
           'file': filename,
           'environment': environment,
           'device': device,
           'speaker': parts[0] if parts else 'unknown',
           'snr': snr,
           'full_path': str(file_path)
       }
   except Exception as e:
       print(f"Error processing {file_path}: {str(e)}")
       return None

def analyze_speaker_groups(df):
   """Analyze SNR differences between allowed and not allowed speakers."""
   allowed_speakers = ["f1", "f7", "f8", "m3", "m6", "m8"]
   allowed_data = df[df['speaker'].isin(allowed_speakers)]['snr']
   not_allowed_data = df[~df['speaker'].isin(allowed_speakers)]['snr']
   
   t_stat, p_value = stats.ttest_ind(allowed_data, not_allowed_data)
   
   return {
       'allowed_mean_snr': allowed_data.mean(),
       'not_allowed_mean_snr': not_allowed_data.mean(),
       'p_value': p_value,
       't_statistic': t_stat
   }

def plot_snr_analysis(df, output_dir):
   """Create separate visualizations for SNR analysis."""
   Path(output_dir).mkdir(parents=True, exist_ok=True)
   plt.rcParams['font.size'] = 10
   
   # Speaker analysis
   plt.figure(figsize=(15, 5))
   sns.boxplot(data=df, x='speaker', y='snr')
   plt.title('SNR Distribution by Speaker')
   plt.xticks(rotation=45)
   plt.ylabel('SNR (dB)')
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'snr_by_speaker.png'), dpi=300)
   plt.close()
   
   # Environment analysis
   plt.figure(figsize=(15, 5))
   sns.boxplot(data=df, x='environment', y='snr')
   plt.title('SNR Distribution by Environment')
   plt.xticks(rotation=45)
   plt.ylabel('SNR (dB)')
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'snr_by_environment.png'), dpi=300)
   plt.close()
   
   # Device analysis
   plt.figure(figsize=(15, 5))
   sns.violinplot(data=df, x='device', y='snr')
   plt.title('SNR Distribution by Device')
   plt.ylabel('SNR (dB)')
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'snr_by_device.png'), dpi=300)
   plt.close()
   
   # Environment-Device heatmap
   plt.figure(figsize=(15, 5))
   pivot_table = df.pivot_table(
       values='snr', 
       index='device',
       columns='environment',
       aggfunc='mean'
   )
   sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
   plt.title('Mean SNR by Device and Environment (dB)')
   plt.tight_layout()
   plt.savefig(os.path.join(output_dir, 'snr_heatmap.png'), dpi=300)
   plt.close()
   
   return df.groupby(['environment', 'device'])['snr'].agg([
       'count', 'mean', 'std', 'min', 'max'
   ]).round(2)

def find_wav_files(data_dir):
   """Find WAV files in all relevant directories."""
   target_dirs = [
       'clean', 'cleanraw', 'ipad_balcony1', 'ipad_bedroom1', 
       'ipad_confroom1', 'ipad_confroom2', 'ipadflat_confroom1', 
       'ipadflat_office1', 'ipad_livingroom1', 'ipad_office1', 
       'ipad_office2', 'iphone_balcony1', 'iphone_bedroom1', 
       'iphone_livingroom1', 'produced'
   ]
   all_files = []
   data_dir_path = Path(data_dir)
   
   print(f"\nSearching in: {data_dir_path.absolute()}")
   
   for root, dirs, files in os.walk(data_dir):
       if any(target_dir in root for target_dir in target_dirs):
           for file in files:
               if file.endswith('.wav') and not file.startswith('._'):
                   full_path = Path(root) / file
                   all_files.append(full_path)
   
   print(f"\nTotal WAV files found: {len(all_files)}")
   return all_files

def analyze_snr(data_dir):
   """Analyze SNR across all audio files in dataset."""
   audio_files = find_wav_files(data_dir)
   print(f"Found {len(audio_files)} WAV files")
   
   with ThreadPoolExecutor() as executor:
       results = list(filter(None, executor.map(process_audio_file, audio_files)))
   
   return pd.DataFrame(results)

def main():
   parser = argparse.ArgumentParser(description='Analyze SNR in WAV files')
   parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing WAV files')
   parser.add_argument('--output_dir', type=str, default='snr_analysis', help='Output directory for results')
   args = parser.parse_args()

   print(f"Analyzing files in: {args.input_dir}")
   
   df = analyze_snr(args.input_dir)
   
   if df.empty:
       print("No valid audio files were processed!")
       return
       
   summary_stats = plot_snr_analysis(df, args.output_dir)
   speaker_stats = analyze_speaker_groups(df)
   
   print("\nSpeaker Group Analysis:")
   print(f"Allowed Speakers Mean SNR: {speaker_stats['allowed_mean_snr']:.2f} dB")
   print(f"Not Allowed Speakers Mean SNR: {speaker_stats['not_allowed_mean_snr']:.2f} dB")
   print(f"T-statistic: {speaker_stats['t_statistic']:.2f}")
   print(f"P-value: {speaker_stats['p_value']:.4f}")
   
   print("\nEnvironment-wise SNR Summary:")
   print(df.groupby('environment')['snr'].mean().sort_values().round(2))
   
   print("\nDevice-wise SNR Summary:")
   print(df.groupby('device')['snr'].mean().sort_values().round(2))
   
   print(f"\nDetailed results saved to {args.output_dir}/")

if __name__ == "__main__":
   main()