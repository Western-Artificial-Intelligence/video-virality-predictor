"""
Audio Feature Extraction Script
--------------------------------
Extracts key audio features from downloaded .m4a files including:
- Duration
- Average loudness level
- Amount of silence
- Spectral centroid (dominant frequency measure)
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa


AUDIO_DIR = "raw_audio"
OUTPUT_CSV = "audio_features.csv"
ERROR_LOG_FILE = "audio_analysis_errors.log"
SILENCE_THRESHOLD = -40
MIN_SILENCE_LEN = 100
SAMPLE_RATE = 22050


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger("audio_analyzer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def extract_video_id(filename: str) -> str:
    """Extract video ID from filename."""
    return Path(filename).stem


def calculate_duration(audio: AudioSegment) -> float:
    """Calculate audio duration in seconds."""
    return len(audio) / 1000.0


def calculate_loudness(audio: AudioSegment) -> float:
    """Calculate average loudness in dBFS."""
    return audio.dBFS


def calculate_silence(audio: AudioSegment, threshold: int = SILENCE_THRESHOLD, 
                     min_silence_len: int = MIN_SILENCE_LEN) -> tuple:
    """Calculate total silence duration and percentage."""
    silence_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=threshold
    )
    
    total_silence_ms = sum(end - start for start, end in silence_ranges)
    total_duration_ms = len(audio)
    
    silence_seconds = total_silence_ms / 1000.0
    silence_percentage = (total_silence_ms / total_duration_ms * 100) if total_duration_ms > 0 else 0.0
    
    return silence_seconds, silence_percentage


def calculate_librosa_features(audio_path: str, sr: int = SAMPLE_RATE) -> Dict:
    """Calculate all librosa features: spectral centroid, RMS energy, ZCR, tempo."""
    features = {
        'spectral_centroid_mean': np.nan,
        'spectral_centroid_std': np.nan,
        'rms_energy_mean': np.nan,
        'rms_energy_std': np.nan,
        'zcr_mean': np.nan,
        'zcr_std': np.nan,
        'tempo_bpm': np.nan
    }
    
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy_mean'] = float(np.mean(rms))
        features['rms_energy_std'] = float(np.std(rms))
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # Tempo estimation
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo_bpm'] = float(tempo)
        
    except Exception as e:
        pass
    
    return features


def analyze_audio_file(audio_path: str, logger: logging.Logger, script_dir: Path = None) -> Dict:
    """Extract all features from a single audio file."""
    filename = os.path.basename(audio_path)
    video_id = extract_video_id(filename)
    
    # Store relative path for better portability
    if script_dir:
        try:
            file_path = os.path.relpath(audio_path, script_dir)
        except ValueError:
            file_path = audio_path
    else:
        file_path = audio_path
    
    features = {
        'video_id': video_id,
        'file_path': file_path,
        'duration_sec': np.nan,
        'loudness_dbfs': np.nan,
        'silence_total_sec': np.nan,
        'silence_percent': np.nan,
        'spectral_centroid_mean': np.nan,
        'spectral_centroid_std': np.nan,
        'rms_energy_mean': np.nan,
        'rms_energy_std': np.nan,
        'zcr_mean': np.nan,
        'zcr_std': np.nan,
        'tempo_bpm': np.nan,
        'status': 'error',
        'error': ''
    }
    
    try:
        audio = AudioSegment.from_file(audio_path, format="m4a")
        
        duration = calculate_duration(audio)
        loudness = calculate_loudness(audio)
        silence_seconds, silence_percentage = calculate_silence(audio)
        
        features['duration_sec'] = round(duration, 3)
        features['loudness_dbfs'] = round(loudness, 2)
        features['silence_total_sec'] = round(silence_seconds, 3)
        features['silence_percent'] = round(silence_percentage, 2)
        
        librosa_features = calculate_librosa_features(audio_path)
        features.update({
            'spectral_centroid_mean': round(librosa_features['spectral_centroid_mean'], 2) if not np.isnan(librosa_features['spectral_centroid_mean']) else np.nan,
            'spectral_centroid_std': round(librosa_features['spectral_centroid_std'], 2) if not np.isnan(librosa_features['spectral_centroid_std']) else np.nan,
            'rms_energy_mean': round(librosa_features['rms_energy_mean'], 4) if not np.isnan(librosa_features['rms_energy_mean']) else np.nan,
            'rms_energy_std': round(librosa_features['rms_energy_std'], 4) if not np.isnan(librosa_features['rms_energy_std']) else np.nan,
            'zcr_mean': round(librosa_features['zcr_mean'], 4) if not np.isnan(librosa_features['zcr_mean']) else np.nan,
            'zcr_std': round(librosa_features['zcr_std'], 4) if not np.isnan(librosa_features['zcr_std']) else np.nan,
            'tempo_bpm': round(librosa_features['tempo_bpm'], 2) if not np.isnan(librosa_features['tempo_bpm']) else np.nan,
            'status': 'success',
            'error': ''
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error analyzing {audio_path}: {error_msg}")
        logger.error(f"[ERROR] {audio_path} | {error_msg}", exc_info=False)
        features['status'] = 'error'
        features['error'] = error_msg
    
    return features


def analyze_all_audio_files(audio_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """Analyze all audio files in the directory."""
    script_dir = Path(__file__).parent.absolute()
    audio_full_path = (script_dir / audio_dir).resolve()
    
    if not audio_full_path.exists():
        logger.error(f"Audio directory not found: {audio_full_path}")
        sys.exit(1)
    
    audio_files = list(audio_full_path.glob("*.m4a"))
    total_files = len(audio_files)
    
    if total_files == 0:
        logger.error(f"No .m4a files found in {audio_full_path}")
        sys.exit(1)
    
    logger.info(f"Found {total_files} audio files to analyze")
    logger.info(f"Audio directory: {audio_full_path}\n")
    
    features_list = []
    
    for idx, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{idx}/{total_files}] Analyzing: {audio_file.name}")
        
        features = analyze_audio_file(str(audio_file), logger, script_dir)
        features_list.append(features)
        
        if features['status'] == 'success':
            logger.info(f"    Duration: {features['duration_sec']}s, Loudness: {features['loudness_dbfs']} dBFS, "
                       f"Silence: {features['silence_percent']}%, Centroid: {features['spectral_centroid_mean']} Hz")
        else:
            logger.error(f"    Failed: {features['error']}")
    
    df = pd.DataFrame(features_list)
    success_count = len(df[df['status'] == 'success'])
    failed_count = len(df[df['status'] == 'error'])
    logger.info(f"\nSuccessfully analyzed {success_count} files ({failed_count} failed)")
    
    return df


def save_results(df: pd.DataFrame, output_csv: str, logger: logging.Logger):
    """Save results to CSV file with correct column order."""
    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir / output_csv
    
    # Ensure column order matches requirements
    column_order = [
        'video_id',
        'file_path',
        'duration_sec',
        'loudness_dbfs',
        'silence_total_sec',
        'silence_percent',
        'spectral_centroid_mean',
        'spectral_centroid_std',
        'rms_energy_mean',
        'rms_energy_std',
        'zcr_mean',
        'zcr_std',
        'tempo_bpm',
        'status',
        'error'
    ]
    
    df = df[column_order]
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Total features extracted: {len(df)} files")


def print_summary(df: pd.DataFrame, logger: logging.Logger):
    """Print summary statistics."""
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files processed: {len(df)}")
    logger.info(f"Success: {len(df[df['status'] == 'success'])}")
    logger.info(f"Failed: {len(df[df['status'] == 'error'])}")
    
    success_df = df[df['status'] == 'success']
    
    if len(success_df) > 0:
        logger.info(f"\nDuration Statistics (seconds):")
        logger.info(f"  Mean: {success_df['duration_sec'].mean():.2f}")
        logger.info(f"  Min: {success_df['duration_sec'].min():.2f}")
        logger.info(f"  Max: {success_df['duration_sec'].max():.2f}")
        
        logger.info(f"\nLoudness Statistics (dBFS):")
        logger.info(f"  Mean: {success_df['loudness_dbfs'].mean():.2f}")
        logger.info(f"  Min: {success_df['loudness_dbfs'].min():.2f}")
        logger.info(f"  Max: {success_df['loudness_dbfs'].max():.2f}")
        
        logger.info(f"\nSilence Statistics:")
        logger.info(f"  Mean percentage: {success_df['silence_percent'].mean():.2f}%")
        logger.info(f"  Mean total seconds: {success_df['silence_total_sec'].mean():.2f}s")
        
        logger.info(f"\nSpectral Centroid Statistics (Hz):")
        valid_centroids = success_df['spectral_centroid_mean'].dropna()
        if len(valid_centroids) > 0:
            logger.info(f"  Mean: {valid_centroids.mean():.2f}")
            logger.info(f"  Std Mean: {success_df['spectral_centroid_std'].mean():.2f}")
        
        logger.info(f"\nRMS Energy Statistics:")
        valid_rms = success_df['rms_energy_mean'].dropna()
        if len(valid_rms) > 0:
            logger.info(f"  Mean: {valid_rms.mean():.4f}")
            logger.info(f"  Std Mean: {success_df['rms_energy_std'].mean():.4f}")
        
        logger.info(f"\nZero-Crossing Rate Statistics:")
        valid_zcr = success_df['zcr_mean'].dropna()
        if len(valid_zcr) > 0:
            logger.info(f"  Mean: {valid_zcr.mean():.4f}")
            logger.info(f"  Std Mean: {success_df['zcr_std'].mean():.4f}")
        
        logger.info(f"\nTempo Statistics (BPM):")
        valid_tempo = success_df['tempo_bpm'].dropna()
        if len(valid_tempo) > 0:
            logger.info(f"  Mean: {valid_tempo.mean():.2f}")
            logger.info(f"  Min: {valid_tempo.min():.2f}")
            logger.info(f"  Max: {valid_tempo.max():.2f}")
    
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract audio features from downloaded .m4a files"
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default=AUDIO_DIR,
        help=f'Audio directory (default: {AUDIO_DIR})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_CSV,
        help=f'Output CSV file (default: {OUTPUT_CSV})'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    log_file = script_dir / ERROR_LOG_FILE
    logger = setup_logging(str(log_file))
    
    logger.info("Audio Feature Extraction")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    df = analyze_all_audio_files(args.audio_dir, logger)
    save_results(df, args.output, logger)
    print_summary(df, logger)
    
    logger.info("\nAnalysis complete.")


if __name__ == "__main__":
    main()

