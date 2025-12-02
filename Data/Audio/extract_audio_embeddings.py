"""
Audio Embedding Extraction using Wav2Vec2
-----------------------------------------
Extracts 768-dimensional audio embeddings from .m4a files using Wav2Vec2 model.
Wav2Vec2 captures high-level audio semantics, similar to how VideoMAE captures video motion and transitions.
"""

import os
import sys
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model


AUDIO_DIR = "raw_audio"
OUTPUT_CSV = "audio_embeddings.csv"
LINKS_CSV = "../Links/shorts_data/shorts_links_wide.csv"
EMBEDDINGS_DIR = "embeddings"
ERROR_LOG_FILE = "audio_embedding_errors.log"
EMBEDDING_SIZE = 768
SAMPLE_RATE = 16000
MODEL_NAME = "facebook/wav2vec2-base-960h"


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger("audio_embedder")
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


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_labels(links_csv_path: Path, logger: logging.Logger) -> Dict[str, Dict]:
    """Load labels (query, category_type, url) from links CSV."""
    if not links_csv_path.exists():
        logger.warning(f"Links CSV not found: {links_csv_path}")
        return {}
    
    try:
        links_df = pd.read_csv(links_csv_path)
        logger.info(f"Loaded {len(links_df)} links from {links_csv_path}")
        
        # Extract video IDs from URLs
        links_df['video_id'] = links_df['url'].apply(extract_video_id_from_url)
        
        # Remove rows where video ID couldn't be extracted
        valid_links = links_df[links_df['video_id'].notna()]
        
        # Create mapping dictionary
        label_map = {}
        for _, row in valid_links.iterrows():
            label_map[row['video_id']] = {
                'query': row['query'],
                'category_type': row['category_type'],
                'url': row['url']
            }
        
        logger.info(f"Created label mapping for {len(label_map)} videos")
        return label_map
        
    except Exception as e:
        logger.warning(f"Error loading labels: {str(e)}")
        return {}


# Global model and processor (loaded once)
_model = None
_processor = None

def load_model():
    """Load Wav2Vec2 model and processor."""
    global _model, _processor
    if _model is None:
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor

def extract_embedding(audio_path: str, logger: logging.Logger) -> Optional[np.ndarray]:
    """Extract Wav2Vec2 embedding from audio file."""
    try:
        # Load model
        model, processor = load_model()
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Process audio
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(0)
        
        # Average pooling across time dimension to get single embedding vector
        embedding = embeddings.mean(dim=0).numpy()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding from {audio_path}: {str(e)}")
        return None


def embedding_exists(video_id: str, embeddings_dir: Path) -> bool:
    """Check if embedding file already exists."""
    npy_path = embeddings_dir / f"{video_id}.npy"
    return npy_path.exists() and npy_path.stat().st_size > 0


def save_embedding_npy(embedding: np.ndarray, video_id: str, embeddings_dir: Path):
    """Save embedding as .npy file."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    npy_path = embeddings_dir / f"{video_id}.npy"
    np.save(npy_path, embedding)
    return npy_path


def process_audio_file_for_embedding(audio_path: str, embeddings_dir: Path, logger: logging.Logger, labels: Dict[str, Dict] = None) -> Dict:
    """Extract embedding from a single audio file."""
    filename = os.path.basename(audio_path)
    video_id = extract_video_id(filename)
    
    # Get labels for this video
    video_labels = labels.get(video_id, {}) if labels else {}
    
    result = {
        'video_id': video_id,
        'url': video_labels.get('url', ''),
        'query': video_labels.get('query', ''),
        'category_type': video_labels.get('category_type', ''),
        'file_path': audio_path,
        'embedding_file': '',
        'embedding_shape': '',
        'status': 'error',
        'error': ''
    }
    
    # Check if embedding already exists
    npy_path = embeddings_dir / f"{video_id}.npy"
    if embedding_exists(video_id, embeddings_dir):
        logger.info(f"Skipped (exists): {video_id}")
        result['embedding_file'] = str(npy_path)
        result['embedding_shape'] = f"({EMBEDDING_SIZE},)"
        result['status'] = 'success'
        result['error'] = ''
        return result
    
    try:
        logger.info(f"Extracting embedding: {video_id}")
        embedding = extract_embedding(audio_path, logger)
        
        if embedding is None:
            result['error'] = 'Failed to extract embedding'
            return result
        
        if embedding.shape[0] != EMBEDDING_SIZE:
            result['error'] = f'Unexpected embedding size: {embedding.shape[0]} (expected {EMBEDDING_SIZE})'
            return result
        
        # Save embedding as .npy file
        npy_path = save_embedding_npy(embedding, video_id, embeddings_dir)
        
        result['embedding_file'] = str(npy_path)
        result['embedding_shape'] = f"({EMBEDDING_SIZE},)"
        result['status'] = 'success'
        result['error'] = ''
        
        logger.info(f"  Success: {video_id} - Shape: {embedding.shape}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing {audio_path}: {error_msg}")
        logger.error(f"[ERROR] {audio_path} | {error_msg}", exc_info=False)
        result['error'] = error_msg
    
    return result


def extract_all_embeddings(audio_dir: str, embeddings_dir: str, links_csv: str, output_csv: str, logger: logging.Logger) -> pd.DataFrame:
    """Extract embeddings from all audio files, skipping already processed ones."""
    script_dir = Path(__file__).parent.absolute()
    audio_full_path = (script_dir / audio_dir).resolve()
    embeddings_full_path = (script_dir / embeddings_dir).resolve()
    links_csv_path = (script_dir / links_csv).resolve()
    csv_path = script_dir / output_csv
    
    if not audio_full_path.exists():
        logger.error(f"Audio directory not found: {audio_full_path}")
        sys.exit(1)
    
    # Load labels
    labels = load_labels(links_csv_path, logger)
    
    # Load existing CSV if it exists
    existing_df = None
    processed_video_ids = set()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            processed_video_ids = set(existing_df[existing_df['status'] == 'success']['video_id'].dropna().unique())
            logger.info(f"Found existing CSV with {len(processed_video_ids)} processed embeddings")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    
    audio_files = list(audio_full_path.glob("*.m4a"))
    total_files = len(audio_files)
    
    if total_files == 0:
        logger.error(f"No .m4a files found in {audio_full_path}")
        sys.exit(1)
    
    logger.info(f"Found {total_files} audio files to process")
    logger.info(f"Audio directory: {audio_full_path}")
    logger.info(f"Embeddings directory: {embeddings_full_path}")
    logger.info(f"Embedding size: {EMBEDDING_SIZE} dimensions\n")
    
    results_list = []
    skipped_count = 0
    
    for idx, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        video_id = extract_video_id(filename)
        
        # Skip if embedding already exists and is in CSV
        if video_id in processed_video_ids and embedding_exists(video_id, embeddings_full_path):
            logger.info(f"[{idx}/{total_files}] Skipped (already in CSV): {video_id}")
            skipped_count += 1
            continue
        
        logger.info(f"[{idx}/{total_files}] Processing: {audio_file.name}")
        
        result = process_audio_file_for_embedding(str(audio_file), embeddings_full_path, logger, labels)
        
        # Convert relative path for CSV
        if result['embedding_file']:
            rel_path = os.path.relpath(result['embedding_file'], script_dir)
            result['embedding_file'] = rel_path
        
        # Convert file_path to relative
        if result['file_path']:
            try:
                result['file_path'] = os.path.relpath(result['file_path'], script_dir)
            except ValueError:
                pass
        
        results_list.append(result)
    
    # Create DataFrame from new results
    new_df = pd.DataFrame(results_list)
    
    # Merge with existing results if any
    if existing_df is not None and len(new_df) > 0:
        df = pd.concat([existing_df, new_df], ignore_index=True)
        logger.info(f"\nMerged {len(new_df)} new embeddings with {len(existing_df)} existing records")
    elif existing_df is not None:
        df = existing_df
        logger.info(f"\nNo new files to process, using existing CSV")
    else:
        df = new_df
    
    success_count = len(df[df['status'] == 'success'])
    failed_count = len(df[df['status'] == 'error'])
    labeled_count = len(df[df['query'].notna() & (df['query'] != '')])
    
    logger.info(f"\nSuccessfully extracted embeddings: {success_count} files ({failed_count} failed)")
    logger.info(f"Labeled embeddings: {labeled_count} files")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} files (already in CSV)")
    
    return df


def save_results(df: pd.DataFrame, output_csv: str, logger: logging.Logger):
    """Save results to CSV file."""
    script_dir = Path(__file__).parent.absolute()
    output_path = script_dir / output_csv
    
    # Ensure column order (labels first)
    column_order = [
        'video_id',
        'url',
        'query',
        'category_type',
        'file_path',
        'embedding_file',
        'embedding_shape',
        'status',
        'error'
    ]
    
    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    
    df = df[column_order]
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Total embeddings extracted: {len(df[df['status'] == 'success'])} files")


def print_summary(df: pd.DataFrame, logger: logging.Logger):
    """Print summary statistics."""
    logger.info("\n" + "="*60)
    logger.info("EMBEDDING EXTRACTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files processed: {len(df)}")
    logger.info(f"Success: {len(df[df['status'] == 'success'])}")
    logger.info(f"Failed: {len(df[df['status'] == 'error'])}")
    logger.info(f"Embedding dimension: {EMBEDDING_SIZE}")
    logger.info("="*60)
    
    if len(df[df['status'] == 'success']) == 0:
        logger.warning("\nNo embeddings were successfully extracted")
        if len(df[df['status'] == 'error']) > 0:
            logger.info("\nErrors encountered:")
            errors_df = df[df['status'] == 'error']
            for idx, row in errors_df.head(5).iterrows():
                logger.info(f"  {row['video_id']}: {row['error']}")
    else:
        logger.info("\nAll embeddings saved as .npy files in embeddings/ directory")
        logger.info(f"Each file contains a {EMBEDDING_SIZE}-dimensional vector representing the audio content")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings using Wav2Vec2"
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default=AUDIO_DIR,
        help=f'Audio directory (default: {AUDIO_DIR})'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default=EMBEDDINGS_DIR,
        help=f'Embeddings output directory (default: {EMBEDDINGS_DIR})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_CSV,
        help=f'Output CSV file (default: {OUTPUT_CSV})'
    )
    parser.add_argument(
        '--links-csv',
        type=str,
        default=LINKS_CSV,
        help=f'Links CSV file (default: {LINKS_CSV})'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    log_file = script_dir / ERROR_LOG_FILE
    logger = setup_logging(str(log_file))
    
    logger.info("Audio Embedding Extraction (Wav2Vec2)")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    df = extract_all_embeddings(args.audio_dir, args.embeddings_dir, args.links_csv, args.output, logger)
    save_results(df, args.output, logger)
    print_summary(df, logger)
    
    logger.info("\nExtraction complete.")
    
    if len(df[df['status'] == 'error']) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

