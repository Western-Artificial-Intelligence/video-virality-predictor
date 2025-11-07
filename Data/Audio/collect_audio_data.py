"""
YouTube Shorts Audio Collector
-------------------------------
Downloads audio-only tracks from YouTube Shorts links in CSV format.
Converts to .m4a format with idempotent downloads, retry logic, and error logging.
"""

import os
import re
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
import yt_dlp


DEFAULT_CSV_PATH = "../Links/shorts_data/shorts_links_wide.csv"
DEFAULT_OUTPUT_DIR = "raw_audio"
ERROR_LOG_FILE = "audio_download_errors.log"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2


def setup_logging(log_file: str) -> logging.Logger:
    """Set up logging to both file and console."""
    logger = logging.getLogger("audio_collector")
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


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube Shorts URL."""
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


def get_output_filename(video_id: str, output_dir: str) -> str:
    """Get the full path for the output .m4a file."""
    return os.path.join(output_dir, f"{video_id}.m4a")


def file_exists(video_id: str, output_dir: str) -> bool:
    """Check if the audio file already exists."""
    output_file = get_output_filename(video_id, output_dir)
    return os.path.exists(output_file) and os.path.getsize(output_file) > 0


def download_audio(
    url: str,
    video_id: str,
    output_dir: str,
    logger: logging.Logger,
    retry_count: int = 0
) -> Tuple[bool, Optional[str]]:
    """Download audio from YouTube URL with retry logic."""
    output_file = get_output_filename(video_id, output_dir)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, f'{video_id}.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'noplaylist': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return True, None
        else:
            error_msg = f"File not created or empty: {output_file}"
            return False, error_msg
            
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg or "Video unavailable" in error_msg:
            return False, f"Video unavailable: {error_msg}"
        elif retry_count < MAX_RETRIES:
            delay = RETRY_DELAY_BASE * (2 ** retry_count)
            logger.info(f"    Retrying in {delay}s (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
            time.sleep(delay)
            return download_audio(url, video_id, output_dir, logger, retry_count + 1)
        else:
            return False, f"Download failed after {MAX_RETRIES + 1} attempts: {error_msg}"
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if retry_count < MAX_RETRIES:
            delay = RETRY_DELAY_BASE * (2 ** retry_count)
            logger.info(f"    Retrying in {delay}s (attempt {retry_count + 2}/{MAX_RETRIES + 1})")
            time.sleep(delay)
            return download_audio(url, video_id, output_dir, logger, retry_count + 1)
        else:
            return False, error_msg


def collect_audio_data(
    csv_path: str,
    output_dir: str,
    logger: logging.Logger
) -> dict:
    """Main function to collect audio data from CSV."""
    script_dir = Path(__file__).parent.absolute()
    csv_full_path = (script_dir / csv_path).resolve()
    
    if not csv_full_path.exists():
        logger.error(f"CSV file not found: {csv_full_path}")
        sys.exit(1)
    
    output_full_path = (script_dir / output_dir).resolve()
    output_full_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading CSV: {csv_full_path}")
    try:
        df = pd.read_csv(csv_full_path)
        if "url" not in df.columns:
            logger.error("CSV must contain a 'url' column.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        sys.exit(1)
    
    urls = df["url"].dropna().unique().tolist()
    total_urls = len(urls)
    
    logger.info(f"Found {total_urls} unique URLs in CSV")
    logger.info(f"Output directory: {output_full_path}")
    logger.info(f"Error log: {script_dir / ERROR_LOG_FILE}\n")
    
    stats = {
        'total': total_urls,
        'skipped': 0,
        'downloaded': 0,
        'failed': 0,
        'errors': []
    }
    for idx, url in enumerate(urls, 1):
        video_id = extract_video_id(url)
        
        if not video_id:
            error_msg = f"Could not extract video ID from URL: {url}"
            logger.error(f"[{idx}/{total_urls}] Failed: {error_msg}")
            stats['failed'] += 1
            stats['errors'].append({'url': url, 'error': error_msg})
            continue
        
        if file_exists(video_id, str(output_full_path)):
            logger.info(f"[{idx}/{total_urls}] Skipped (exists): {video_id}")
            stats['skipped'] += 1
            continue
        
        logger.info(f"[{idx}/{total_urls}] Downloading: {video_id} | {url}")
        success, error_msg = download_audio(url, video_id, str(output_full_path), logger)
        
        if success:
            logger.info(f"    Success: {video_id}.m4a")
            stats['downloaded'] += 1
        else:
            logger.error(f"    Failed: {error_msg}")
            logger.error(f"[ERROR] {url} | {error_msg}", exc_info=False)
            stats['failed'] += 1
            stats['errors'].append({'url': url, 'video_id': video_id, 'error': error_msg})
    
    return stats


def print_summary(stats: dict, logger: logging.Logger):
    """Print summary statistics."""
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Total URLs processed:  {stats['total']}")
    logger.info(f"Successfully downloaded: {stats['downloaded']}")
    logger.info(f"Skipped (already exists): {stats['skipped']}")
    logger.info(f"Failed:                  {stats['failed']}")
    logger.info("="*60)
    
    if stats['errors']:
        logger.info(f"\n{len(stats['errors'])} errors logged to {ERROR_LOG_FILE}")
        logger.info("First few errors:")
        for i, error in enumerate(stats['errors'][:5], 1):
            logger.info(f"  {i}. {error.get('video_id', 'N/A')}: {error['error'][:80]}")
        if len(stats['errors']) > 5:
            logger.info(f"  ... and {len(stats['errors']) - 5} more (see {ERROR_LOG_FILE})")
    else:
        logger.info("\nAll downloads completed successfully")
    
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube Shorts links in CSV format"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f'Path to CSV file (default: {DEFAULT_CSV_PATH})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    log_file = script_dir / ERROR_LOG_FILE
    logger = setup_logging(str(log_file))
    
    logger.info("YouTube Shorts Audio Collector")
    logger.info("="*60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    stats = collect_audio_data(args.csv, args.output, logger)
    print_summary(stats, logger)
    
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
