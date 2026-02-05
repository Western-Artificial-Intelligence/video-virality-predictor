"""
Combine individual audio embeddings into a single .npy file.
Matches the format expected by the fusion script.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

EMBEDDINGS_DIR = "embeddings"
CSV_FILE = "audio_embeddings.csv"
OUTPUT_FILE = "audio_embeddings.npy"

def combine_embeddings():
    """Combine all audio embeddings into a single .npy file."""
    script_dir = Path(__file__).parent.absolute()
    
    # Read CSV to get order and video IDs
    csv_path = script_dir / CSV_FILE
    df = pd.read_csv(csv_path)
    
    # Only use successful embeddings
    df = df[df['status'] == 'success']
    
    # Sort by video_id for consistency
    df = df.sort_values('video_id').reset_index(drop=True)
    
    embeddings_dir = script_dir / EMBEDDINGS_DIR
    embeddings_list = []
    video_ids = []
    
    print(f"Loading embeddings from {len(df)} files...")
    
    for idx, row in df.iterrows():
        video_id = row['video_id']
        embedding_file = row['embedding_file']
        
        # Load embedding
        embedding_path = script_dir / embedding_file
        if embedding_path.exists():
            embedding = np.load(embedding_path)
            embeddings_list.append(embedding)
            video_ids.append(video_id)
        else:
            print(f"Warning: Embedding file not found: {embedding_file}")
    
    # Stack into single array: (num_videos, embedding_dim)
    embeddings_array = np.stack(embeddings_list, axis=0)
    
    # Save combined embeddings
    output_path = script_dir / OUTPUT_FILE
    np.save(output_path, embeddings_array)
    
    print(f"\nCombined embeddings saved to: {output_path}")
    print(f"Shape: {embeddings_array.shape} (num_videos={len(video_ids)}, dim={embeddings_array.shape[1]})")
    
    # Save video_id mapping
    mapping_path = script_dir / "audio_embeddings_order.txt"
    with open(mapping_path, 'w') as f:
        for vid_id in video_ids:
            f.write(f"{vid_id}\n")
    
    print(f"Video ID order saved to: {mapping_path}")
    
    return embeddings_array, video_ids

if __name__ == "__main__":
    print("Combining Audio Embeddings")
    print("="*60)
    combine_embeddings()
    print("\nDone.")



