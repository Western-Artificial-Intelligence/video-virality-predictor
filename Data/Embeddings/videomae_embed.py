import os
import numpy as np
import torch
import av
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from transformers import VideoMAEImageProcessor, VideoMAEModel

# --------------------------------------------------------
# CONFIG: You can change these paths/values as needed.
# --------------------------------------------------------

# Folder where your input videos are stored.
# Each file in here with one of the allowed extensions will be processed.
VIDEO_DIR = "Data/Videos"

# Where to save the final numpy file containing:
#   - video_ids: array of strings (file names without extension)
#   - embeddings: array of shape (N, D) where D is embedding dim
# OUT_PATH = "Data/Embeddings/videomae_embeddings.npy"
OUT_PATH = "Data/Embeddings/videomae_embeddings.parquet"

# How many frames to uniformly sample per video.
# Higher = more temporal coverage but more compute.
NUM_FRAMES = 16

# Name of the pretrained VideoMAE model to use.
# You can swap to a different size/checkpoint if needed.
MODEL_NAME = "MCG-NJU/videomae-base"
# --------------------------------------------------------

# Choose the best available device:
# 1) CUDA GPU if available
# 2) Apple Silicon GPU (MPS) if available
# 3) Fallback to CPU
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def sample_frames(video_path, num_frames=16):
    """
    Robust frame sampler that:
    - Decodes all video frames (works even when stream.frames == 0).
    - Uniformly samples `num_frames` frames across the decoded sequence.
    """
    container = av.open(video_path)

    # Decode ALL frames from the first video stream.
    frames = [
        frame.to_ndarray(format="rgb24")
        for frame in container.decode(video=0)
    ]

    total = len(frames)
    if total == 0:
        # This really means PyAV couldn't decode any video frames (e.g. audio-only or corrupt file).
        raise ValueError(f"No decodable video frames found in {video_path}")

    # Pick `num_frames` indices uniformly from [0, total-1].
    indices = np.linspace(0, total - 1, num_frames).astype(int)

    # Gather the sampled frames.
    sampled = [frames[i] for i in indices]

    return np.stack(sampled)  # (T, H, W, 3)


class VideoMAEEmbedder:
    """
    Wrapper class around a pretrained VideoMAE model.

    Responsibilities:
    - Load the VideoMAE model + processor.
    - Given a video path, sample frames and prepare them as model input.
    - Run a forward pass and extract a single embedding (CLS token).
    """

    def __init__(self):
        # Load the image processor that handles resizing, cropping, normalization, etc.
        # This ensures our frames match exactly the training-time preprocessing of VideoMAE.
        print(f"Loading model `{MODEL_NAME}` on {DEVICE} ...")
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

        # Load the pretrained VideoMAE model weights and move them to the chosen device.
        self.model = VideoMAEModel.from_pretrained(MODEL_NAME).to(DEVICE)

        # Put the model in evaluation mode (disables dropout, etc.) since we're only embedding.
        self.model.eval()

    @torch.no_grad()
    def embed_video(self, video_path):
        """
        Compute a single L2-normalized embedding for an entire video.

        Steps:
        - Sample NUM_FRAMES frames from the video.
        - Use the processor to turn frames into model input tensors.
        - Run the model, take the [CLS] token as a global video representation.
        - L2-normalize the embedding so its norm is 1.
        - Return as a NumPy vector of shape (D,).
        """
        # Sample frames from the video as raw RGB arrays.
        frames = sample_frames(video_path, num_frames=NUM_FRAMES)

        # The processor expects a sequence of images (frames) and returns ready-to-use tensors.
        # It will handle resizing, cropping, normalization, and casting to torch.Tensor.
        inputs = self.processor(list(frames), return_tensors="pt")

        # Move all tensors (e.g., pixel_values) to the correct device (CPU/GPU/MPS).
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Forward pass through VideoMAE.
        # outputs.last_hidden_state has shape (batch_size, num_tokens, hidden_dim).
        outputs = self.model(**inputs)

        # The first token (index 0) is the [CLS] token, a summary representation of the video.
        # We take it from batch index 0 because our batch size = 1.
        cls_emb = outputs.last_hidden_state[:, 0][0]  # shape: (hidden_dim,)

        # L2-normalize the embedding so it lies on the unit sphere.
        # This often makes downstream similarity / regression more stable.
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)

        # Move back to CPU and convert to NumPy for storage or further processing.
        return cls_emb.cpu().numpy()




def save_embeddings_parquet(video_ids, embeddings, out_path):
    """
    Save embeddings to a Parquet file with columns:
    - video_id (string)
    - embedding (fixed-length list of floats)
    """

    # Convert to list for compatibility with Arrow
    ids = list(video_ids)
    embs = [emb.tolist() for emb in embeddings]

    df = pd.DataFrame({
        "video_id": ids,
        "embedding": embs,
    })

    # Convert DataFrame → Arrow Table → Parquet
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)
    print(f"Saved Parquet file → {out_path}")


def main():
    """
    Main entry point.

    Steps:
    - Ensure the output directory exists.
    - Instantiate the embedder (loads model onto device).
    - Find all videos in VIDEO_DIR with allowed extensions.
    - For each video: compute embedding and store video id + embedding.
    - Save everything into a single .npy file (dict with video_ids & embeddings).
    """
    # Make sure the directory for OUT_PATH exists (e.g., Data/).
    # exist_ok=True means "don't error if it already exists".
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Initialize the VideoMAE embedder (loads model and processor).
    embedder = VideoMAEEmbedder()

    # Gather all video files in VIDEO_DIR with supported extensions.
    video_files = [
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".mov", ".mkv"))
    ]

    print(f"Found {len(video_files)} videos in `{VIDEO_DIR}`.")

    video_ids = []   # will hold file names without extension
    embeddings = []  # will hold one embedding vector per video

    # Loop over every video file we found.
    for path in video_files:
        # Use the filename (without extension) as the video ID.
        vid_id = os.path.splitext(os.path.basename(path))[0]
        print(f"Embedding {vid_id} ...")

        # Compute a (D,) embedding vector for this video.
        try:
            emb = embedder.embed_video(path)
        except Exception as e:
            print(f"  Skipping {vid_id} due to error: {e}")
            continue

        # Store ID and embedding in parallel lists.
        video_ids.append(vid_id)
        embeddings.append(emb)


        ##### for npy
#     # Convert to a dict so it's easy to load later.
#     data = {
#         "video_ids": np.array(video_ids),
#         "embeddings": np.stack(embeddings),  # shape: (N, D)
#     }

#     # Save the dict into a single .npy file.
#     # We use allow_pickle=True when loading later to reconstruct the dict.
#     np.save(OUT_PATH, data)
#     print(f"\nDone. Saved {len(video_ids)} embeddings to `{OUT_PATH}`")

    ##### for parquet
    # ---- Save to Parquet here ----
    save_embeddings_parquet(video_ids, embeddings, OUT_PATH)

    print(f"\nDone. Saved {len(video_ids)} embeddings to `{OUT_PATH}`")



# Standard Python pattern: only run main() if this file is executed directly.
# If you import this file as a module, main() will NOT run automatically.
if __name__ == "__main__":
    main()
