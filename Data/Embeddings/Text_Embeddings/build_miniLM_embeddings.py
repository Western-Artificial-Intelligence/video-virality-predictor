from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Input from previous step (where you saved it)
INPUT_PATH = Path("final_text_inputs.parquet")

# Output paths
OUTPUT_PARQUET_PATH = Path("text_embeddings.parquet")       # keeps your current parquet artifact
OUTPUT_NPY_WITH_IDS = Path("text_embeddings_with_ids.npy")  # NEW: {text_ids, embeddings} for fusion

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Loading text inputs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Rows loaded: {len(df)}")

    # Hard requirement: we need a stable ID column for fusion.
    # Prefer 'video_id' if you have it; otherwise fall back to URL.
    id_col = "video_id" if "video_id" in df.columns else "url"
    print(f"Using id column for fusion: {id_col}")

    # Load MiniLM model
    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode metadata text
    print("Encoding metadata text (text_meta)...")
    emb_meta = model.encode(
        df["text_meta"].fillna("").tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # Encode transcript text
    print("Encoding transcript text (text_transcript)...")
    emb_transcript = model.encode(
        df["text_transcript"].fillna("").tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # --- Choose what "the text embedding" is for fusion ---
    # If your fusion expects "meta_transcript_concat", concatenate the two embeddings.
    # Result: (N, 768) if each is (N, 384), etc.
    print("Creating fused text embedding for fusion (concat meta + transcript)...")
    emb_fused_text = np.concatenate([emb_meta, emb_transcript], axis=1).astype(np.float32)
    text_ids = df[id_col].astype(str).to_numpy()

    # Build output dataframe (keeps your previous outputs)
    print("Building output dataframe...")
    out_df = pd.DataFrame({
        id_col: df[id_col],
        "url": df["url"] if "url" in df.columns else df[id_col],
        "text_meta": df["text_meta"],
        "text_transcript": df["text_transcript"],
        "embedding_meta": [emb_meta[i].tolist() for i in range(len(df))],
        "embedding_transcript": [emb_transcript[i].tolist() for i in range(len(df))],
        "embedding_fused": [emb_fused_text[i].tolist() for i in range(len(df))],
        "embed_dim_meta": [emb_meta.shape[1]] * len(df),
        "embed_dim_transcript": [emb_transcript.shape[1]] * len(df),
        "embed_dim_fused": [emb_fused_text.shape[1]] * len(df),
    })

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving embeddings to: {OUTPUT_PARQUET_PATH}")
    out_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)

    # NEW: Save a dict-wrapped .npy for your fusion script
    # This is the key artifact you need to align by ID.
    print(f"Saving fusion-ready npy to: {OUTPUT_NPY_WITH_IDS}")
    np.save(
        OUTPUT_NPY_WITH_IDS,
        {"text_ids": text_ids, "embeddings": emb_fused_text},
        allow_pickle=True
    )

    print("Done.")
    print(f"Fusion-ready shapes: text_ids={text_ids.shape}, embeddings={emb_fused_text.shape}")


if __name__ == "__main__":
    main()
