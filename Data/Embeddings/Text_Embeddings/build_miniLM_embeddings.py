from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

# Input from previous step (where you saved it)
INPUT_PATH = Path("Data/Embeddings/Text_Embeddings/final_text_inputs.parquet")

# Output path for embeddings
OUTPUT_PATH = Path("Data/Embeddings/Text_Embeddings/text_embeddings.parquet")

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print(f"Loading text inputs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    print(f"Rows loaded: {len(df)}")

    # Load MiniLM model
    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode metadata text
    print("Encoding metadata text (text_meta)...")
    emb_meta = model.encode(
        df["text_meta"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Encode transcript text
    print("Encoding transcript text (text_transcript)...")
    emb_transcript = model.encode(
        df["text_transcript"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build output dataframe
    print("Building output dataframe...")
    out_df = pd.DataFrame({
        "url": df["url"],
        "text_meta": df["text_meta"],
        "text_transcript": df["text_transcript"],
        "embedding_meta": [emb_meta[i].tolist() for i in range(len(df))],
        "embedding_transcript": [emb_transcript[i].tolist() for i in range(len(df))],
        "embed_dim": [emb_meta.shape[1]] * len(df),
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving embeddings to: {OUTPUT_PATH}")
    out_df.to_parquet(OUTPUT_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
