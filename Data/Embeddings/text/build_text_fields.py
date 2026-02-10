from pathlib import Path
import pandas as pd

# Paths are relative to the project root:
# C:\CLIP Farming\video-virality-predictor\
metadata_path = Path("../../Metadata/shorts_metadata.csv")
transcript_path = Path("../../Text/text_results_english.csv")
output_path = Path("final_text_inputs.parquet")


def build_meta_text(row) -> str:
    """Combine title, description, and query into one text string."""
    parts = []
    for col in ["title", "description", "query"]:
        val = row.get(col)
        if isinstance(val, str):
            s = val.strip()
            if s:
                parts.append(s)
    return " [SEP] ".join(parts)


def main():
    print(f"Loading metadata from: {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    print(f"Loading transcripts from: {transcript_path}")
    transcripts = pd.read_csv(transcript_path)

    # We assume both have a 'url' column
    print("Merging on 'url'...")
    merged = metadata.merge(transcripts[["url", "transcript"]], on="url", how="inner")

    print(f"Merged rows: {len(merged)}")

    print("Building text_meta and text_transcript...")
    merged["text_meta"] = merged.apply(build_meta_text, axis=1)
    merged["text_transcript"] = (
        merged["transcript"].fillna("").astype(str).str.strip()
    )

    # Keep only what we need
    final_df = merged[["url", "text_meta", "text_transcript"]]

    # Make sure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_path}")
    final_df.to_parquet(output_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
