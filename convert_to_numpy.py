import pandas as pd
import numpy as np

# 1) Load the parquet file
df = pd.read_parquet("Data/Embeddings/Text_Embeddings/text_embeddings.parquet")

print("Columns in parquet:")
print(df.columns)

# 2) Keep only numeric columns (assumed to be the embeddings)
numeric_df = df.select_dtypes(include=["number"])

print("\nUsing these numeric columns as embeddings:")
print(numeric_df.columns)

# 3) Convert to numpy array
emb = numeric_df.values.astype(np.float32)

# 4) Save as .npy
np.save("Data/Embeddings/Text_Embeddings/text_embeddings.npy", emb)

print("\nSaved numpy embeddings to Data/Embeddings/Text_Embeddings/text_embeddings.npy")
print("Shape:", emb.shape)
