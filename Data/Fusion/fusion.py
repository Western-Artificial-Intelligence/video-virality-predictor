# Fusing VA & VA + T embeddings using simple MLPs

#MACROS
VIDEO_EMB_PATH = "video-virality-predictor/Data/Embeddings/videomae_embeddings.npy"
AUDIO_EMB_PATH = ""
TEXT_EMB_PATH  = "video-virality-predictor/Data/Embeddings/Text_Embeddings/text_embeddings.npy"
OUTPUT_PATH = "fused.pt"
FUSED_DIM = 512 # dimension of fused embeddings (for now)

#IMPORTS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
load_emb = lambda p: torch.tensor(pd.read_parquet(p).values, dtype=torch.float32)

#MLP Definition
class MLP(nn.Module):
    def __init__(self, d_in, d_out): 
        super().__init__(); self.f = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in,d_out), nn.GELU(), nn.Linear(d_out,d_out), nn.GELU())
    def forward(self, x): return self.f(x)

#Basically Just checking the same # of observations exist in each modality
video, audio, text = load_emb(VIDEO_EMB_PATH).to(DEVICE), load_emb(AUDIO_EMB_PATH).to(DEVICE), load_emb(TEXT_EMB_PATH).to(DEVICE)
assert video.shape[0] == audio.shape[0] == text.shape[0]

vid_mlp, aud_mlp, txt_mlp = MLP(video.shape[1],FUSED_DIM).to(DEVICE), MLP(audio.shape[1],FUSED_DIM).to(DEVICE), MLP(text.shape[1],FUSED_DIM).to(DEVICE)
va_fuse, final_fuse = MLP(2*FUSED_DIM,FUSED_DIM).to(DEVICE), MLP(2*FUSED_DIM,FUSED_DIM).to(DEVICE)

#Apply Simple MLPs to each modality and fuse
v, a, t = vid_mlp(video), aud_mlp(audio), txt_mlp(text)  
va = va_fuse(torch.cat([v,a],1))
fused = final_fuse(torch.cat([va,t],1))

# Save & We done
print("Fused:", fused.shape)
torch.save(fused.cpu(), OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)
