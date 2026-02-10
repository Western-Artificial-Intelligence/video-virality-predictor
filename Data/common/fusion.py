# Fusing V + A + T using mapping-table join by (query, index)
# video_ids look like: "funny Shorts10"
# map_df["query"] looks like: "funny Shorts"
# map_df["url"] holds the actual YouTube shorts URLs

VIDEO_EMB_PATH = "../embeddings/video/videomae_embeddings.npy"
AUDIO_EMB_PATH = "../embeddings/audio/audio_embeddings.npy"
TEXT_EMB_PATH  = "../embeddings/text/text_embeddings_with_ids.npy"

MAP_PATH = "../Links/shorts_data/shorts_links_wide.csv"   # csv or parquet

OUTPUT_PATH = "fused.pt"
FUSED_DIM = 512

FUSE_TEXT = True
TEXT_REQUIRED = True

import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ID_KEYS  = ["ids", "video_ids", "audio_ids", "text_ids", "video_id", "clip_id", "id", "url"]
EMB_KEYS = ["embeddings", "embedding", "features", "feats", "data", "x", "arr_0"]


def _pick_key(d: dict, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def load_npy_ids_and_emb_or_raw(path: str, label: str):
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    if isinstance(raw, dict):
        emb_key = _pick_key(raw, EMB_KEYS)
        if emb_key is None:
            raise KeyError(f"{label}: dict missing embedding key. keys={list(raw.keys())}")
        emb = np.asarray(raw[emb_key], dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"{label}: expected embeddings [N,D], got {emb.shape}")

        id_key = _pick_key(raw, ID_KEYS)
        ids = np.asarray(raw[id_key]).astype(str) if id_key is not None else None
        if ids is not None and ids.shape[0] != emb.shape[0]:
            raise ValueError(f"{label}: ids len {ids.shape[0]} != emb rows {emb.shape[0]}")
        return ids, torch.from_numpy(emb)

    if isinstance(raw, np.ndarray) and raw.dtype != object:
        emb = np.asarray(raw, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"{label}: expected embeddings [N,D], got {emb.shape}")
        return None, torch.from_numpy(emb)

    raise TypeError(f"{label}: unsupported npy serialization type: {type(raw)}")


def extract_youtube_id(s: str) -> str:
    s = str(s).strip()
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,32})", s)
    if m: return m.group(1)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,32})", s)
    if m: return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,32})", s)
    if m: return m.group(1)
    return s


def load_map_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"MAP_PATH must be .csv or .parquet. Got: {path}")


def parse_video_label(label: str):
    """
    "funny Shorts10" -> ("funny Shorts", 10)
    Returns (base_query, idx_int) or (None, None) if it doesn't match.
    """
    s = str(label).strip()
    m = re.match(r"^(.*?)(\d+)$", s)  # everything then trailing digits
    if not m:
        return None, None
    base = m.group(1).strip()
    idx = int(m.group(2))
    return base, idx


# -------------------------
# Load embeddings
# -------------------------
v_ids, video = load_npy_ids_and_emb_or_raw(VIDEO_EMB_PATH, "video")
_, audio = load_npy_ids_and_emb_or_raw(AUDIO_EMB_PATH, "audio")

print("Loaded:")
print("  video:", tuple(video.shape), video.dtype, "ids:", None if v_ids is None else len(v_ids))
print("  audio:", tuple(audio.shape), audio.dtype, "ids: None (raw)")

if v_ids is None:
    raise ValueError("Video embeddings must include IDs (video_ids) for any sound text join.")
if video.shape[0] != audio.shape[0]:
    raise ValueError(f"VA row mismatch: video {video.shape[0]} vs audio {audio.shape[0]}")

print("[WARN] Audio has no IDs; assuming audio row i corresponds to video row i.")

# -------------------------
# Text + join
# -------------------------
if FUSE_TEXT:
    t_ids, text = load_npy_ids_and_emb_or_raw(TEXT_EMB_PATH, "text")
    print("  text :", tuple(text.shape), text.dtype, "ids:", None if t_ids is None else len(t_ids))
    if t_ids is None:
        raise ValueError("Text embeddings must include IDs (text_ids/url) to fuse by ID.")

    # Text ids are URLs -> normalize to youtube id
    text_yids = [extract_youtube_id(x) for x in t_ids]
    text_map = {text_yids[i]: text[i] for i in range(len(text_yids))}

    # Load mapping file (must have columns: query, url)
    map_df = load_map_df(MAP_PATH)
    if "query" not in map_df.columns or "url" not in map_df.columns:
        raise ValueError(f"Mapping file must have columns ['query','url']. Have {list(map_df.columns)}")

    # Build ordered URL lists per query (preserve file order)
    map_df["query"] = map_df["query"].astype(str).str.strip()
    map_df["url"] = map_df["url"].astype(str).str.strip()
    query_to_urls = map_df.groupby("query", sort=False)["url"].apply(list).to_dict()

    # Debug: show a couple
    print("\nMapping sample:")
    for q in list(query_to_urls.keys())[:3]:
        print(f"  {q}: {len(query_to_urls[q])} urls")

    # Align by (base_query, index)
    video_index = {str(v_ids[i]).strip(): i for i in range(len(v_ids))}

    kept_labels = []
    V_list, A_list, T_list = [], [], []

    missing_parse = 0
    missing_query = 0
    oob_index = 0
    missing_text = 0

    # track whether 1-based or 0-based seems to work
    hits_1based = 0
    hits_0based = 0

    # Pre-scan to decide base (1-based usually correct)
    for label in v_ids[:200]:
        base, idx = parse_video_label(label)
        if base is None:
            continue
        urls = query_to_urls.get(base)
        if not urls:
            continue
        if 1 <= idx <= len(urls):
            hits_1based += 1
        if 0 <= idx < len(urls):
            hits_0based += 1

    use_one_based = hits_1based >= hits_0based
    print(f"\nIndexing guess: 1-based hits={hits_1based}, 0-based hits={hits_0based} -> using {'1-based' if use_one_based else '0-based'}")

    for label in v_ids:
        label = str(label).strip()
        base, idx = parse_video_label(label)
        if base is None:
            missing_parse += 1
            continue

        urls = query_to_urls.get(base)
        if urls is None:
            missing_query += 1
            continue

        j = (idx - 1) if use_one_based else idx
        if j < 0 or j >= len(urls):
            oob_index += 1
            continue

        url = urls[j]
        yid = extract_youtube_id(url)

        t_emb = text_map.get(yid)
        if t_emb is None:
            missing_text += 1
            continue

        i = video_index[label]
        kept_labels.append(label)
        V_list.append(video[i])
        A_list.append(audio[i])
        T_list.append(t_emb)

    print("\nJoin stats:")
    print(f"  kept={len(kept_labels)}")
    print(f"  missing_parse={missing_parse} (video_id didn't end with digits)")
    print(f"  missing_query={missing_query} (base query not in map)")
    print(f"  oob_index={oob_index} (index outside url list length)")
    print(f"  missing_text={missing_text} (url->yid not in text embeddings)")

    if TEXT_REQUIRED and len(kept_labels) == 0:
        raise ValueError(
            "After (query,index) join, 0 samples matched.\n"
            "This likely means your numeric suffix (e.g., Shorts10) is NOT the rank within map_df[query],\n"
            "or map_df is not in the same ordering used when naming video_ids.\n"
            "Next fix is to export video embeddings with URL/YouTubeID, or create a proper manifest mapping label->url."
        )

    V = torch.stack(V_list, dim=0)
    A = torch.stack(A_list, dim=0)
    T = torch.stack(T_list, dim=0)
    final_ids_original = kept_labels

else:
    V, A, T = video, audio, None
    final_ids_original = list(v_ids)

# -------------------------
# Move to device
# -------------------------
V = V.to(DEVICE)
A = A.to(DEVICE)
if FUSE_TEXT:
    T = T.to(DEVICE)

# -------------------------
# Models
# -------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
            nn.GELU(),
        )
    def forward(self, x): return self.f(x)

vid_mlp = MLP(V.shape[1], FUSED_DIM).to(DEVICE)
aud_mlp = MLP(A.shape[1], FUSED_DIM).to(DEVICE)
va_fuse = MLP(2 * FUSED_DIM, FUSED_DIM).to(DEVICE)

if FUSE_TEXT:
    txt_mlp = MLP(T.shape[1], FUSED_DIM).to(DEVICE)
    final_fuse = MLP(2 * FUSED_DIM, FUSED_DIM).to(DEVICE)

# -------------------------
# Forward
# -------------------------
v = vid_mlp(V)
a = aud_mlp(A)
va = va_fuse(torch.cat([v, a], dim=1))

if FUSE_TEXT:
    t = txt_mlp(T)
    fused = final_fuse(torch.cat([va, t], dim=1))
else:
    fused = va

print("\nFused:", fused.shape)
torch.save({"ids": final_ids_original, "fused": fused.detach().cpu()}, OUTPUT_PATH)
print("Saved:", OUTPUT_PATH)
