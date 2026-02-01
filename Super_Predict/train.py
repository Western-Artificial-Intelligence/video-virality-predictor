import pandas as pd
from virality_model import ViralityConfig, train_virality_model

FUSED_PT = "video-virality-predictor/Data/Fusion/fused.pt"
TRAIN_CSV = "video-virality-predictor/Super_Predict/train.csv" 

train = pd.read_csv(TRAIN_CSV)

# ---- Sanity checks
required = ["video_id", "log_view_count"]
missing = [c for c in required if c not in train.columns]
if missing:
    raise ValueError(f"Missing required columns in {TRAIN_CSV}: {missing}")

cfg = ViralityConfig(
    target_col="log_view_count",
    categorical_cols=["cluster_id"] if "cluster_id" in train.columns else None
)

out = train_virality_model(
    fused_pt_path=FUSED_PT,
    features_df=train,
    config=cfg
)

print(out["metrics"])
print(out["feature_importance"].head(20))
