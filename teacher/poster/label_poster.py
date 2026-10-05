"""
POSTER V2 teacher (face side): per-frame 8-way AffectNet emotion probabilities for every face crop
of one MAV-Celeb version. Run once per version:

    python teacher/poster/label_poster.py --faces_root mavceleb/v1/faces \
        --out_dir teacher_outputs/poster/mavceleb_v1 \
        --poster_repo POSTER_V2 --ckpt affectnet-8-model_best.pth

Writes one <speaker>__<Language>__<video>.npz per video folder: timestamps, track_ids,
probs (N x 8 in POSTER/AffectNet-8 order, NaN for unreadable frames), filenames, emotion_labels.
Existing files are skipped, so an interrupted run can simply be restarted.
Needs the POSTER V2 code (https://github.com/Talented-Q/POSTER_V2) and its AffectNet-8 checkpoint.
"""
import os
import sys
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

# ============ CONFIG ============
ap = argparse.ArgumentParser()
ap.add_argument("--faces_root", required=True, help="<mavceleb>/v{N}/faces  (id*/Language/video/*.jpg)")
ap.add_argument("--out_dir", required=True, help="e.g. teacher_outputs/poster/mavceleb_v{N}")
ap.add_argument("--poster_repo", default="POSTER_V2", help="clone of the POSTER V2 code")
ap.add_argument("--ckpt", default="affectnet-8-model_best.pth", help="POSTER V2 AffectNet-8 checkpoint")
ARGS = ap.parse_args()
DATASET_ROOT = ARGS.faces_root
OUTPUT_DIR   = ARGS.out_dir
CHECKPOINT   = ARGS.ckpt

BATCH_SIZE       = 128
TRACK_GAP_MS     = 500         # gap > this between consecutive frames -> new track
IMAGE_SIZE       = 224         # POSTER V2 input size
USE_MTCNN        = False       # MAV-Celeb faces are pre-cropped
MAX_SPEAKERS     = None        # e.g. 2 for a smoke test

EMOTION_LABELS = ['neutral', 'happiness', 'sadness', 'surprise',
                  'fear', 'disgust', 'anger', 'contempt']
NUM_EMOTIONS = 8

# ============ POSTER V2 import ============
POSTER_PATH = os.path.abspath(ARGS.poster_repo)
if POSTER_PATH not in sys.path:
    sys.path.append(POSTER_PATH)
from models.PosterV2_8cls import pyramid_trans_expr2

# Required for checkpoint unpickling (don't remove)
class RecorderMeter1:
    def __init__(self, *args, **kwargs): pass
class RecorderMeter:
    def __init__(self, *args, **kwargs): pass

# ============ MODEL ============
def load_model(device):
    model = pyramid_trans_expr2(img_size=IMAGE_SIZE, num_classes=NUM_EMOTIONS)
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model

# ============ TRACK SPLITTING ============
def assign_track_ids(timestamps, gap_ms=TRACK_GAP_MS):
    ids = np.zeros(len(timestamps), dtype=np.int32)
    cur = 0
    for i in range(1, len(timestamps)):
        if timestamps[i] - timestamps[i-1] > gap_ms:
            cur += 1
        ids[i] = cur
    return ids

# ============ PREPROCESS ============
def preprocess_batch(jpg_paths, mean, std, device):
    tensors, valid_mask = [], []
    for path in jpg_paths:
        img = cv2.imread(path)
        if img is None:
            valid_mask.append(False)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        tensors.append(t)
        valid_mask.append(True)
    if not tensors:
        return None, valid_mask
    batch = torch.stack(tensors).to(device, non_blocking=True)
    batch = (batch - mean) / std
    return batch, valid_mask

# ============ ONE YOUTUBE-ID FOLDER ============
def process_youtube_folder(yid_path, model, mean, std, device, out_path):
    files = sorted(f for f in os.listdir(yid_path) if f.endswith('.jpg'))
    if not files:
        return None

    timestamps = np.array([int(f.split('.')[0]) for f in files], dtype=np.int64)
    track_ids  = assign_track_ids(timestamps)
    n = len(files)
    all_probs = np.full((n, NUM_EMOTIONS), np.nan, dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_paths = [os.path.join(yid_path, f) for f in files[start:end]]
        batch, valid_mask = preprocess_batch(batch_paths, mean, std, device)
        if batch is None:
            continue
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        vi = 0
        for i, ok in enumerate(valid_mask):
            if ok:
                all_probs[start + i] = probs[vi]
                vi += 1

    np.savez_compressed(
        out_path,
        timestamps=timestamps,
        track_ids=track_ids,
        probs=all_probs,
        filenames=np.array(files),
        emotion_labels=np.array(EMOTION_LABELS),
    )
    return n, int(track_ids.max()) + 1

# ============ MAIN ============
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output dir:   {OUTPUT_DIR}")

    model = load_model(device)
    mean  = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std   = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    speakers = sorted(d for d in os.listdir(DATASET_ROOT) if d.startswith("id"))
    if MAX_SPEAKERS is not None:
        speakers = speakers[:MAX_SPEAKERS]
        print(f"SMOKE TEST: processing first {MAX_SPEAKERS} speaker(s)")
    print(f"Speakers to process: {len(speakers)}\n")

    total_frames = 0
    total_tracks = 0
    total_folders = 0
    skipped = 0

    for si, spk in enumerate(speakers, 1):
        spk_path = os.path.join(DATASET_ROOT, spk)
        if not os.path.isdir(spk_path):
            continue
        for lang in os.listdir(spk_path):
            lang_path = os.path.join(spk_path, lang)
            if not os.path.isdir(lang_path):
                continue
            for yid in os.listdir(lang_path):
                yid_path = os.path.join(lang_path, yid)
                if not os.path.isdir(yid_path):
                    continue

                out_name = f"{spk}__{lang}__{yid}.npz"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                if os.path.exists(out_path):
                    skipped += 1
                    continue

                result = process_youtube_folder(yid_path, model, mean, std, device, out_path)
                if result is None:
                    print(f"[{si}/{len(speakers)}] EMPTY  {spk}/{lang}/{yid}")
                    continue
                n_frames, n_tracks = result
                total_frames  += n_frames
                total_tracks  += n_tracks
                total_folders += 1
                print(f"[{si}/{len(speakers)}] {spk}/{lang}/{yid:30s}  "
                      f"{n_frames:5d} frames, {n_tracks} track(s)")

    print(f"\n=== DONE ===")
    print(f"Folders processed: {total_folders} (skipped existing: {skipped})")
    print(f"Total tracks:      {total_tracks}")
    print(f"Total frames:      {total_frames}")

if __name__ == "__main__":
    main()