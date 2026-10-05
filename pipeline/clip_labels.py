"""
Step 1 -- clip-level soft labels from the raw per-frame teacher outputs (both teachers).

For every video, the frame probabilities of a teacher are MAX-pooled over all frames with a face and
renormalised to sum to 1. The resulting 8-way vector is the label of every audio segment (wav) of that
video. Only videos that both teachers labelled and that have audio are kept (44,083 clips per teacher).

  * Probability columns are in POSTER V2 / AffectNet-8 order. SENet (FER+) columns are remapped BY NAME.
  * Video ids may themselves contain '__' (YouTube ids), so file names are split with maxsplit=2.

Inputs : $MAVCELEB_TEACHER_ROOT/poster/mavceleb_v{N}/*.npz, .../senet/mavceleb_v{N}/*.csv,
         $MAVCELEB_DATA_ROOT/v{N}/voices/<speaker>/<Language>/<video>/*.wav
Output : data/clip_labels.csv   one row per (clip, teacher); wav_path relative to DATA_ROOT

data/clip_labels.csv ships with the repository, so this step is only needed to re-derive it.
"""
import os, glob
import numpy as np
import pandas as pd

from paths import DATA_ROOT, TEACHER_ROOT, CLIP_LABELS, VERSIONS, POSTER_ORDER

POSTER_OUT = os.path.join(TEACHER_ROOT, "poster")
SENET_OUT = os.path.join(TEACHER_ROOT, "senet")
# The order the SENet dumper writes its CSV columns in (FER+)
SENET_ORDER = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
PCOLS = [f"p_{e}" for e in POSTER_ORDER]
COLS = (["clip_id", "wav_path", "speaker_id", "version", "language", "teacher", "video_id",
         "n_frames_total", "n_frames_with_face"] + PCOLS + ["argmax_label", "max_prob", "entropy"])


def parse_name(path):
    """id__language__video_id  ->  (id, language, video_id); video ids may contain '__'."""
    base = os.path.basename(path)
    base = base[:-4] if base.lower().endswith((".npz", ".csv")) else base
    parts = base.split("__", 2)
    return tuple(parts) if len(parts) == 3 else None


def pool(P):
    """Max-pool an (N,8) frame-probability array to one 8-vector, then renormalise."""
    if P.size == 0:
        return None
    v = np.nanmax(P, axis=0)
    s = v.sum()
    return v / s if s > 0 else np.ones(8) / 8


def entropy(p):
    """Shannon entropy in nats of a normalised probability vector."""
    q = np.clip(p, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())


def load_poster(npz_path):
    probs = np.load(npz_path, allow_pickle=True)["probs"].astype(np.float64)   # (N,8) in POSTER_ORDER
    keep = ~np.isnan(probs).all(axis=1)
    return pool(probs[keep]), probs.shape[0], int(keep.sum())


def load_senet(csv_path):
    P = pd.read_csv(csv_path)[SENET_ORDER].to_numpy(dtype=np.float64)          # select BY NAME
    keep = ~np.isnan(P).all(axis=1) if P.shape[0] else np.zeros(0, bool)
    v = pool(P[keep])
    if v is None:
        return None, P.shape[0], int(keep.sum())
    return v[[SENET_ORDER.index(c) for c in POSTER_ORDER]], P.shape[0], int(keep.sum())


def build(version):
    """Clip rows of one version, both teachers (rows of one video alternate teacher by teacher)."""
    pf = {parse_name(f): f for f in glob.glob(os.path.join(POSTER_OUT, f"mavceleb_{version}", "*.npz"))}
    sf = {parse_name(f): f for f in glob.glob(os.path.join(SENET_OUT, f"mavceleb_{version}", "*.csv"))}
    pf = {k: v for k, v in pf.items() if k}
    sf = {k: v for k, v in sf.items() if k}
    common = sorted(set(pf) & set(sf))

    rows, n_no_wav = [], 0
    for speaker, language, video in common:
        vdir = os.path.join(DATA_ROOT, version, "voices", speaker, language, video)
        wavs = sorted(glob.glob(os.path.join(vdir, "*.wav"))) if os.path.isdir(vdir) else []
        if not wavs:
            n_no_wav += 1
            continue
        for teacher, loader, src in (("poster", load_poster, pf[(speaker, language, video)]),
                                     ("senet", load_senet, sf[(speaker, language, video)])):
            v, n_total, n_face = loader(src)
            if v is None:
                continue
            am = int(np.argmax(v))
            for w in wavs:
                r = {
                    # speaker ids are version-local (v1/id0001 != v3/id0001), so the version is part of the key
                    "clip_id": f"{version}__{speaker}__{language}__{video}__{os.path.splitext(os.path.basename(w))[0]}",
                    "wav_path": os.path.relpath(w, DATA_ROOT).replace(os.sep, "/"),
                    "speaker_id": speaker, "version": version, "language": language.lower(),
                    "teacher": teacher, "video_id": video,
                    "n_frames_total": n_total, "n_frames_with_face": n_face,
                }
                r.update({c: float(v[i]) for i, c in enumerate(PCOLS)})
                r["argmax_label"] = POSTER_ORDER[am]
                r["max_prob"] = float(v[am])
                r["entropy"] = entropy(v)
                rows.append(r)
    print(f"[{version}] matched_videos={len(common)} skipped_no_wav={n_no_wav} clip_rows={len(rows)}")
    return pd.DataFrame(rows)


def main():
    full = pd.concat([build(v) for v in VERSIONS], ignore_index=True)

    s = full[PCOLS].sum(axis=1)
    assert np.allclose(s, 1.0, atol=1e-9), f"prob rows not normalised: {s.min()}..{s.max()}"
    recomputed = [POSTER_ORDER[i] for i in full[PCOLS].to_numpy().argmax(axis=1)]
    assert (np.array(recomputed) == full["argmax_label"].to_numpy()).all(), "argmax mismatch"

    os.makedirs(os.path.dirname(CLIP_LABELS), exist_ok=True)
    full[COLS].to_csv(CLIP_LABELS, index=False)
    print(f"wrote {CLIP_LABELS}  n={len(full)}")


if __name__ == "__main__":
    main()
