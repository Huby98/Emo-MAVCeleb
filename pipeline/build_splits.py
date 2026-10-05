
import os
import numpy as np
import pandas as pd

from paths import CLIP_LABELS, SPLITS, VERSIONS, PAIRS, POSTER_ORDER

SEED = 1337
POOLED_VAL_FRAC = 0.10
TEST_FRAC = VAL_FRAC = 0.15          # known splits


def to_train_format(df):
    """Clip-label rows -> the columns student/train_student.EmoVoxDataset reads (labels selected BY NAME)."""
    out = df.rename(columns={f"p_{e}": e for e in POSTER_ORDER}).copy()
    out["celebrity"] = out["speaker_id"]
    out["n_frames"] = out["n_frames_total"]
    out["pred_class"] = out["argmax_label"]
    return out[["wav_path", "celebrity", "language", "version", "video_id", "n_frames"]
               + POSTER_ORDER + ["pred_class"]]


def write(df, *parts):
    p = os.path.join(SPLITS, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  {os.path.relpath(p, SPLITS):48s} n={len(df):6d}")


def train_val(pool, frac):
    """Seeded random clip-level validation split."""
    pool = pool.reset_index(drop=True)
    idx = np.arange(len(pool))
    np.random.default_rng(SEED).shuffle(idx)
    n_val = int(round(len(pool) * frac))
    return pool.iloc[idx[n_val:]], pool.iloc[idx[:n_val]]


def split_videos(df, rng):
    """Assign whole videos to test, then val (~15% of the clips each); every speaker keeps one train video."""
    vids = (df.groupby(["celebrity", "video_id"]).size().rename("n").reset_index()
              .sort_values(["celebrity", "video_id"]).reset_index(drop=True))
    vids = vids.iloc[rng.permutation(len(vids))].reset_index(drop=True)
    left = vids.groupby("celebrity").size().to_dict()          # videos per speaker not yet held out
    total = vids.n.sum()
    assign = pd.Series("train", index=vids.index)
    for part, frac in (("test", TEST_FRAC), ("val", VAL_FRAC)):
        got = 0
        for i, v in vids.iterrows():
            if got >= frac * total:
                break
            if assign[i] != "train" or left[v.celebrity] <= 1:
                continue
            assign[i] = part
            left[v.celebrity] -= 1
            got += v.n
    vids["split"] = assign
    return df.merge(vids[["celebrity", "video_id", "split"]], on=["celebrity", "video_id"])


def main():
    full = pd.read_csv(CLIP_LABELS, float_precision="round_trip")
    t = full[full.teacher == "poster"]

    print("all clips per version-language")
    for v in VERSIONS:
        for lang in ("english", PAIRS[v]):
            write(to_train_format(t[(t.version == v) & (t.language == lang)].reset_index(drop=True)),
                  "all", f"poster_{v}_{lang}_all.csv")

    print("pooled")
    for name, pool in (("english", t[t.language == "english"]), ("nonenglish", t[t.language != "english"])):
        tr, va = train_val(pool, POOLED_VAL_FRAC)
        write(to_train_format(tr), "pooled", f"poster_pooled_{name}_train.csv")
        write(to_train_format(va), "pooled", f"poster_pooled_{name}_val.csv")

    print("unknown speakers")
    tr, va = train_val(t[t.version.isin(["v1", "v2"]) & (t.language == "english")], POOLED_VAL_FRAC)
    write(to_train_format(tr), "unknown", "poster_unknown_train.csv")
    write(to_train_format(va), "unknown", "poster_unknown_val.csv")
    v3_people = set(t[t.version == "v3"].speaker_id)
    assert set(tr.version) == {"v1", "v2"}, "unknown-speaker training pool must not contain v3"

    print("known speakers (video-disjoint)")
    # the pooled train+val files hold every clip exactly once; read back exactly as in the original run
    allc = pd.concat([pd.read_csv(os.path.join(SPLITS, "pooled", f"poster_pooled_{n}_{p}.csv"))
                      for n in ("english", "nonenglish") for p in ("train", "val")], ignore_index=True)
    assert allc.wav_path.is_unique and len(allc) == len(t)
    cols = list(allc.columns)
    rng = np.random.default_rng(SEED)
    for v in VERSIONS:
        for lang in ("english", PAIRS[v]):
            s = split_videos(allc[(allc.version == v) & (allc.language.str.lower() == lang)], rng)
            for part in ("train", "val", "test"):
                write(s[s.split == part][cols], "known", f"poster_{v}_{lang}_{part}.csv")
            vid = lambda part: set(s[s.split == part].celebrity + "/" + s[s.split == part].video_id)
            assert not (vid("train") & vid("test")) and not (vid("train") & vid("val")) and not (vid("val") & vid("test"))
            tr_spk = set(s[s.split == "train"].celebrity)
            assert set(s[s.split == "test"].celebrity) <= tr_spk and set(s[s.split == "val"].celebrity) <= tr_spk
    print(f"\nOK  (v3 test speakers: {len(v3_people)}; known splits video-disjoint, all val/test speakers in train)")


if __name__ == "__main__":
    main()
