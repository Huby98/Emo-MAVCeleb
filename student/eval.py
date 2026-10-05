"""
eval.py — score a trained student checkpoint against any test CSV.

Drop-in for train_student.py: reuses EmoVoxDataset, EMOS, and the model
classes from that file, so the checkpoint loads exactly as it was saved
(torch.save(model.state_dict(), "best.pt")).

Outputs the metrics that actually decide the thesis (not just raw agree):
  - top-1 agreement (matches train_student's 'agree')
  - balanced accuracy (macro-averaged recall) -- robust to neutral imbalance
  - macro-F1                                   -- robust to neutral imbalance
  - per-class recall + support
  - full confusion matrix (rows = teacher dominant class, cols = student pred)
  - neutral prediction rate  <-- the core thesis metric
  - Cohen's kappa

Usage:
  python student/eval.py --ckpt runs/vgg_v1_english/best.pt \
                         --csv  data/splits/known/poster_v1_english_test.csv \
                         --arch vgg --out_json runs/vgg_v1_english/eval.json

  # SSL student (must pass the SAME --ssl_ckpt used in training):
  python student/eval.py --ckpt runs/wavlm_v1_english/best.pt --csv ... \
                         --arch ssl --ssl_ckpt microsoft/wavlm-base-plus

Writes a JSON with every metric; pipeline/make_tables.py reads macro_f1 from it.
"""

import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader

# Reuse the EXACT definitions from training so there is no interface drift.
from train_student import EmoVoxDataset, EMOS, TinyAudioCNN, enable_wav_cache
from VGGnet import EmoVGGVoxStudent
from SSLnet import SSLEmotionStudent

NEUTRAL_IDX = EMOS.index("neutral")  # = 0, but compute it rather than hardcode


def build_model(arch, device, ssl_ckpt="microsoft/wavlm-base-plus"):
    if arch == "vgg":
        model = EmoVGGVoxStudent(num_classes=len(EMOS))
    elif arch == "ssl":
        # eval is always frozen-encoder shape; the saved state_dict restores weights
        model = SSLEmotionStudent(ckpt=ssl_ckpt, num_classes=len(EMOS), freeze_encoder=True)
    else:
        model = TinyAudioCNN(n_classes=len(EMOS))
    return model.to(device)


@torch.no_grad()
def collect_predictions(model, dl, device):
    """Return (y_true, y_pred) as int arrays of dominant-class indices."""
    model.eval()
    y_true, y_pred = [], []
    for spec, y_t, _ in dl:
        spec = spec.to(device)
        logits = model(spec)
        y_pred.append(logits.argmax(-1).cpu().numpy())
        y_true.append(y_t.argmax(-1).cpu().numpy())  # teacher dominant class
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics_from_cm(cm):
    k = cm.shape[0]
    support = cm.sum(axis=1)                       # true count per class
    pred_count = cm.sum(axis=0)                    # predicted count per class
    diag = np.diag(cm).astype(np.float64)
    total = cm.sum()

    # per-class recall / precision (NaN where undefined -> reported as None)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, diag / np.maximum(support, 1), np.nan)
        precision = np.where(pred_count > 0, diag / np.maximum(pred_count, 1), np.nan)

    # --- macro-F1 FIX ---------------------------------------------------
    # A class the student never gets right must contribute F1 = 0 to the
    # macro-average, NOT be silently dropped. The previous version set F1
    # to NaN for such classes and averaged with np.nanmean, which skipped
    # them -- so under total neutral collapse the "macro-F1" collapsed to
    # neutral's OWN F1 (~0.92) and REWARDED the collapse. We now treat
    # undefined precision/recall as 0 before forming F1, so zero-recall
    # classes correctly count as 0.
    prec0 = np.nan_to_num(precision, nan=0.0)
    rec0 = np.nan_to_num(recall, nan=0.0)
    denom = prec0 + rec0
    f1 = np.where(denom > 0, 2 * prec0 * rec0 / (denom + 1e-12), 0.0)

    present = support > 0                          # classes the teacher actually used
    # balanced accuracy legitimately averages over present classes only: recall is
    # undefined without support (this matches sklearn.balanced_accuracy_score).
    balanced_acc = np.nanmean(recall[present]) if present.any() else float("nan")

    # --- macro-F1 FIX, PART 2 -------------------------------------------
    # The denominator must be the FIXED label set, not `present`. Averaging over
    # `present` only made the denominator depend on the teacher being evaluated:
    # a teacher that collapses to neutral drives whole classes out of the gold
    # labels (SEnet leaves only ~5 of 8), so its macro-average is divided by ~5
    # while POSTER's is divided by 8. That handed SEnet a spurious +0.05..+0.16
    # and made collapse look GOOD -- the same failure as PART 1 above, one level
    # up: collapse must never shrink the thing we divide by. Zero-support classes
    # contribute F1 = 0, which is what sklearn's
    # f1_score(..., labels=range(8), average="macro", zero_division=0) does, and
    # what evaluate_student.py has always done.
    macro_f1 = float(np.mean(f1))
    top1 = diag.sum() / max(total, 1)

    # Cohen's kappa
    po = top1
    pe = (support.astype(np.float64) * pred_count.astype(np.float64)).sum() / max(total * total, 1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")

    neutral_pred_rate = pred_count[NEUTRAL_IDX] / max(total, 1)

    return {
        "n": int(total),
        "top1_agree": float(top1),
        "balanced_acc": float(balanced_acc),
        "macro_f1": float(macro_f1),
        # how many of the 8 classes the teacher's gold labels actually cover.
        # A collapsed teacher covers fewer; report it so the reader can see that
        # macro_f1 is still averaged over all 8 regardless.
        "n_classes_in_gold": int(present.sum()),
        "cohens_kappa": float(kappa),
        "neutral_pred_rate": float(neutral_pred_rate),
        "per_class": {
            EMOS[i]: {
                "support": int(support[i]),
                "predicted": int(pred_count[i]),
                "recall": (None if np.isnan(recall[i]) else float(recall[i])),
                "precision": (None if np.isnan(precision[i]) else float(precision[i])),
                # F1 is None only for classes with no support in this test set
                # (truly absent); present-but-collapsed classes report 0.0.
                "f1": (None if support[i] == 0 else float(f1[i])),
            } for i in range(k)
        },
        "confusion_matrix": cm.tolist(),
    }


def pretty_print(tag, m):
    print(f"\n=== {tag} ===")
    print(f"  n={m['n']}")
    print(f"  top-1 agree     : {m['top1_agree']:.3f}")
    print(f"  balanced acc    : {m['balanced_acc']:.3f}   <- macro recall, imbalance-robust")
    print(f"  macro-F1        : {m['macro_f1']:.3f}")
    print(f"  Cohen's kappa   : {m['cohens_kappa']:.3f}")
    print(f"  NEUTRAL pred rate: {m['neutral_pred_rate']:.3f}   <- core thesis metric")
    print("  per-class recall (support):")
    for e in EMOS:
        pc = m["per_class"][e]
        r = "  n/a" if pc["recall"] is None else f"{pc['recall']:.3f}"
        print(f"    {e:10s} recall={r}  support={pc['support']:5d}  predicted={pc['predicted']:5d}")
    print("  confusion matrix (rows=teacher, cols=student pred); order:")
    print("   ", " ".join(f"{e[:4]:>5s}" for e in EMOS))
    cm = np.array(m["confusion_matrix"])
    for i, e in enumerate(EMOS):
        print(f"    {e[:4]:>4s} " + " ".join(f"{v:5d}" for v in cm[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--arch", choices=["vgg", "tiny", "ssl"], default="vgg")
    ap.add_argument("--ssl_ckpt", default="microsoft/wavlm-base-plus",
                    help="must match the checkpoint used to train an --arch ssl model")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur_s", type=float, default=4.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--tag", default=None, help="label for this scoring (e.g. poster_v1_en_known)")
    ap.add_argument("--wav_cache", default=None,
                    help="lossless int16 SSD waveform cache; see train_student.enable_wav_cache")
    args = ap.parse_args()

    enable_wav_cache(args.wav_cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or os.path.basename(args.csv)

    feat = "wav" if args.arch == "ssl" else "spec"
    ds = EmoVoxDataset(args.csv, args.sr, args.dur_s, train=False, feat=feat)  # train=False -> deterministic center crop
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.arch, device, args.ssl_ckpt)
    state = torch.load(args.ckpt, map_location=device)
    # tolerate either a raw state_dict (how train_student saves) or a wrapped dict
    if isinstance(state, dict) and "state_dict" in state and not any(k.startswith(("feat", "fc", "features")) for k in state):
        state = state["state_dict"]
    model.load_state_dict(state)

    y_true, y_pred = collect_predictions(model, dl, device)
    cm = confusion(y_true, y_pred, len(EMOS))
    m = metrics_from_cm(cm)
    m["tag"] = tag
    m["ckpt"] = os.path.abspath(args.ckpt)
    m["csv"] = os.path.abspath(args.csv)

    pretty_print(tag, m)

    out = args.out_json
    if out is None:
        ck_dir = os.path.dirname(os.path.abspath(args.ckpt))
        out = os.path.join(ck_dir, f"eval_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
