import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

from VGGnet import EmoVGGVoxStudent  # VGG-M style student
# SSL student (WavLM / wav2vec2 / XLS-R). Import is safe without transformers
# installed; it only errors if you actually instantiate --arch ssl.
from SSLnet import SSLEmotionStudent

try:
    import torchaudio
    if hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass

# ---- optional lossless int16 waveform cache (pure I/O speed-up, not needed to reproduce)
# The dataset sits on a spinning HDD and every epoch re-reads it as thousands of small
# random reads, which pinned throughput at ~15 clips/s cold. All source wavs are
# 16 kHz mono PCM_16, so load_wav_any never resamples and int16/32768 reproduces the
# float32 tensor EXACTLY. Enabling the cache is therefore a pure I/O change.
_WAV_CACHE = None


def enable_wav_cache(cache_dir):
    """Memory-map the waveform blob and its index. Returns the number of cached clips.

    Entries are int16 (bit-exact for the 16 kHz mono PCM_16 majority) or float32
    (the 0.31% that load_wav_any resamples). The index records which.
    """
    global _WAV_CACHE
    if not cache_dir:
        return 0
    idx = pd.read_csv(os.path.join(cache_dir, "index.csv"))
    blob = np.memmap(os.path.join(cache_dir, "waves.bin"), dtype=np.uint8, mode="r")
    _WAV_CACHE = {"blob": blob,
                  "map": {p: (int(o), int(n), d) for p, o, n, d in
                          zip(idx.wav_path, idx.byte_offset, idx.n_samples, idx.dtype)}}
    print(f"wav cache: {len(_WAV_CACHE['map'])} clips from {cache_dir}")
    return len(_WAV_CACHE["map"])


def load_wav_any(path: str, target_sr: int):
    if _WAV_CACHE is not None and target_sr == 16000:
        hit = _WAV_CACHE["map"].get(path)
        if hit is not None:
            o, n, dt = hit
            if dt == "int16":
                raw = _WAV_CACHE["blob"][o:o + n * 2].view(np.int16)
                a = raw.astype(np.float32) / 32768.0
            else:
                a = np.array(_WAV_CACHE["blob"][o:o + n * 4].view(np.float32))
            return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0), target_sr
    try:
        import torchaudio
        wav, sr = torchaudio.load(path); ok = True
    except Exception:
        ok = False
    if not ok:
        try:
            import soundfile as sf
            x, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(x.T); ok = True
        except Exception:
            ok = False
    if not ok:
        import wave, contextlib
        with contextlib.closing(wave.open(path, "rb")) as w:
            sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels(); sampwidth = w.getsampwidth()
            raw = w.readframes(n)
        if sampwidth == 2: a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1: a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4: a = (np.frombuffer(raw, dtype=np.int32).astype(np.float32)) / 2147483648.0
        else: raise RuntimeError(f"Unsupported WAV sampwidth={sampwidth} for {path}")
        a = a.reshape(-1, ch).T
        wav = torch.from_numpy(a)
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        try:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr); up, down = target_sr // g, sr // g
            wav = torch.from_numpy(resample_poly(wav.numpy(), up, down, axis=1).copy())
        sr = target_sr
    return wav, sr

EMOS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

# wav_path in the published split CSVs is relative to the MAV-Celeb root, i.e. the folder that
# holds v1/, v2/, v3/ (each with voices/). Absolute paths are used as they are.
DATA_ROOT = os.environ.get("MAVCELEB_DATA_ROOT",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mavceleb"))


def set_seed(seed: int):
    """Seed every RNG that affects training: weight init, shuffling, random crop."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Give each DataLoader worker a deterministic, distinct numpy seed.

    Workers are forked/spawned after set_seed, so without this they would either
    share one numpy stream or get OS-random ones -- and the random crop in
    EmoVoxDataset.__getitem__ draws from numpy.
    """
    import random
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)

class EmoVoxDataset(Dataset):
    def __init__(self, csv_path, sr=16000, dur_s=4.0, train=True, feat="spec"):
        # feat="spec" -> (1,512,400) magnitude spectrogram (VGG/tiny path, unchanged)
        # feat="wav"  -> (S,) raw normalized waveform (SSL path)
        self.df = pd.read_csv(csv_path)
        self.df["wav_path"] = [p if os.path.isabs(p) else os.path.join(DATA_ROOT, p)
                               for p in self.df["wav_path"]]
        self.sr = sr; self.train = train; self.n_samples = int(sr * dur_s)
        self.feat = feat
        for c in EMOS:
            self.df[c] = (self.df[c].astype(str)
                          .str.replace(r"[\[\]]", "", regex=True)
                          .str.replace(",", ".", regex=False))
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df[EMOS] = self.df[EMOS].clip(lower=0)
        s = self.df[EMOS].sum(axis=1)
        self.df = self.df[s > 0].copy()
        self.df[EMOS] = self.df[EMOS].div(self.df[EMOS].sum(axis=1), axis=0)
        win_len = int(0.025 * sr); hop_len = int(0.010 * sr)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=1024, win_length=win_len, hop_length=hop_len,
            window_fn=torch.hamming_window, power=1.0, center=True)

    def __len__(self): return len(self.df)

    def _fix_time(self, x, target_T=400):
        T = x.size(-1)
        if T == target_T: return x
        if T < target_T:  return F.pad(x, (0, target_T - T))
        start = 0 if not self.train else int(np.random.randint(0, T - target_T + 1))
        return x[..., start:start+target_T]

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y_np = pd.to_numeric(r[EMOS], errors="coerce").fillna(0.0).to_numpy()
        y_np = np.clip(y_np, 0.0, None); s = float(y_np.sum())
        y_np = (np.ones(len(EMOS))/len(EMOS) if (not np.isfinite(s) or s<=0)
                else (y_np / s).astype(np.float32))
        y = torch.from_numpy(y_np)
        y_cls = int(np.argmax(y_np))          # dominant class index (for class weighting)
        wav, _ = load_wav_any(r["wav_path"], self.sr)
        T = wav.shape[1]; n = self.n_samples
        if T < n: wav = F.pad(wav, (0, n - T))
        elif T > n:
            start = 0 if not self.train else int(np.random.randint(0, T - n + 1))
            wav = wav[:, start:start+n]
        if self.feat == "wav":
            # raw waveform for SSL encoders: per-utterance zero-mean/unit-var,
            # exactly what the HF feature extractor would apply.
            w = wav.squeeze(0).float()
            w = (w - w.mean()) / (w.std() + 1e-6)
            return w, y, y_cls
        spec = self.spec(wav)[:, :512, :]
        spec = self._fix_time(spec, 400)
        mean = spec.mean(dim=-1, keepdim=True); std = spec.std(dim=-1, keepdim=True).clamp_min(1e-5)
        spec = (spec - mean) / std
        return spec, y, y_cls

class TinyAudioCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1,1)))
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        return self.fc(self.feat(x).squeeze(-1).squeeze(-1))

# ---- distillation loss, now optionally per-example weighted ----
def kd_ce(student_logits, teacher_probs, T=2.0, weights=None):
    t = torch.clamp(teacher_probs, 1e-8, 1.0)
    t = torch.softmax(torch.log(t)/T, dim=-1)
    s_log = torch.log_softmax(student_logits/T, dim=-1)
    per_example = -(t * s_log).sum(dim=-1)            # (B,)
    if weights is not None:
        per_example = per_example * weights           # weight each example
        return (per_example.sum() / weights.sum()) * (T*T)
    return per_example.mean() * (T*T)

@torch.no_grad()
def evaluate(model, dl, device, T=2.0):
    model.eval(); n=0; loss_sum=0.0; agree=0
    for spec, y_t, _ in dl:
        spec, y_t = spec.to(device), y_t.to(device)
        logits = model(spec)
        loss = kd_ce(logits, y_t, T=T)
        loss_sum += float(loss.item()) * spec.size(0)
        agree += int((logits.argmax(-1) == y_t.argmax(-1)).sum().item())
        n += spec.size(0)
    return loss_sum/max(n,1), agree/max(n,1)

def compute_class_weights(csv_path):
    """Inverse-frequency weights over the teacher's dominant class."""
    df = pd.read_csv(csv_path)
    for c in EMOS:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",",".",regex=False), errors="coerce")
    df = df.dropna(subset=EMOS)
    dom = df[EMOS].to_numpy().argmax(axis=1)
    counts = np.bincount(dom, minlength=len(EMOS)).astype(np.float64)
    counts = np.clip(counts, 1, None)                 # avoid div by zero
    w = counts.sum() / (len(EMOS) * counts)           # inverse-frequency, normalized
    w = w / w.mean()                                  # mean weight = 1
    print("Class weights (dominant-class inverse freq):")
    for e, c, ww in zip(EMOS, counts, w):
        print(f"  {e:10s} count={int(c):6d}  weight={ww:.3f}")
    return torch.tensor(w, dtype=torch.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--arch", choices=["vgg","tiny","ssl"], default="vgg")
    ap.add_argument("--ssl_ckpt", default="microsoft/wavlm-base-plus",
                    help="HF checkpoint for --arch ssl (e.g. facebook/wav2vec2-xls-r-300m)")
    ap.add_argument("--ft_encoder", action="store_true",
                    help="fine-tune the SSL encoder (default: frozen)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr_init", type=float, default=1e-3)
    ap.add_argument("--lr_final", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur_s", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="0 is MUCH faster here. Measured on v1-en (4,836 clips), "
                         "data-loading only: nw=0 10.1s/epoch, nw=2 37.3s, nw=4 47.2s, "
                         "nw=8 85s. Each sample is a (1,512,400) float32 spectrogram = "
                         "819 KB, so worker->main IPC on Windows costs far more than the "
                         "decode it parallelises (single-thread __getitem__ alone runs at "
                         "509 clips/s). The old default of 4 made every run ~4.7x slower. "
                         "NOTE: changing this changes the random-crop RNG stream, so keep "
                         "it fixed across runs you intend to compare.")
    ap.add_argument("--class_weights", action="store_true",
                    help="apply inverse-frequency class weighting to the distillation loss")
    ap.add_argument("--seed", type=int, default=1337,
                    help="seeds torch, numpy and the DataLoader shuffle/worker streams. "
                         "Matches the split seed used by make_splits.py. Pass the SAME "
                         "seed to both teachers so the comparison isolates the labels.")
    ap.add_argument("--wav_cache", default=None,
                    help="optional lossless int16 SSD cache of the waveforms (index.csv + "
                         "waves.bin, keyed by wav_path). Only an I/O speed-up; leave unset.")
    ap.add_argument("--deterministic", action="store_true",
                    help="bitwise-reproducible runs: disables cudnn.benchmark and forces "
                         "deterministic kernels. Without it, two same-seed runs agree only "
                         "to ~2e-3 in weights. Measured cost on this box: NONE (247s vs "
                         "247-281s for 3 epochs) -- training is wav-decode bound, not "
                         "kernel bound. Use it for every run whose numbers get quoted.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- reproducibility ------------------------------------------------
    # Without set_seed the weight init (VGGnet kaiming), the DataLoader shuffle and
    # the random 4s crop were all unseeded, so "identical seed for both teachers"
    # was unverifiable.
    #
    # Seeding alone does NOT give bitwise reproducibility on CUDA: with
    # cudnn.benchmark the kernel choice is timed at runtime and can differ between
    # runs, and some kernels accumulate with atomics. Measured on this box:
    #   same seed, benchmark on  -> maxabsdiff 2.1e-3  (NOT identical)
    #   different seed           -> maxabsdiff 1.2e+1  (seed dominates by ~4 OOM)
    #   same seed, --deterministic -> maxabsdiff 0.0   (bitwise identical)
    # and --deterministic measured FREE here (247s vs 247-281s for 3 epochs),
    # because throughput is bound by wav decoding in the DataLoader, not by cuDNN.
    # So there is no reason not to use it for the deliverable runs.
    enable_wav_cache(args.wav_cache)
    set_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    print(f"Seed: {args.seed}  deterministic={args.deterministic}")

    cls_w = compute_class_weights(args.train_csv).to(device) if args.class_weights else None

    feat = "wav" if args.arch == "ssl" else "spec"
    tr_ds = EmoVoxDataset(args.train_csv, args.sr, args.dur_s, train=True,  feat=feat)
    va_ds = EmoVoxDataset(args.val_csv,   args.sr, args.dur_s, train=False, feat=feat)
    g = torch.Generator(); g.manual_seed(args.seed)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                       worker_init_fn=seed_worker, generator=g)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       worker_init_fn=seed_worker)

    if args.arch == "ssl":
        model = SSLEmotionStudent(ckpt=args.ssl_ckpt, num_classes=len(EMOS),
                                  freeze_encoder=not args.ft_encoder).to(device)
    elif args.arch == "vgg":
        model = EmoVGGVoxStudent(num_classes=len(EMOS)).to(device)
    else:
        model = TinyAudioCNN(n_classes=len(EMOS)).to(device)

    if args.arch == "ssl":
        # AdamW over the trainable params only (head + layer weights when frozen).
        # SGD-50ep is the from-scratch-CNN recipe; a pretrained encoder needs its
        # own fair recipe, otherwise the swap is rigged to look like "no change".
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr_init, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_init,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    def lr_at(e):
        if args.epochs <= 1: return args.lr_final
        t = (e-1)/(args.epochs-1)
        return args.lr_init * ((args.lr_final/args.lr_init) ** t)

    best_val = float("inf"); log=[]
    for epoch in range(1, args.epochs+1):
        for g in opt.param_groups: g["lr"] = lr_at(epoch)
        model.train(); seen=0; train_loss=0.0
        for spec, y_t, y_cls in tqdm(tr_dl, desc=f"Epoch {epoch}/{args.epochs} (lr={opt.param_groups[0]['lr']:.2e})"):
            spec, y_t = spec.to(device), y_t.to(device)
            w = cls_w[y_cls.to(device)] if cls_w is not None else None
            loss = kd_ce(model(spec), y_t, T=args.temperature, weights=w)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += float(loss.item()) * spec.size(0); seen += spec.size(0)
        train_loss /= max(seen,1)
        val_loss, val_agree = evaluate(model, va_dl, device, T=args.temperature)
        log.append({"epoch":epoch, "lr":opt.param_groups[0]['lr'],
                    "train_loss":train_loss, "val_loss":val_loss, "val_top1_agree":val_agree})
        print(f"[{epoch:02d}] lr={opt.param_groups[0]['lr']:.2e}  train={train_loss:.4f}  val={val_loss:.4f}  agree={val_agree:.3f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "train_log.csv"), index=False)
    # Dump the full config next to the checkpoint so a run's provenance is
    # recoverable from the run dir alone -- Phase 0 found several checkpoints
    # whose training data and teacher could not be determined after the fact.
    import json as _json
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        _json.dump({**vars(args), "best_val_loss": best_val}, f, indent=2)
    print("Best val loss:", best_val)

if __name__ == "__main__":
    main()
