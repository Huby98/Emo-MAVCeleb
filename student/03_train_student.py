import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

from VGGnet import EmoVGGVoxStudent  # your VGG-M style student

# make torchaudio prefer sox_io if available (quietly ignore failures)
try:
    import torchaudio
    if hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend("sox_io")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F

def load_wav_any(path: str, target_sr: int):
    """Load WAV robustly: torchaudio -> soundfile -> wave; resample if needed."""
    # 1) try torchaudio
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)  # (C, T) float32 in [-1,1]
        ok = True
    except Exception:
        ok = False

    # 2) try soundfile
    if not ok:
        try:
            import soundfile as sf
            x, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
            x = x.T  # (C, T)
            wav = torch.from_numpy(x)
            ok = True
        except Exception:
            ok = False

    # 3) last resort: wave (PCM only)
    if not ok:
        import wave, contextlib
        with contextlib.closing(wave.open(path, "rb")) as w:
            sr = w.getframerate()
            n  = w.getnframes()
            ch = w.getnchannels()
            sampwidth = w.getsampwidth()
            raw = w.readframes(n)
        if sampwidth == 2:
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:
            a = (np.frombuffer(raw, dtype=np.int32).astype(np.float32)) / 2147483648.0
        else:
            raise RuntimeError(f"Unsupported WAV sampwidth={sampwidth} for {path}")
        a = a.reshape(-1, ch).T  # (C, T)
        wav = torch.from_numpy(a)

    # mono
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav = wav.mean(0, keepdim=True)  # (1, T)

    # resample if needed
    if sr != target_sr:
        try:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr)
            up, down = target_sr // g, sr // g
            x = wav.numpy()
            x = resample_poly(x, up, down, axis=1)
            wav = torch.from_numpy(x.copy())
        sr = target_sr

    return wav, sr

EMOS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

# ---------------- Dataset ----------------
class EmoVoxDataset(Dataset):
    def __init__(self, csv_path, sr=16000, dur_s=4.0, train=True):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.train = train
        self.n_samples = int(sr * dur_s)

        # clean/normalize soft labels
        for c in EMOS:
            self.df[c] = (self.df[c].astype(str)
                                        .str.replace(r"[\[\]]", "", regex=True)
                                        .str.replace(",", ".", regex=False))
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df[EMOS] = self.df[EMOS].clip(lower=0)
        s = self.df[EMOS].sum(axis=1)
        self.df = self.df[s > 0].copy()
        self.df[EMOS] = self.df[EMOS].div(self.df[EMOS].sum(axis=1), axis=0)

        # amplitude STFT (paper: 25ms window, 10ms hop) -> (F=512, T≈400)
        win_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=1024, win_length=win_len, hop_length=hop_len,
            window_fn=torch.hamming_window, power=1.0, center=True
        )

    def __len__(self): return len(self.df)

    def _fix_time(self, x, target_T=400):
        T = x.size(-1)
        if T == target_T: return x
        if T < target_T:  return F.pad(x, (0, target_T - T))
        start = 0 if not self.train else int(np.random.randint(0, T - target_T + 1))
        return x[..., start:start+target_T]

    def __getitem__(self, i):
        r = self.df.iloc[i]

        # teacher probs (robust)
        y_np = pd.to_numeric(r[EMOS], errors="coerce").fillna(0.0).to_numpy()
        y_np = np.clip(y_np, 0.0, None)
        s = float(y_np.sum())
        y_np = (np.ones(len(EMOS))/len(EMOS) if (not np.isfinite(s) or s<=0)
                else (y_np / s).astype(np.float32))
        y = torch.from_numpy(y_np)

        wav, _ = load_wav_any(r["wav_path"], self.sr)
        T = wav.shape[1]; n = self.n_samples
        if T < n:
            wav = F.pad(wav, (0, n - T))
        elif T > n:
            start = 0 if not self.train else int(np.random.randint(0, T - n + 1))
            wav = wav[:, start:start+n]

        # spectrogram -> [1, 512, 400]
        spec = self.spec(wav)          # [1, F, T]
        spec = spec[:, :512, :]
        spec = self._fix_time(spec, 400)

        # per-freq CMVN across time
        mean = spec.mean(dim=-1, keepdim=True)
        std  = spec.std(dim=-1, keepdim=True).clamp_min(1e-5)
        spec = (spec - mean) / std

        return spec, y

# ---------------- Tiny fallback (optional) ----------------
class TinyAudioCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1,1)),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x):
        h = self.feat(x).squeeze(-1).squeeze(-1)
        return self.fc(h)

# ---------------- Distillation loss (CE with T, matches paper) ----------------
def kd_ce(student_logits, teacher_probs, T=2.0):
    # teacher soft targets with temperature
    t = torch.clamp(teacher_probs, 1e-8, 1.0)
    t = torch.softmax(torch.log(t)/T, dim=-1)
    # student log-probs with temperature
    s_log = torch.log_softmax(student_logits/T, dim=-1)
    # cross-entropy H(t, s) * T^2
    return (- (t * s_log).sum(dim=-1)).mean() * (T*T)

@torch.no_grad()
def evaluate(model, dl, device, T=2.0):
    model.eval(); n=0; loss_sum=0.0; agree=0
    for spec, y_t in dl:
        spec, y_t = spec.to(device), y_t.to(device)
        logits = model(spec)
        loss = kd_ce(logits, y_t, T=T)
        loss_sum += float(loss.item()) * spec.size(0)
        agree += int((logits.argmax(-1) == y_t.argmax(-1)).sum().item())
        n += spec.size(0)
    return loss_sum/max(n,1), agree/max(n,1)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  default=None)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--arch", choices=["vgg","tiny"], default="vgg")
    # ===== baseline training defaults =====
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr_init", type=float, default=1e-4)
    ap.add_argument("--lr_final", type=float, default=1e-5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur_s", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    tr_ds = EmoVoxDataset(args.train_csv, args.sr, args.dur_s, train=True)
    va_ds = EmoVoxDataset(args.val_csv,   args.sr, args.dur_s, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.num_workers, pin_memory=True,
                   drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True)

    model = (EmoVGGVoxStudent(num_classes=len(EMOS)) if args.arch=="vgg"
             else TinyAudioCNN(n_classes=len(EMOS)))
    model = model.to(device)

    # SGD (paper): momentum=0.9, weight_decay=5e-4
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr_init,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=False)

    # geometric (log) decay from lr_init -> lr_final over epochs
    def lr_at(e):
        if args.epochs <= 1: return args.lr_final
        t = (e-1) / (args.epochs-1)
        return args.lr_init * ((args.lr_final/args.lr_init) ** t)

    best_val = float("inf"); log=[]
    for epoch in range(1, args.epochs+1):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)

        model.train(); seen=0; train_loss=0.0
        for spec, y_t in tqdm(tr_dl, desc=f"Epoch {epoch}/{args.epochs} (lr={opt.param_groups[0]['lr']:.6e})"):
            spec, y_t = spec.to(device), y_t.to(device)
            loss = kd_ce(model(spec), y_t, T=args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += float(loss.item()) * spec.size(0); seen += spec.size(0)
        train_loss /= max(seen,1)

        val_loss, val_agree = evaluate(model, va_dl, device, T=args.temperature)
        log.append({"epoch": epoch, "lr": opt.param_groups[0]['lr'],
                    "train_loss": train_loss, "val_loss": val_loss, "val_top1_agree": val_agree})
        print(f"[{epoch:02d}] lr={opt.param_groups[0]['lr']:.6e}  train={train_loss:.4f}  val={val_loss:.4f}  agree={val_agree:.3f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

    pd.DataFrame(log).to_csv(os.path.join(args.out_dir, "train_log.csv"), index=False)
    print("Best val KL (CE equiv.):", best_val)

    if args.test_csv:
        te_ds = EmoVoxDataset(args.test_csv, args.sr, args.dur_s, train=False)
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device))
        test_loss, test_agree = evaluate(model, te_dl, device, T=args.temperature)
        print(f"[TEST] KL={test_loss:.4f}  Top1={test_agree*100:.1f}%")
        with open(os.path.join(args.out_dir, "test_metrics.txt"), "w") as f:
            f.write(f"KL={test_loss:.6f}\nTop1={test_agree:.6f}\n")

if __name__ == "__main__":
    main()
