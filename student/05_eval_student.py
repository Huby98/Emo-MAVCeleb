import os, argparse, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- import your model ----
from VGGnet import EmoVGGVoxStudent   # VGG-M/VGGVox-style student

EMOS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

# ------- robust audio loading & resampling (same as train) -------
def load_wav_any(path: str, target_sr: int):
    import numpy as np, torch
    # 1) torchaudio
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        ok = True
    except Exception:
        ok = False
    # 2) soundfile
    if not ok:
        try:
            import soundfile as sf
            x, sr = sf.read(path, dtype="float32", always_2d=True)  # (T,C)
            wav = torch.from_numpy(x.T)  # (C,T)
            ok = True
        except Exception:
            ok = False
    # 3) wave (PCM)
    if not ok:
        import wave, contextlib
        with contextlib.closing(wave.open(path, "rb")) as w:
            sr = w.getframerate(); n = w.getnframes(); ch = w.getnchannels(); sw = w.getsampwidth()
            raw = w.readframes(n)
        if   sw == 2: a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 1: a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sw == 4: a =  np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:  raise RuntimeError(f"Unsupported WAV sampwidth={sw} for {path}")
        a = a.reshape(-1, ch).T
        wav = torch.from_numpy(a)

    if wav.dim()==1: wav = wav.unsqueeze(0)
    wav = wav.mean(0, keepdim=True)  # mono

    if sr != target_sr:
        # try torchaudio; fallback to scipy
        try:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sr, target_sr); up = target_sr//g; down = sr//g
            x = wav.numpy()
            x = resample_poly(x, up, down, axis=1)
            wav = torch.from_numpy(x.copy())
        sr = target_sr
    return wav, sr

# ------- dataset (mirrors your training preprocessing) -------
class EmoVoxEvalDS(Dataset):
    def __init__(self, csv_path, sr=16000, dur_s=4.0, deterministic=True):
        self.df = pd.read_csv(csv_path)
        self.sr = sr
        self.n_samples = int(sr*dur_s)
        self.deterministic = deterministic

        # clean soft labels
        for c in EMOS:
            self.df[c] = (self.df[c].astype(str)
                                     .str.replace(r"[\[\]]","",regex=True)
                                     .str.replace(",",".",regex=False))
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        self.df[EMOS] = self.df[EMOS].clip(lower=0)
        s = self.df[EMOS].sum(axis=1)
        self.df = self.df[s>0].copy()
        self.df[EMOS] = self.df[EMOS].div(self.df[EMOS].sum(axis=1), axis=0)

        # spectrogram: 25ms window, 10ms hop, power=1.0 (amplitude)
        import torchaudio, torch
        win_len = int(0.025*sr); hop_len = int(0.010*sr)
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=1024, win_length=win_len, hop_length=hop_len,
            window_fn=torch.hamming_window, power=1.0, center=True
        )

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # target probs
        y = torch.tensor(r[EMOS].to_numpy(dtype=np.float32))
        # audio -> fixed 4s, deterministic crop if requested
        wav, sr = load_wav_any(r["wav_path"], self.sr)
        T = wav.shape[1]; n = self.n_samples
        if T < n:
            pad_left = (n - T)//2
            pad_right = n - T - pad_left
            wav = F.pad(wav, (pad_left, pad_right))
        elif T > n:
            if self.deterministic:
                start = max(0, (T - n)//2)     # center crop
            else:
                start = int(np.random.randint(0, T - n + 1))
            wav = wav[:, start:start+n]
        # spectrogram + CMVN to [1, 512, 400]
        S = self.spec(wav)[:, :512, :400]
        # pad/crop time to 400
        if S.size(-1) < 400:
            S = F.pad(S, (0, 400 - S.size(-1)))
        elif S.size(-1) > 400:
            if self.deterministic:
                st = (S.size(-1) - 400)//2
            else:
                st = int(np.random.randint(0, S.size(-1) - 400 + 1))
            S = S[..., st:st+400]
        mean = S.mean(dim=-1, keepdim=True); std = S.std(dim=-1, keepdim=True).clamp_min(1e-5)
        S = (S - mean)/std
        # language for per-language metrics
        lang = str(r.get("language","unknown")).lower()
        # teacher hard label for per-emotion metrics
        y_top1 = int(np.argmax(r[EMOS].to_numpy(dtype=np.float32)))
        return S, y, lang, y_top1

# ------- losses & metrics -------
def kd_ce(student_logits, teacher_probs, T=2.0):
    t = torch.clamp(teacher_probs, 1e-8, 1.0)
    t = torch.softmax(torch.log(t)/T, dim=-1)
    s_log = torch.log_softmax(student_logits/T, dim=-1)
    return (- (t * s_log).sum(dim=-1)).mean() * (T*T)

@torch.no_grad()
def eval_loop(model, dl, device, T=2.0):
    model.eval()
    tot_n = 0
    sum_loss = 0.0
    agree = 0
    # per-language accumulators
    langs = {}
    # per-emotion (by teacher top1) accumulators
    emos = {i: {"n":0, "agree":0, "loss_sum":0.0} for i in range(len(EMOS))}
    for S, y_t, lang, y_top1 in tqdm(dl, desc="Eval"):
        S = S.to(device); y_t = y_t.to(device)
        logits = model(S)
        loss = kd_ce(logits, y_t, T=T)
        sum_loss += float(loss.item())*S.size(0)
        pred = logits.argmax(-1)
        gold = y_t.argmax(-1)
        agree += int((pred==gold).sum().item())
        # language
        for i in range(S.size(0)):
            L = lang[i]
            if isinstance(L, torch.Tensor): L = str(L)
            langs.setdefault(L, {"n":0,"agree":0,"loss_sum":0.0})
            langs[L]["n"] += 1
            langs[L]["agree"] += int(pred[i].item()==gold[i].item())
            langs[L]["loss_sum"] += float(kd_ce(logits[i:i+1], y_t[i:i+1], T=T).item())
        # emotion (teacher top1)
        for i in range(S.size(0)):
            k = int(y_top1[i].item())
            emos[k]["n"] += 1
            emos[k]["agree"] += int(pred[i].item()==gold[i].item())
            emos[k]["loss_sum"] += float(kd_ce(logits[i:i+1], y_t[i:i+1], T=T).item())
        tot_n += S.size(0)

    out = {
        "overall": {
            "N": tot_n,
            "KL": sum_loss/max(tot_n,1),
            "Top1": agree/max(tot_n,1)
        },
        "by_language": {
            L: {"N":v["n"],
                "KL": v["loss_sum"]/max(v["n"],1),
                "Top1": v["agree"]/max(v["n"],1)} for L,v in langs.items()
        },
        "by_emotion": {
            EMOS[k]: {"N":emos[k]["n"],
                      "KL": emos[k]["loss_sum"]/max(emos[k]["n"],1),
                      "Top1": emos[k]["agree"]/max(emos[k]["n"],1)} for k in emos
        }
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="audio_pseudolabels_all.csv to evaluate")
    ap.add_argument("--ckpt", required=True, help="path to best.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur_s", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EmoVoxEvalDS(args.csv, sr=args.sr, dur_s=args.dur_s, deterministic=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    model = EmoVGGVoxStudent(num_classes=len(EMOS)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    res = eval_loop(model, dl, device, T=args.temperature)

    # print & save
    print("\n=== Overall ===")
    print(f"N={res['overall']['N']}  KL={res['overall']['KL']:.4f}  Top1={100*res['overall']['Top1']:.1f}%")

    print("\n=== By language ===")
    for L,v in res["by_language"].items():
        print(f"{L:8s}  N={v['N']:5d}  KL={v['KL']:.4f}  Top1={100*v['Top1']:.1f}%")

    print("\n=== By teacher top-1 (per emotion) ===")
    for k in EMOS:
        v = res["by_emotion"][k]
        print(f"{k:9s}  N={v['N']:5d}  KL={v['KL']:.4f}  Top1={100*v['Top1']:.1f}%")

    # CSVs for Overleaf
    pd.DataFrame([res["overall"]]).to_csv(os.path.join(args.out_dir,"overall.csv"), index=False)
    pd.DataFrame([
        {"language":L, **v} for L,v in res["by_language"].items()
    ]).to_csv(os.path.join(args.out_dir,"by_language.csv"), index=False)
    pd.DataFrame([
        {"emotion":k, **v} for k,v in res["by_emotion"].items()
    ]).to_csv(os.path.join(args.out_dir,"by_emotion.csv"), index=False)

if __name__ == "__main__":
    main()
