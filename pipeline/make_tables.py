import os, json, argparse
import pandas as pd

from paths import REPO, RUNS, RESULTS, CLIP_LABELS, PAIRS, POSTER_ORDER

STUDENTS = [("vgg", "VGGVox"), ("wavlm", "WavLM")]
# WavLM trained on V3-German alone collapsed with seed 1337 and was retrained with seed 1338
RERUNS = {"wavlm_v3_german": "wavlm_v3_german_seed1338"}
NOTE = (r" WavLM trained on V3-German alone collapsed to a single class with the default seed "
        r"and was retrained once with the next seed.")
cells = []


def score(evals, run, tag, **meta):
    run = RERUNS.get(run, run)
    path = os.path.join(evals, run, f"eval_{run}__{tag}.json")
    v = None
    if os.path.exists(path):
        v = round(json.load(open(path))["macro_f1"] * 100, 1)
    cells.append(dict(**meta, value=v, source=os.path.relpath(path, REPO).replace(os.sep, "/")))
    return v


def f(v, bold=False):
    if v is None:
        return r"\tbd"
    s = f"{v:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def fd(a, b):
    if a is None or b is None:
        return r"\tbd"
    d = round(b - a, 1)
    return f"+{d:.1f}" if d > 0 else f"{d:.1f}"


def know_table(evals):
    rows = []
    for key, label in STUDENTS:
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}}}} \\")
        for train, rlabel in ((None, "English"), ("pair", "Other language")):
            vals = []
            for v, pair in PAIRS.items():
                tl = pair if train else "english"
                for test in ("english", pair):
                    vals.append(score(evals, f"{key}_{v}_{tl}", f"{v}_{test}_test", table="known",
                                      student=key, train=f"{v}_{tl}", test=f"{v}_{test}_test"))
            rows.append(f"{rlabel}\n    & " + " & ".join(f(x) for x in vals) + r" \\")
        if key == "vgg":
            rows.append(r"\midrule")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\footnotesize
\centering
\caption{{Emotion recognition results (Macro-F1) across languages for a
\textit{{known}} set of speakers under different training and test configurations
of \ourdataset{{}}. Test sets contain held-out videos of speakers seen during training.
WavLM is initialised from self-supervised speech pretraining, whereas VGGVox is
trained from scratch.{NOTE}}}
\label{{tab:know_set_of_speaker}}
\setlength{{\tabcolsep}}{{7pt}}
\renewcommand{{\arraystretch}}{{1.1}}
\resizebox{{\columnwidth}}{{!}}{{
\begin{{tabular}}{{lcc|cc|cc}}
\toprule
& \multicolumn{{2}}{{c|}}{{\textbf{{V1 -- English/Urdu}}}}
& \multicolumn{{2}}{{c|}}{{\textbf{{V2 -- English/Hindi}}}}
& \multicolumn{{2}}{{c}}{{\textbf{{V3 -- English/German}}}} \\

\cmidrule(lr){{2-3}}
\cmidrule(lr){{4-5}}
\cmidrule(lr){{6-7}}

\textbf{{Training}}
& \textbf{{Eng.}} & \textbf{{Urdu}}
& \textbf{{Eng.}} & \textbf{{Hindi}}
& \textbf{{Eng.}} & \textbf{{German}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
}}
\end{{table}}
"""


def pooled_block(evals, direction):
    lines = []
    for key, label in STUDENTS:
        ind, poo = [], []
        for v, pair in PAIRS.items():
            train, test, pooled = (("english", pair, "pooled_english") if direction == "en2x"
                                   else (pair, "english", "pooled_nonenglish"))
            meta = dict(table="pooled", student=key, test=f"{v}_{test}_all")
            ind.append(score(evals, f"{key}_{v}_{train}", f"{v}_{test}_all", train=f"{v}_{train}", **meta))
            poo.append(score(evals, f"{key}_{pooled}", f"{v}_{test}_all", train=pooled, **meta))
        names = ("Individual EN", "Pooled EN") if direction == "en2x" else ("Individual", "Pooled UR+HI+DE")
        best = lambda a, b: a is not None and b is not None and a > b
        lines += [rf"\multicolumn{{4}}{{l}}{{\textit{{{label}}}}} \\",
                  f"{names[0]}\n    & " + " & ".join(f(a, best(a, b)) for a, b in zip(ind, poo)) + r" \\",
                  f"{names[1]}\n    & " + " & ".join(f(b, best(b, a)) for a, b in zip(ind, poo)) + r" \\",
                  r"\midrule",
                  r"$\Delta$" + "\n    & " + " & ".join(fd(a, b) for a, b in zip(ind, poo)) + r" \\"]
        if key == "vgg":
            lines.append(r"\midrule")
    return "\n".join(lines)


def pooled_table(evals):
    return rf"""\begin{{table}}[t]
\footnotesize
\centering
\caption{{Emotion recognition results (Macro-F1) across individual and pooled training for cross-lingual emotion transfer across a known set of speakers. All evaluations are performed on all clips of an unheard language. WavLM is initialised from self-supervised speech pretraining, whereas VGGVox is trained from scratch.{NOTE}}}
\label{{tab:pooled_transfer}}
\setlength{{\tabcolsep}}{{7pt}}
\renewcommand{{\arraystretch}}{{1.1}}

\begin{{tabular}}{{lccc}}
\toprule
& \multicolumn{{3}}{{c}}{{\textbf{{English $\rightarrow$ Paired Language}}}} \\
\cmidrule(lr){{2-4}}
\textbf{{Training}} &
\textbf{{V1--UR}} &
\textbf{{V2--HI}} &
\textbf{{V3--DE}} \\
\midrule
{pooled_block(evals, "en2x")}
\midrule
\midrule

& \multicolumn{{3}}{{c}}{{\textbf{{Paired Language $\rightarrow$ English}}}} \\
\cmidrule(lr){{2-4}}
\textbf{{Training}} &
\textbf{{V1--EN}} &
\textbf{{V2--EN}} &
\textbf{{V3--EN}} \\
\midrule
{pooled_block(evals, "x2en")}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def unknown_table(evals):
    rows = []
    for key, label in STUDENTS:
        r = {"student": label, "train": "v1+v2 English"}
        for lang in ("english", "german"):
            r[f"v3_{lang}"] = score(evals, f"{key}_unknown", f"v3_{lang}_all", table="unknown",
                                    student=key, train="unknown", test=f"v3_{lang}_all")
        rows.append(r)
    return pd.DataFrame(rows)


def class_distribution():
    df = pd.read_csv(CLIP_LABELS)
    out = []
    for (t, ver, lang), g in df.groupby(["teacher", "version", "language"]):
        c = g["argmax_label"].value_counts(normalize=True) * 100
        rec = {"teacher": t, "version": ver, "language": lang, "n_clips": len(g)}
        rec.update({e: round(float(c.get(e, 0.0)), 2) for e in POSTER_ORDER})
        out.append(rec)
    return pd.DataFrame(out).sort_values(["version", "language", "teacher"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=RUNS, help="folder with <run>/eval_*.json (default: runs/)")
    args = ap.parse_args()
    out = os.path.join(RESULTS, "tables")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "tables_known_and_pooled.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(know_table(args.evals) + "\n\n" + pooled_table(args.evals))
    unknown = unknown_table(args.evals)
    unknown.to_csv(os.path.join(out, "unknown_speakers.csv"), index=False)
    class_distribution().to_csv(os.path.join(out, "clip_label_class_distribution.csv"), index=False)
    c = pd.DataFrame(cells).drop_duplicates()
    c.to_csv(os.path.join(out, "cells.csv"), index=False)

    print(f"{c.value.notna().sum()}/{len(c)} cells filled -> {out}")
    print(unknown.to_string(index=False))


if __name__ == "__main__":
    main()
