"""
Step 3 -- train and evaluate every student model reported in the paper (POSTER V2 labels).

    python pipeline/run_experiments.py                      # everything, VGGVox first
    python pipeline/run_experiments.py --students vgg       # only VGGVox
    python pipeline/run_experiments.py --only vgg_v1_urdu   # a single run

Runs (per student, VGGVox and WavLM):
  known    {student}_{v}_{lang}      6 per-version models, data/splits/known
  pooled   {student}_pooled_english  v1+v2+v3 English          -> all Urdu / Hindi / German clips
           {student}_pooled_nonenglish  Urdu+Hindi+German       -> all English clips
  unknown  {student}_unknown         v1+v2 English              -> all v3 English / German clips

Each run writes runs/<name>/best.pt (lowest validation loss), train_log.csv, run_config.json and one
eval_<name>__<test>.json per test set. Finished trainings and existing eval JSONs are skipped, so the
script can be restarted after an interruption. One GPU job at a time.
"""
import os, sys, argparse, subprocess

from paths import REPO, SPLITS, RUNS, STUDENT, PAIRS

RECIPES = {
    # VGGVox: trained from scratch, SGD (momentum 0.9), lr 1e-4 -> 1e-5 (log-linear)
    "vgg":   ["--arch", "vgg", "--epochs", "50", "--lr_init", "1e-4", "--lr_final", "1e-5",
              "--seed", "1337", "--num_workers", "0", "--class_weights", "--deterministic"],
    # WavLM Base+: frozen encoder, only layer weights + head train; AdamW, lr 1e-3 -> 1e-4
    "wavlm": ["--arch", "ssl", "--ssl_ckpt", "microsoft/wavlm-base-plus", "--epochs", "50",
              "--lr_init", "0.001", "--lr_final", "0.0001", "--batch_size", "32",
              "--seed", "1337", "--num_workers", "0", "--class_weights", "--deterministic"],
}
EVAL_ARCH = {"vgg": ["--arch", "vgg"], "wavlm": ["--arch", "ssl", "--ssl_ckpt", "microsoft/wavlm-base-plus"]}
# WavLM trained on V3-German alone collapsed to a single class with seed 1337 (best epoch 1/50);
# as stated in the paper it was retrained once with the next seed, and that run is reported.
EXTRA_SEEDS = {"wavlm_v3_german": 1338}


def all_clips(v, lang):
    return os.path.join(SPLITS, "all", f"poster_{v}_{lang}_all.csv")


def runs(student):
    """(name, train_csv, val_csv, [(test_tag, test_csv)])"""
    out = []
    for v, pair in PAIRS.items():
        for lang, unheard in (("english", pair), (pair, "english")):
            k = lambda part, l=lang: os.path.join(SPLITS, "known", f"poster_{v}_{l}_{part}.csv")
            tests = [(f"{v}_{l}_test", os.path.join(SPLITS, "known", f"poster_{v}_{l}_test.csv"))
                     for l in ("english", pair)]
            tests.append((f"{v}_{unheard}_all", all_clips(v, unheard)))
            out.append((f"{student}_{v}_{lang}", k("train"), k("val"), tests))
    for name, test_lang in (("english", None), ("nonenglish", "english")):
        p = lambda part, n=name: os.path.join(SPLITS, "pooled", f"poster_pooled_{n}_{part}.csv")
        tests = [(f"{v}_{test_lang or pair}_all", all_clips(v, test_lang or pair)) for v, pair in PAIRS.items()]
        out.append((f"{student}_pooled_{name}", p("train"), p("val"), tests))
    u = lambda part: os.path.join(SPLITS, "unknown", f"poster_unknown_{part}.csv")
    out.append((f"{student}_unknown", u("train"), u("val"),
                [(f"v3_{l}_all", all_clips("v3", l)) for l in ("english", "german")]))
    return out


def run(cmd, log_path):
    print("$", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        sys.exit(f"failed (exit {rc}), see {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", nargs="+", choices=list(RECIPES), default=list(RECIPES))
    ap.add_argument("--only", nargs="+", help="run names to run (default: all)")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    py = sys.executable

    queue = []
    for student in args.students:
        for name, train_csv, val_csv, tests in runs(student):
            queue.append((student, name, train_csv, val_csv, tests, None))
            if name in EXTRA_SEEDS:
                queue.append((student, f"{name}_seed{EXTRA_SEEDS[name]}", train_csv, val_csv, tests, EXTRA_SEEDS[name]))

    for student, name, train_csv, val_csv, tests, seed in queue:
        if args.only and name not in args.only:
            continue
        out = os.path.join(RUNS, name)
        os.makedirs(out, exist_ok=True)
        log_path = os.path.join(out, "log.txt")
        recipe = list(RECIPES[student])
        if seed is not None:
            recipe[recipe.index("--seed") + 1] = str(seed)
        if not os.path.exists(os.path.join(out, "run_config.json")):   # written when training finishes
            run([py, os.path.join(STUDENT, "train_student.py"), "--train_csv", train_csv, "--val_csv", val_csv,
                 "--out_dir", out, *recipe], log_path)
        for tag, csv in tests:
            out_json = os.path.join(out, f"eval_{name}__{tag}.json")
            if not os.path.exists(out_json):
                run([py, os.path.join(STUDENT, "eval.py"), "--ckpt", os.path.join(out, "best.pt"), "--csv", csv,
                     *EVAL_ARCH[student], "--num_workers", "0", "--tag", f"{name}__{tag}", "--out_json", out_json],
                    log_path)
        print(f"done: {name}", flush=True)


if __name__ == "__main__":
    main()
