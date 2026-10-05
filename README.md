# EMO-MAVCeleb

<img width="1292" height="516" alt="Picture-ID0005-New" src="https://github.com/user-attachments/assets/c1a661ae-2382-444f-80d9-088a226b32a3" />

Cross-modal emotion distillation on MAV-Celeb (v1 English/Urdu, v2 English/Hindi, v3 English/German).
A face-emotion teacher labels the video frames; the frame labels are pooled per video and used as soft
targets for an audio student (VGGVox trained from scratch, or a frozen WavLM Base+ with a trainable head).

This repository contains the code and labels needed to reproduce the results in the paper.

## Repository layout

```
teacher/poster/label_poster.py      POSTER V2 (AffectNet-8) per-frame face labels      -> "our approach"
teacher/senet/dump_frames_senet.m   SENet-50 FER+ per-frame face labels (MatConvNet)  -> baseline (Albanie et al.)
student/                            audio students: training, evaluation, VGGVox and WavLM models
pipeline/clip_labels.py             1  frame labels -> clip labels (max-pooled per video)
pipeline/build_splits.py            2  all train / val / test splits (seed 1337)
pipeline/run_experiments.py         3  train + evaluate every student model of the paper
pipeline/make_tables.py             4  paper tables from the eval JSONs
data/clip_labels.csv                clip labels of both teachers (output of step 1)
results/evals/                      eval JSONs of the models reported in the paper
results/tables/                     tables generated from them
```

## Setup

Python 3.12 with a CUDA GPU (tested with PyTorch 2.5.1, CUDA 12.1):

```bash
pip install -r requirements.txt
```

Download MAV-Celeb v1, v2 and v3 and point `MAVCELEB_DATA_ROOT` to the folder that contains them:

```
<MAVCELEB_DATA_ROOT>/v1/voices/<speaker>/<Language>/<video>/00000.wav
<MAVCELEB_DATA_ROOT>/v1/faces/<speaker>/<Language>/<video>/<timestamp>.jpg
<MAVCELEB_DATA_ROOT>/v2/...
<MAVCELEB_DATA_ROOT>/v3/...
```

All `wav_path` entries in the CSVs are relative to this folder. Default: `./mavceleb`.

## Reproducing the results

Steps 2-4 only need the audio; the teacher labels are included in `data/clip_labels.csv`.

```bash
python pipeline/build_splits.py
python pipeline/run_experiments.py
python pipeline/make_tables.py
```

`run_experiments.py` trains 37 models (18 per student, plus one retrain, see below) one after the other
and can be restarted at any time; use `--students vgg` or `--only <run>` to run a subset. Training is
deterministic (`--deterministic`, seed 1337).

To check the paper numbers without training, build the tables from the shipped eval JSONs:

```bash
python pipeline/make_tables.py --evals results/evals
```

| Paper table | Models (`run_experiments.py`) | Test sets (`data/splits/`) | Output (`results/tables/`) |
|---|---|---|---|
| Emotion distribution per version | - | - | `clip_label_class_distribution.csv` |
| Known speakers | `{vgg,wavlm}_{v}_{language}` | `known/*_test.csv` | `tables_known_and_pooled.tex` |
| Individual vs. pooled | per-version models, `*_pooled_english`, `*_pooled_nonenglish` | `all/` | `tables_known_and_pooled.tex` |
| Unknown speakers | `*_unknown` (train v1+v2 English) | `all/poster_v3_*_all.csv` | `unknown_speakers.csv` |

All numbers are Macro-F1 against the POSTER V2 labels.

### Splits

* **Known speakers:** for each version and language, ~70/15/15 split by video. All segments of a video
  share one label, so a whole video stays in one split. Every speaker keeps at least one training video,
  so every test speaker was seen during training.
* **Pooled:** English of v1+v2+v3, or Urdu+Hindi+German (10% random validation). Tested on all clips of
  the language that was not trained on.
* **Unknown speakers:** English of v1+v2, tested on v3. Speaker ids are local to each version; the
  versions share no identities.

### Training recipes

| | VGGVox | WavLM Base+ |
|---|---|---|
| Initialisation | from scratch | `microsoft/wavlm-base-plus`, encoder frozen |
| Optimiser | SGD, momentum 0.9, weight decay 5e-4 | AdamW, weight decay 5e-4 |
| Learning rate | 1e-4 -> 1e-5 (log-linear) | 1e-3 -> 1e-4 (log-linear) |
| Common | 50 epochs, batch 32, 4 s crops at 16 kHz, distillation temperature 2, class-weighted loss, checkpoint with the lowest validation loss |

WavLM trained on V3-German alone collapsed to a single class with seed 1337 (best epoch 1). As stated in
the paper, it was retrained once with seed 1338 (`wavlm_v3_german_seed1338`), and that run is reported.

## Teacher labels (optional)

Only needed to re-create `data/clip_labels.csv`. Write the outputs to
`$MAVCELEB_TEACHER_ROOT/{poster,senet}/mavceleb_v{1,2,3}/` (default `./teacher_outputs`), then run
`python pipeline/clip_labels.py`.

**POSTER V2.** Clone [POSTER V2](https://github.com/Talented-Q/POSTER_V2), download its AffectNet-8
checkpoint, and run once per version:

```bash
python teacher/poster/label_poster.py --faces_root mavceleb/v1/faces --out_dir teacher_outputs/poster/mavceleb_v1 --poster_repo POSTER_V2 --ckpt affectnet-8-model_best.pth
```

**SENet FER+.** Uses the model of [Albanie et al., 2018](https://github.com/albanie/mcnCrossModalEmotions)
in [MatConvNet](https://www.vlfeat.org/matconvnet/). We used MATLAB R2019b with Visual Studio 2017,
started MATLAB from the Visual Studio developer prompt, and compiled MatConvNet as described on its
website. Then, once per version (absolute paths):

```matlab
dump_frames_senet('D:\mavceleb\v1\faces', 'D:\teacher_outputs\senet\mavceleb_v1', 'D:\matconvnet-1.0-beta25')
```

## Cite

If you use this work, please also cite the baseline:

```bibtex
@inproceedings{albanie2018emotion,
  title={Emotion recognition in speech using cross-modal transfer in the wild},
  author={Albanie, Samuel and Nagrani, Arsha and Vedaldi, Andrea and Zisserman, Andrew},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={292--301},
  year={2018}
}
```
