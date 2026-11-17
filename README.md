# PhD Datasets Analysis: SER privacy

Dataset preparation, exploration and first leakage checks for the PhD work on **paralinguistic inference attacks in federated speech emotion recognition (SER)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/SER_Privacy_Dataset_Prep.ipynb)

## What's here

| Path | What it is |
|---|---|
| `notebooks/SER_Privacy_Dataset_Prep.ipynb` | Unified metadata for IEMOCAP, CREMA-D, RAVDESS, EmoDB, SAVEE; exploration; MFCC / log-mel / wav2vec2 / HuBERT features; SER baseline (UAR, macro-F1); gender / speaker / age attribute probes; minimal FedAvg with a server-side gender inference attack and a DP switch |
| `notebooks/<Dataset>/<Dataset>_Explore.ipynb` | One notebook per dataset (see below) |
| `docs/datasets_folder_survey.md` | Survey of the local `Datasets` folder: what each corpus holds and its gaps for attribute inference |
| `requirements.txt` | Packages for running locally |

**No audio or dataset files are in this repository.** The corpora are licensed; keep them on Google Drive or your own machine. `.gitignore` blocks audio, video, archives and cached features.

## Per-dataset notebooks

One notebook per corpus, each in its own folder, so you can analyse a dataset on its own. Each one covers, for that dataset:

1. **Threat model** (honest-but-curious server; assets: gender, age, speaker identity, membership)
2. **Data inspection**: rows, columns, data types, missing values (NaN), a detailed look at every column, duplicates
3. **Acoustic feature table** (duration, sample rate, loudness, clipping, silence, pitch, spectral shape, SNR)
4. **Descriptive statistics**: central tendency, dispersion, skewness and kurtosis, overall and by emotion and gender
5. **Outliers** (IQR and z-score), **data cleaning** (logged) and **wrangling** (encoding, speaker-independent train/val/test split, class weights)
6. **Analysis and visualisation**, with statistical tests (chi-square, Kruskal-Wallis, Mann-Whitney)
7. **Machine learning models**: logistic regression, SVM, random forest and MLP on MFCC, log-mel and wav2vec2 features
8. **Paralinguistic inference attacks** on the embeddings (gender, speaker, age)
9. **Federated learning** (FedAvg, one client per speaker), with server-side attacks on gender, age, speaker linkage and membership
10. **Defences**: local DP at ε = 1, 3, 5, 8; Secure Aggregation; SecAgg with distributed DP; gradient reversal
11. **Evaluation**: a privacy-utility-efficiency table and the 5 pp / 5 pt decision gate

Sections the data cannot support are skipped with a message (for example, gender attacks on all-male SAVEE, or SER and FL on Common Voice, which has no emotion labels). Outputs go to `MyDrive/SER_outputs/<dataset>`. The combined notebook above is unchanged.

| Dataset | Notebook | Colab |
|---|---|---|
| IEMOCAP | [notebooks/IEMOCAP/IEMOCAP_Explore.ipynb](notebooks/IEMOCAP/IEMOCAP_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/IEMOCAP/IEMOCAP_Explore.ipynb) |
| CREMA-D | [notebooks/CREMA-D/CREMA-D_Explore.ipynb](notebooks/CREMA-D/CREMA-D_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/CREMA-D/CREMA-D_Explore.ipynb) |
| RAVDESS | [notebooks/RAVDESS/RAVDESS_Explore.ipynb](notebooks/RAVDESS/RAVDESS_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/RAVDESS/RAVDESS_Explore.ipynb) |
| EmoDB | [notebooks/EmoDB/EmoDB_Explore.ipynb](notebooks/EmoDB/EmoDB_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/EmoDB/EmoDB_Explore.ipynb) |
| SAVEE | [notebooks/SAVEE/SAVEE_Explore.ipynb](notebooks/SAVEE/SAVEE_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/SAVEE/SAVEE_Explore.ipynb) |
| TESS | [notebooks/TESS/TESS_Explore.ipynb](notebooks/TESS/TESS_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/TESS/TESS_Explore.ipynb) |
| MSP-IMPROV | [notebooks/MSP-IMPROV/MSP-IMPROV_Explore.ipynb](notebooks/MSP-IMPROV/MSP-IMPROV_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/MSP-IMPROV/MSP-IMPROV_Explore.ipynb) |
| MSP-Podcast | [notebooks/MSP-PODCAST/MSP-PODCAST_Explore.ipynb](notebooks/MSP-PODCAST/MSP-PODCAST_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/MSP-PODCAST/MSP-PODCAST_Explore.ipynb) |
| Common Voice Spontaneous Speech | [notebooks/CommonVoice-SPS/CommonVoice-SPS_Explore.ipynb](notebooks/CommonVoice-SPS/CommonVoice-SPS_Explore.ipynb) | [Open in Colab](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/CommonVoice-SPS/CommonVoice-SPS_Explore.ipynb) |

Dataset notes: SAVEE (all male) and TESS (2 female speakers) skip the gender probe. MSP-IMPROV has only session 1 and no label file, so it uses the *target* emotion from the file name. MSP-Podcast has audio only for now (no `labels_consensus.csv`), so it runs in audio-only mode until you add the labels to `MSP-POSCAST/Labels/`. Common Voice Spontaneous Speech has no emotion labels (speaker ID and, for `pne`, gender/age only). Same Drive layout as below; the extra folders are `TESS/`, `MSP-IMPROV/`, `MSP-POSCAST/` and `mozilla-common-voice/` directly under `MyDrive/Datasets`.

`tools/make_dataset_notebooks.py` regenerates these notebooks if you want to change all of them at once.

## Run on Google Colab

1. **Put the datasets on Google Drive.** Create `MyDrive/Datasets` and copy in the same layout as your local `C:\Desktop\RGU PhD\New folder\Datasets`. For this notebook you only need:
   ```
   MyDrive/Datasets/
   ├── IEMOCAP_full_release/            (session folders + iemocap_full_dataset.csv)
   ├── CREMA-D-master/VideoDemographics.csv
   └── Datasets/
       ├── Crema/                        (7,442 wav)
       ├── Ravdess/audio_speech_actors_01-24/
       ├── EmoDB/                        (535 wav)
       ├── emoDB demographics - Sheet1.csv
       └── Savee/
   ```
   Google Drive for desktop is the easiest way to sync these. Any corpus that is missing is skipped automatically.
2. Click the **Open in Colab** badge above.
3. **Runtime > Change runtime type > T4 GPU**.
4. Run the cells top to bottom. The first cell mounts Drive (approve the Google prompt), installs missing packages and prints the GPU.
5. If your Drive folder has a different name, edit `DATA_ROOT` in the configuration cell.

Outputs (unified metadata, cached features, figures, result CSVs) go to `MyDrive/SER_outputs`, so they survive when the Colab session ends and cached features are reused on the next run.

Start with `N_PER_CORPUS = 300` (the default). Raise it, or set it to `None` for all utterances, once everything runs.

Tip: Colab reads many small files from Drive slowly. If loading is slow, copy the corpora to the Colab disk first, e.g. `!cp -r "/content/drive/MyDrive/Datasets" /content/Datasets`, then set `DATA_ROOT = Path("/content/Datasets")`.

## Run locally

```bash
pip install -r requirements.txt
jupyter lab notebooks/SER_Privacy_Dataset_Prep.ipynb
```
Locally, `DATA_ROOT` defaults to `C:\Desktop\RGU PhD\New folder\Datasets` and outputs go to `notebooks/outputs/`.
