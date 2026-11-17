# PhD Datasets Analysis: SER privacy

Dataset preparation, exploration and first leakage checks for the PhD work on **paralinguistic inference attacks in federated speech emotion recognition (SER)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Charles9257/PhD-Datasets-Analysis/blob/claude/project-thread-9pn62q/notebooks/SER_Privacy_Dataset_Prep.ipynb)

## What's here

| Path | What it is |
|---|---|
| `notebooks/SER_Privacy_Dataset_Prep.ipynb` | Unified metadata for IEMOCAP, CREMA-D, RAVDESS, EmoDB, SAVEE; exploration; MFCC / log-mel / wav2vec2 / HuBERT features; SER baseline (UAR, macro-F1); gender / speaker / age attribute probes; minimal FedAvg with a server-side gender inference attack and a DP switch |
| `docs/datasets_folder_survey.md` | Survey of the local `Datasets` folder: what each corpus holds and its gaps for attribute inference |
| `requirements.txt` | Packages for running locally |

**No audio or dataset files are in this repository.** The corpora are licensed; keep them on Google Drive or your own machine. `.gitignore` blocks audio, video, archives and cached features.

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
