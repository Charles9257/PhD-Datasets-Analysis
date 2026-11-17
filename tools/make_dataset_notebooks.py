"""Generate one Colab-ready exploration notebook per dataset under notebooks/<Dataset>/.

Run from the repo root:  python tools/make_dataset_notebooks.py
The combined notebook (notebooks/SER_Privacy_Dataset_Prep.ipynb) is not touched.
"""
import json
from pathlib import Path

REPO = "Charles9257/PhD-Datasets-Analysis"
BRANCH = "claude/project-thread-9pn62q"
ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Per-dataset pieces: paths (relative to DATA_ROOT), loader code, description and caveats
# --------------------------------------------------------------------------------------
DATASETS = {
"IEMOCAP": dict(
    title="IEMOCAP",
    blurb="Interactive Emotional Dyadic Motion Capture database: 5 sessions, 10 actors (5 F / 5 M), scripted and improvised dyadic dialogues, about 10k utterances with categorical labels.",
    caveats="* About 2,500 utterances have no agreed label (`xxx`) and are dropped.\n* `IEMOCAP_4CLASS = True` gives the standard 4-class setup (angry, happy + excited, neutral, sad), about 5,500 utterances.\n* No age labels.",
    paths='''PATHS = {
    "IEMOCAP_DIR": DATA_ROOT / "IEMOCAP_full_release",
    "IEMOCAP_CSV": DATA_ROOT / "IEMOCAP_full_release" / "iemocap_full_dataset.csv",
}
REQUIRED = ["IEMOCAP_CSV"]
IEMOCAP_4CLASS = True       # standard setup: ang / hap(+exc) / neu / sad''',
    loader='''def load_dataset():
    df = pd.read_csv(PATHS["IEMOCAP_CSV"])
    m = {"neu":"neutral","hap":"happy","exc":"happy" if IEMOCAP_4CLASS else "excited","sad":"sad","ang":"angry",
         "fru":"frustrated","fea":"fear","sur":"surprise","dis":"disgust","oth":"other","xxx":None}
    rows = []
    for r in df.itertuples():
        rel = str(r.path).replace("\\\\", "/")          # CSV may hold Windows-style paths
        stem = Path(rel).stem                           # e.g. Ses01F_script02_1_F000
        spk_sex = stem.split("_")[-1][0]                # F/M of the actual speaker
        rows.append(dict(path=str(PATHS["IEMOCAP_DIR"] / rel), corpus="IEMOCAP",
                         emotion=m.get(r.emotion), emotion_raw=r.emotion,
                         speaker=f"IEM_{stem[:5]}{spk_sex}", gender="female" if spk_sex == "F" else "male",
                         age=np.nan, extra=f"session={r.session};method={r.method};agree={r.agreement}"))
    out = pd.DataFrame(rows).dropna(subset=["emotion"])
    if IEMOCAP_4CLASS:
        out = out[out.emotion.isin(["angry","happy","neutral","sad"])]
    return out.reset_index(drop=True)'''),

"CREMA-D": dict(
    title="CREMA-D",
    blurb="Crowd-sourced Emotional Multimodal Actors Dataset: 7,442 clips from 91 actors (48 M / 43 F, aged 20 to 74), 12 sentences, 6 emotions, with age, race and ethnicity metadata. The best corpus here for **age** inference.",
    caveats="* Uses `Datasets/Crema` (complete 7,442 wav). The `CREMA-D-master/AudioMP3` folder is partial (76 of 91 actors).\n* Labels are the intended (acted) emotion from the filename; crowd-vote tables are in `CREMA-D-master` if you need perceived labels.",
    paths='''PATHS = {
    "CREMA_WAV":  DATA_ROOT / "Datasets" / "Crema",                       # complete 7,442 wav
    "CREMA_DEMO": DATA_ROOT / "CREMA-D-master" / "VideoDemographics.csv",
}
REQUIRED = ["CREMA_WAV", "CREMA_DEMO"]''',
    loader='''def load_dataset():
    demo = pd.read_csv(PATHS["CREMA_DEMO"]).set_index("ActorID")
    m = {"ANG":"angry","DIS":"disgust","FEA":"fear","HAP":"happy","NEU":"neutral","SAD":"sad"}
    rows = []
    for f in sorted(PATHS["CREMA_WAV"].glob("*.wav")):
        a, sent, emo, lvl = f.stem.split("_")
        d = demo.loc[int(a)]
        rows.append(dict(path=str(f), corpus="CREMA-D", emotion=m[emo], emotion_raw=emo, speaker=f"CRE_{a}",
                         gender=d.Sex.lower(), age=float(d.Age),
                         extra=f"sentence={sent};level={lvl};race={d.Race};ethnicity={d.Ethnicity}"))
    return pd.DataFrame(rows)'''),

"RAVDESS": dict(
    title="RAVDESS",
    blurb="Ryerson Audio-Visual Database of Emotional Speech and Song, audio speech part: 1,440 clips from 24 actors (12 F / 12 M), 8 emotions, 2 intensities, 2 statements.",
    caveats="* Odd actor numbers are male, even are female.\n* Only the speech audio is used; video and song zips are in `RAVDESS` and `RAVESS`.\n* No age labels.",
    paths='''PATHS = {
    "RAVDESS_WAV": DATA_ROOT / "Datasets" / "Ravdess" / "audio_speech_actors_01-24",
}
REQUIRED = ["RAVDESS_WAV"]''',
    loader='''def load_dataset():
    m = {"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fear","07":"disgust","08":"surprise"}
    rows = []
    for f in sorted(PATHS["RAVDESS_WAV"].rglob("*.wav")):
        mod, voc, emo, inten, stmt, rep, actor = f.stem.split("-")
        rows.append(dict(path=str(f), corpus="RAVDESS", emotion=m[emo], emotion_raw=emo, speaker=f"RAV_{actor}",
                         gender="male" if int(actor) % 2 else "female", age=np.nan,
                         extra=f"intensity={inten};statement={stmt};rep={rep}"))
    return pd.DataFrame(rows)'''),

"EmoDB": dict(
    title="EmoDB",
    blurb="Berlin Database of Emotional Speech: 535 German utterances from 10 actors (5 F / 5 M, aged 21 to 35), 7 emotions.",
    caveats="* Small corpus: results vary a lot between splits.\n* Age range is narrow (21 to 35), so age probes here say little.",
    paths='''PATHS = {
    "EMODB_WAV":  DATA_ROOT / "Datasets" / "EmoDB",
    "EMODB_DEMO": DATA_ROOT / "Datasets" / "emoDB demographics - Sheet1.csv",
}
REQUIRED = ["EMODB_WAV", "EMODB_DEMO"]''',
    loader='''def load_dataset():
    demo = pd.read_csv(PATHS["EMODB_DEMO"]).set_index("ActorID")
    m = {"W":"angry","L":"boredom","E":"disgust","A":"fear","F":"happy","T":"sad","N":"neutral"}
    rows = []
    for f in sorted(PATHS["EMODB_WAV"].glob("*.wav")):
        spk, emo = int(f.stem[:2]), f.stem[5]
        d = demo.loc[spk]
        rows.append(dict(path=str(f), corpus="EmoDB", emotion=m[emo], emotion_raw=emo, speaker=f"EMO_{spk:02d}",
                         gender=d.Gender.lower(), age=float(d.Age), extra=f"text={f.stem[2:5]}"))
    return pd.DataFrame(rows)'''),

"SAVEE": dict(
    title="SAVEE",
    blurb="Surrey Audio-Visual Expressed Emotion: 480 British English utterances from 4 male speakers (DC, JE, JK, KL), 7 emotions, recorded at 44.1 kHz.",
    caveats="* All speakers are male, so gender inference cannot be tested on SAVEE alone (the gender probe is skipped).\n* Only 4 speakers: speaker-independent results are noisy.\n* A second copy (960 wav) is in `AudioData`; this notebook uses `Datasets/Savee`.",
    paths='''PATHS = {
    "SAVEE_WAV": DATA_ROOT / "Datasets" / "Savee",
}
REQUIRED = ["SAVEE_WAV"]''',
    loader='''def load_dataset():
    m = {"a":"angry","d":"disgust","f":"fear","h":"happy","n":"neutral","sa":"sad","su":"surprise"}
    rows = []
    for f in sorted(PATHS["SAVEE_WAV"].glob("*.wav")):
        spk, code_ = f.stem.split("_")
        emo = re.match(r"[a-z]+", code_).group()
        rows.append(dict(path=str(f), corpus="SAVEE", emotion=m[emo], emotion_raw=emo, speaker=f"SAV_{spk}",
                         gender="male", age=np.nan, extra=""))
    return pd.DataFrame(rows)'''),

"TESS": dict(
    title="TESS",
    blurb="Toronto Emotional Speech Set: 2,800 clips, 200 target words spoken in a carrier phrase by 2 female actresses (OAF aged 64, YAF aged 26), 7 emotions.",
    caveats="* The folder holds the set twice (a nested copy); duplicates are removed by file name.\n* Both speakers are female, so the gender probe is skipped. Speaker ID is trivially easy with 2 speakers.\n* Age is only young (26) vs old (64); this is stored in `age` and as `age_group`.",
    paths='''PATHS = {
    "TESS_DIR": DATA_ROOT / "TESS",
}
REQUIRED = ["TESS_DIR"]''',
    loader='''def load_dataset():
    m = {"angry":"angry","disgust":"disgust","fear":"fear","happy":"happy","neutral":"neutral","sad":"sad",
         "ps":"surprise","surprise":"surprise","surprised":"surprise"}
    ages = {"OAF": 64.0, "YAF": 26.0}
    rows, seen = [], set()
    for f in sorted(PATHS["TESS_DIR"].rglob("*.wav")):
        key = f.name.lower()
        if key in seen:                                  # nested duplicate copy
            continue
        seen.add(key)
        parts = f.stem.split("_")
        spk, word, emo = parts[0].upper(), parts[1] if len(parts) > 2 else "", parts[-1].lower()
        if emo not in m:                                 # fall back to the folder name, e.g. YAF_pleasant_surprised
            emo = f.parent.name.split("_")[-1].lower()
        rows.append(dict(path=str(f), corpus="TESS", emotion=m.get(emo, "other"), emotion_raw=emo, speaker=f"TESS_{spk}",
                         gender="female", age=ages.get(spk, np.nan),
                         extra=f"word={word};age_group={'old' if spk == 'OAF' else 'young'}"))
    return pd.DataFrame(rows)'''),

"MSP-IMPROV": dict(
    title="MSP-IMPROV",
    blurb="MSP-IMPROV: acted dyadic improvisations designed to elicit target emotions (angry, happy, neutral, sad). **Only session 1 of 6 is in the folder** (speakers F01 and M01, 989 wav).",
    caveats="* **No `Evaluation.txt` label file is present.** The `emotion` column here is the *target* emotion of the scenario (from the file name / `S01A`-style folder), not the perceived label. Treat results as provisional until sessions 2 to 6 and the labels are added.\n* File names look like `MSP-IMPROV-S01A-F01-P-FM01`: sentence 01, target A(ngry), speaker F01, scenario type (R = read, S/T = improvised, P = natural interaction), dyad order.\n* With one female and one male speaker, a speaker-independent gender probe is impossible (skipped automatically).",
    paths='''PATHS = {
    "MSP_IMPROV_DIR": DATA_ROOT / "MSP-IMPROV",
}
REQUIRED = ["MSP_IMPROV_DIR"]
MSP_SCENARIOS = None        # e.g. ["P"] for natural interaction only, ["R"] for read speech; None = all''',
    loader='''MSP_RE = re.compile(r"MSP-IMPROV-S(\\d+)([AHNS])-([FM])(\\d+)-([A-Z])-([FM]{2})(\\d+)")

def load_dataset():
    m = {"A":"angry","H":"happy","N":"neutral","S":"sad"}
    scen = {"R":"read","S":"improvised_other","T":"improvised_target","P":"natural_interaction"}
    rows, bad = [], 0
    for f in sorted(PATHS["MSP_IMPROV_DIR"].rglob("MSP-IMPROV-*.wav")):
        mt = MSP_RE.match(f.stem)
        if not mt:
            bad += 1; continue
        sent, emo, sex, num, sc, dyad, idx = mt.groups()
        if MSP_SCENARIOS and sc not in MSP_SCENARIOS:
            continue
        rows.append(dict(path=str(f), corpus="MSP-IMPROV", emotion=m[emo], emotion_raw=emo, speaker=f"MSP_{sex}{num}",
                         gender="female" if sex == "F" else "male", age=np.nan,
                         extra=f"sentence={sent};scenario={scen.get(sc, sc)};dyad={dyad};idx={idx};label=target_not_perceived"))
    if bad: print(f"{bad} files did not match the MSP-IMPROV naming pattern and were skipped")
    return pd.DataFrame(rows)'''),

"CommonVoice-SPS": dict(
    title="Common Voice Spontaneous Speech",
    blurb="Mozilla Common Voice **Spontaneous Speech 1.0** (Sep 2025): 21 languages (no English), about 38.5k mp3 clips, about 380 speakers. Useful for **speaker-ID attacks** and cross-lingual checks. It has **no emotion labels**, so the SER sections are skipped.",
    caveats="* Uses the already-extracted folder `mozilla-common-voice/mcv-sps-st-09-2025`. If you only have the `.tar.gz` on Drive, the config cell can extract the languages you pick to the Colab disk (`EXTRACT_FROM_TAR = True`).\n* Gender/age are labelled for only about 4.6% of clips; only Western Penan (`pne`) has both genders and an age spread. Set `LANGS = [\"pne\"]` for gender/age probes.\n* Audio is mp3: loading needs ffmpeg (preinstalled on Colab).\n* This is **not** the standard (read speech) Common Voice release planned for gender/age at scale.",
    paths='''CV_ROOT = DATA_ROOT / "mozilla-common-voice" / "mcv-sps-st-09-2025"
CV_TAR = DATA_ROOT / "mcv-sps-st-09-2025.tar.gz"
LANGS = None                # None = all 21 languages; e.g. ["pne"] (both genders + ages), ["cy", "sco"]
EXTRACT_FROM_TAR = False    # True: extract LANGS from the .tar.gz to /content/cv_sps (Colab disk), if the folder is not on Drive
if EXTRACT_FROM_TAR and CV_TAR.exists():
    import tarfile
    dest = Path("/content/cv_sps" if IN_COLAB else "cv_sps")
    with tarfile.open(CV_TAR) as tf:
        members = [mb for mb in tf.getmembers() if LANGS is None or any(f"-{l}/" in mb.name for l in LANGS)]
        tf.extractall(dest, members=members)
    CV_ROOT = next(p.parent for p in dest.rglob("ss-corpus-*.tsv")).parent
PATHS = {"CV_ROOT": CV_ROOT}
REQUIRED = ["CV_ROOT"]''',
    loader='''AGE_MID = {"teens":16,"twenties":25,"thirties":35,"fourties":45,"forties":45,"fifties":55,"sixties":65,
           "seventies":75,"eighties":85,"nineties":95}

def norm_gender(g):
    g = str(g).lower()
    if g.startswith("female") or g == "woman": return "female"
    if g.startswith("male") or g == "man": return "male"
    return np.nan

def load_dataset():
    frames = []
    for lang_dir in sorted(PATHS["CV_ROOT"].glob("sps-corpus-*")):
        lang = lang_dir.name.split("-")[-1]
        if LANGS and lang not in LANGS: continue
        tsv = lang_dir / f"ss-corpus-{lang}.tsv"
        if not tsv.exists(): continue
        t = pd.read_csv(tsv, sep="\\t", quoting=3, dtype=str, on_bad_lines="skip")
        t["lang"] = lang; t["path"] = [str(lang_dir / "audios" / a) for a in t.audio_file]
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(dict(path=t.path, corpus="CV-SPS", emotion=np.nan, emotion_raw=np.nan,
                             speaker="CV_" + t.client_id.str[:12], gender=t.gender.map(norm_gender),
                             age=t.age.str.lower().map(AGE_MID), lang=t.lang,
                             duration_s=pd.to_numeric(t.duration_ms, errors="coerce") / 1000,
                             extra="split=" + t.get("split", pd.Series("", index=t.index)).fillna("") + ";age_group=" + t.age.fillna("")))'''),
}

# --------------------------------------------------------------------------------------
# Shared cells
# --------------------------------------------------------------------------------------
SETUP = '''import sys, subprocess
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # Colab already has most packages; this makes sure the audio ones are present and recent enough.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "librosa>=0.10", "soundfile", "transformers>=4.40", "ipywidgets"], check=True)
else:
    print("Not on Colab: skipping Drive mount. Install dependencies once with: pip install -r requirements.txt")

try:
    import torch
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("No GPU found, running on CPU. On Colab: Runtime > Change runtime type > T4 GPU, then re-run from the top.")
except ImportError:
    print("PyTorch not installed: the SSL embedding step will be skipped.")'''

CONFIG = '''from pathlib import Path
import os, re, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

DATASET = "__NAME__"

# ---- Paths ----
# DATA_ROOT mirrors the local folder C:\\Desktop\\RGU PhD\\New folder\\Datasets.
# On Colab, copy/sync that folder to Google Drive as MyDrive/Datasets, or change the path below.
IN_COLAB = "google.colab" in __import__("sys").modules
DEFAULT_ROOT = "/content/drive/MyDrive/Datasets" if IN_COLAB else r"C:\\Desktop\\RGU PhD\\New folder\\Datasets"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", DEFAULT_ROOT))

__PATHS__

# ---- Outputs (never inside the source folders) ----
if IN_COLAB:
    OUT_DIR = Path("/content/drive/MyDrive/SER_outputs") / DATASET
else:
    try:
        NB_DIR = Path(__vsc_ipynb_file__).parent          # VS Code
    except NameError:
        NB_DIR = Path.cwd()                               # Jupyter: folder the notebook was opened from
    OUT_DIR = NB_DIR / "outputs"
(OUT_DIR / "features").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

# ---- Knobs ----
SEED = 42
N_SAMPLE = 600              # utterances sampled (spread across speakers) for features/probes; None = all (slow)
MAX_SECONDS = 6.0           # crop long utterances
TARGET_SR = 16_000
RUN_SSL = True              # wav2vec2 embeddings (GPU recommended); set False for handcrafted features only
SSL_MODEL = "facebook/wav2vec2-base"   # or "facebook/hubert-base-ls960"

random.seed(SEED); np.random.seed(SEED)
for k, p in PATHS.items():
    print(f"{k:16s} {'OK ' if p.exists() else 'MISSING'}  {p}")
print("Outputs ->", OUT_DIR)
if not all(PATHS[k].exists() for k in REQUIRED):
    raise FileNotFoundError(f"{DATASET} files not found under {DATA_ROOT}. Check the Drive folder or edit DATA_ROOT / PATHS above.")'''

SUMMARY = '''meta = load_dataset()
meta.to_csv(OUT_DIR / f"{DATASET}_metadata.csv", index=False)
HAS_EMO = meta.emotion.notna().any()
HAS_GENDER = meta.gender.dropna().nunique() > 1
HAS_AGE = meta.age.notna().any() and meta.drop_duplicates("speaker").age.nunique() > 1

print(f"{DATASET}: {len(meta):,} utterances, {meta.speaker.nunique()} speakers")
print("Gender (speakers):", meta.drop_duplicates("speaker").gender.value_counts(dropna=False).to_dict())
if HAS_EMO: print("Emotions:", meta.emotion.value_counts().to_dict())
speakers = meta.groupby("speaker").agg(gender=("gender", "first"), age=("age", "first"), utterances=("path", "size"))
display(meta.head())
speakers'''

PLOTS = '''panels = [p for p, ok in [("emotion", HAS_EMO), ("speaker", True), ("emo_gender", HAS_EMO and HAS_GENDER), ("age", HAS_AGE)] if ok]
fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False); axes = axes[0]
for ax, p in zip(axes, panels):
    if p == "emotion":
        meta.emotion.value_counts().plot.bar(ax=ax, color="tab:blue"); ax.set_title("Utterances per emotion")
    elif p == "speaker":
        top = speakers.sort_values("utterances", ascending=False).head(40)
        top.utterances.plot.bar(ax=ax, color=top.gender.map({"female": "tab:red", "male": "tab:blue"}).fillna("grey").tolist())
        ax.set_title("Utterances per speaker (red F, blue M, grey unknown)"); ax.set_xticklabels([])
    elif p == "emo_gender":
        pd.crosstab(meta.emotion, meta.gender).plot.bar(ax=ax); ax.set_title("Emotion x gender")
    elif p == "age":
        speakers.age.dropna().plot.hist(bins=15, ax=ax); ax.set_title("Speaker age")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(OUT_DIR / "figures" / "distributions.png", dpi=150); plt.show()

if HAS_EMO and HAS_GENDER:
    # Emotion and gender are often correlated in acted corpora; this matters for attribute leakage
    display(pd.crosstab(meta.emotion, meta.gender, normalize="index").round(2))'''

DURATION = '''import librosa, librosa.display, soundfile as sf
from IPython.display import Audio, display

def info(p):
    try:
        i = sf.info(p); return i.duration, i.samplerate
    except Exception:
        return librosa.get_duration(path=p), librosa.get_samplerate(p)

samp = meta.sample(min(len(meta), 300), random_state=SEED).copy()
samp[["dur", "sr"]] = [info(p) for p in samp.path]
print("Sample rates:", samp.sr.value_counts().to_dict())
samp.groupby("emotion" if HAS_EMO else "gender", dropna=False).dur.describe().round(2)'''

EXAMPLES = '''def load_audio(path, sr=TARGET_SR, max_s=None):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_s)
    return y

def show_example(row):
    y = load_audio(row.path, max_s=10)
    fig, ax = plt.subplots(1, 2, figsize=(14, 3))
    librosa.display.waveshow(y, sr=TARGET_SR, ax=ax[0]); ax[0].set_title(f"{row.emotion} | {row.speaker} | {row.gender}")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64), ref=np.max)
    librosa.display.specshow(S, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=ax[1]); ax[1].set_title("log-mel")
    plt.show(); display(Audio(y, rate=TARGET_SR))

by = "emotion" if HAS_EMO else "speaker"
for _, row in meta.dropna(subset=[by]).groupby(by).sample(1, random_state=SEED).head(8).iterrows():
    show_example(row)'''

FEATURES = '''from tqdm.auto import tqdm

def sample_meta(meta, n=N_SAMPLE):
    # sample utterances spreading across speakers (and emotions where available)
    if n is None or n >= len(meta): return meta.reset_index(drop=True)
    per = max(1, n // meta.speaker.nunique())
    s = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in meta.groupby("speaker")])
    if len(s) < n:
        s = pd.concat([s, meta.drop(s.index).sample(min(n - len(s), len(meta) - len(s)), random_state=SEED)])
    return s.sample(min(n, len(s)), random_state=SEED).reset_index(drop=True)

work = sample_meta(meta)
print(len(work), "utterances,", work.speaker.nunique(), "speakers")

def preprocess(path):
    y = load_audio(path)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.abs(y).max() + 1e-8)
    return y[: int(MAX_SECONDS * TARGET_SR)]

def handcrafted(y):
    mf = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=40)
    mf = np.vstack([mf, librosa.feature.delta(mf)])
    lm = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64))
    return np.concatenate([mf.mean(1), mf.std(1)]), np.concatenate([lm.mean(1), lm.std(1)])

tag = f"n{N_SAMPLE}_s{SEED}"
f_cache = OUT_DIR / "features" / f"handcrafted_{tag}.npz"
if f_cache.exists():
    z = np.load(f_cache); X_mfcc, X_mel = z["mfcc"], z["mel"]
else:
    mf, ml = [], []
    for p in tqdm(work.path, desc="MFCC/log-mel"):
        a, b = handcrafted(preprocess(p)); mf.append(a); ml.append(b)
    X_mfcc, X_mel = np.array(mf), np.array(ml)
    np.savez(f_cache, mfcc=X_mfcc, mel=X_mel)
work.to_csv(OUT_DIR / "features" / f"work_meta_{tag}.csv", index=False)
FEATS = {"MFCC": X_mfcc, "log-mel": X_mel}
{k: v.shape for k, v in FEATS.items()}'''

SSL = '''if RUN_SSL:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = SSL_MODEL.split("/")[-1]
    cache = OUT_DIR / "features" / f"{short}_{tag}.npy"
    if cache.exists():
        E = np.load(cache)
    else:
        fe = AutoFeatureExtractor.from_pretrained(SSL_MODEL)
        model = AutoModel.from_pretrained(SSL_MODEL, output_hidden_states=True).to(device).eval()
        out = []
        with torch.no_grad():
            for p in tqdm(work.path, desc=short):
                inp = fe(preprocess(p), sampling_rate=TARGET_SR, return_tensors="pt").to(device)
                hs = model(**inp).hidden_states                                   # tuple(L+1) of [1, T, 768]
                out.append(torch.stack([h.mean(1)[0] for h in hs]).cpu().numpy())  # [L+1, 768]
        E = np.stack(out); np.save(cache, E)                                     # [N, L+1, 768]
    FEATS[short] = E[:, -1]
    print(short, E.shape)
else:
    print("RUN_SSL = False: using handcrafted features only")'''

PROBES = '''from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, mean_absolute_error

def clf(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))

def split(y, groups, speaker_independent, test_size=0.25):
    """Speaker-independent split with >= 2 classes on both sides (tries several seeds); None if impossible."""
    for s in range(SEED, SEED + 30):
        if speaker_independent:
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(y, y, groups))
        else:
            tr, te = next(StratifiedShuffleSplit(1, test_size=test_size, random_state=s).split(y, y))
        if len(set(y[tr])) > 1 and len(set(y[te])) > 1 and set(y[te]) <= set(y[tr]):
            return tr, te
    return None

def eval_task(X, y, groups, speaker_independent=True):
    y = np.asarray(y); sp = split(y, np.asarray(groups), speaker_independent)
    if sp is None: return None
    tr, te = sp; m = clf().fit(X[tr], y[tr]); p = m.predict(X[te])
    maj = pd.Series(y[tr]).mode()[0]
    acc, base = accuracy_score(y[te], p), accuracy_score(y[te], np.full(len(te), maj))
    return dict(UAR=balanced_accuracy_score(y[te], p), F1=f1_score(y[te], p, average="macro"),
                acc=acc, majority_acc=base, advantage=acc - base)

res = []
for fname, X in FEATS.items():
    tasks = []
    if HAS_EMO:
        ok = work.emotion.notna().values
        tasks.append(("emotion (unseen speakers)", ok, work.emotion.values, True))
    if HAS_GENDER:
        ok = work.gender.notna().values
        tasks.append(("gender (unseen speakers)", ok, work.gender.values, True))
    counts = work.speaker.map(work.speaker.value_counts())
    if work.speaker.nunique() > 1:
        ok = (counts >= 4).values
        tasks.append(("speaker ID (closed set)", ok, work.speaker.values, False))
    for task, ok, y, si in tasks:
        r = eval_task(X[ok], y[ok], work.speaker.values[ok], si)
        if r is None:
            print(f"skip {task}: not enough speakers/classes for a valid split"); continue
        r.update(task=task, features=fname); res.append(r)
probe = pd.DataFrame(res)
if len(probe):
    probe = probe[["task", "features", "UAR", "F1", "acc", "majority_acc", "advantage"]].round(3)
    probe.to_csv(OUT_DIR / f"probes_{tag}.csv", index=False)
probe'''

AGE = '''# Age probe: regression MAE on unseen speakers (only if the dataset has age labels for several speakers)
ok = work.age.notna().values
if HAS_AGE and work[ok].speaker.nunique() >= 4:
    g = work[ok]; yv = g.age.values
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(g, groups=g.speaker))
    for fname, X in FEATS.items():
        Xi = X[ok]
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(Xi[tr], yv[tr])
        print(f"{fname:16s} age MAE = {mean_absolute_error(yv[te], m.predict(Xi[te])):.1f} y   "
              f"(predict-mean MAE = {mean_absolute_error(yv[te], np.full(len(te), yv[tr].mean())):.1f})")
else:
    print("Age probe skipped: no age labels or fewer than 4 speakers with age")'''

NEXT = '''## Notes

* Metadata for this corpus is saved as `{DATASET}_metadata.csv` in `OUT_DIR`, with the same columns as the combined notebook (`path, corpus, emotion, emotion_raw, speaker, gender, age, extra`), so tables from the per-dataset notebooks can be concatenated.
* **Speaker-independent** splits are used for emotion and gender; speaker ID is closed-set by definition.
* Attacker **advantage** = accuracy minus majority-class accuracy. Proposed gate: a defence passes if advantage stays within 5 pp while emotion UAR drops by at most 5 points (pending supervisor confirmation).
* For cross-corpus experiments and the FedAvg attack simulation, use the combined notebook `notebooks/SER_Privacy_Dataset_Prep.ipynb`.'''


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def colab_url(rel):
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel}"


def build(name, d):
    folder = name.replace(" ", "_")
    rel = f"notebooks/{folder}/{folder}_Explore.ipynb"
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel)})"
    has_emo = name != "CommonVoice-SPS"
    cells = [
        md(f"{badge}\n\n# {d['title']}: load, explore and probe\n\n{d['blurb']}\n\n"
           "This notebook works on this one dataset only. It builds a metadata table (path, emotion, speaker, gender, age), "
           "shows distributions, durations and audio examples, extracts MFCC / log-mel"
           + (" / wav2vec2" ) + " features, and runs quick probes: "
           + ("an emotion baseline (UAR, macro-F1) and " if has_emo else "")
           + "attribute probes (gender, speaker ID, age where available) as a first leakage check.\n\n"
           f"**Dataset notes**\n{d['caveats']}\n\n"
           "**Source data is never modified.** Outputs go to `OUT_DIR` (on Colab: `MyDrive/SER_outputs/"
           f"{name}`).\n\n**On Google Colab:** Runtime > Change runtime type > **T4 GPU**, then run cells top to bottom."),
        md("## 0. Environment setup (Colab or local)\n\nMounts Google Drive on Colab, installs missing packages and checks for a GPU."),
        code(SETUP),
        md("## 1. Configuration\n\n`DATA_ROOT` should mirror your local `Datasets` folder (default on Colab: `MyDrive/Datasets`)."),
        code(CONFIG.replace("__NAME__", name).replace("__PATHS__", d["paths"])),
        md("## 2. Metadata table\n\nOne row per utterance: `path, corpus, emotion, emotion_raw, speaker, gender, age, extra`."),
        code(d["loader"]),
        code(SUMMARY),
        md("## 3. Exploration"),
        code(PLOTS),
        code(DURATION),
        code(EXAMPLES),
        md("## 4. Preprocessing and features\n\n16 kHz mono, silence trim (`top_db=30`), peak normalisation, crop to `MAX_SECONDS`. "
           "MFCC (40 + deltas, mean/std pooled, 160-d), log-mel (64 bands, mean/std, 128-d), and optionally wav2vec2 last-layer "
           "mean-pooled embeddings (all layers are cached). Cached in `OUT_DIR/features`."),
        code(FEATURES),
        code(SSL),
        md("## 5. Quick probes\n\n" + ("Emotion recognition baseline and attribute" if has_emo else "Attribute")
           + " probes (what an attacker could recover from the representations). Probes that the data cannot support "
           "(for example gender on a single-gender corpus) are skipped with a message."),
        code(PROBES),
        code(AGE),
        md(NEXT),
    ]
    for i, c in enumerate(cells):
        c["id"] = f"{folder.lower().replace('-', '')}-{i:02d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}, "accelerator": "GPU",
                                        "colab": {"provenance": [], "gpuType": "T4"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return name, rel


if __name__ == "__main__":
    for name, d in DATASETS.items():
        n, rel = build(name, d)
        print(f"| {d['title']} | [{rel}]({rel}) | [Open in Colab]({colab_url(rel)}) |")
