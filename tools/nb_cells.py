"""Shared cells for the per-dataset notebooks (used by make_dataset_notebooks.py).

Each entry is notebook source. Dataset-specific pieces (paths, loader) live in make_dataset_notebooks.py.
"""

SETUP = r'''import sys, subprocess
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # Colab already has most packages; this makes sure the audio / stats ones are present and recent enough.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "librosa>=0.10", "soundfile", "transformers>=4.40", "ipywidgets", "scipy", "seaborn"], check=True)
else:
    print("Not on Colab: skipping Drive mount. Install dependencies once with: pip install -r requirements.txt")

try:
    import torch
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("No GPU found, running on CPU. On Colab: Runtime > Change runtime type > T4 GPU, then re-run from the top.")
except ImportError:
    print("PyTorch not installed: SSL embeddings and the federated learning sections will be skipped.")'''

CONFIG = r'''from pathlib import Path
import os, re, json, random, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50); pd.set_option("display.width", 200)

DATASET = "__NAME__"

# ---- Paths ----
# DATA_ROOT mirrors the local folder C:\Desktop\RGU PhD\New folder\Datasets.
# On Colab, copy/sync that folder to Google Drive as MyDrive/Datasets, or change the path below.
IN_COLAB = "google.colab" in __import__("sys").modules
DEFAULT_ROOT = "/content/drive/MyDrive/Datasets" if IN_COLAB else r"C:\Desktop\RGU PhD\New folder\Datasets"
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
for sub in ["features", "figures", "tables"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ---- Data knobs ----
SEED = 42
N_INSPECT = 2000            # files analysed acoustically (duration, loudness, pitch, ...); None = all files (slow on Drive)
INSPECT_MAX_S = 10.0        # seconds of audio read per file for the acoustic table
MIN_DURATION_S = 0.5        # cleaning: shorter clips are removed
MIN_CLASS_COUNT = 10        # cleaning: emotion classes with fewer clips are removed
N_SAMPLE = 600              # utterances (spread across speakers) used for features / models / FL; None = all (slow)
MAX_SECONDS = 6.0           # crop long utterances for features
TARGET_SR = 16_000
SPEAKER_LABEL = "speaker"   # what the speaker column means (a loader may change this, e.g. to a proxy grouping)
RUN_SSL = True              # wav2vec2 embeddings (GPU recommended); False = handcrafted features only
SSL_MODEL = "facebook/wav2vec2-base"   # or "facebook/hubert-base-ls960"

# ---- Federated learning, attacks and defences ----
ROUNDS, LOCAL_EPOCHS, LR, CLIENT_FRAC = 30, 2, 0.05, 0.5
MIN_CLIENTS = 8             # if there are fewer speakers, each speaker's data is split into several clients
CLIP = 1.0                  # L2 clipping norm for DP
DP_EPSILONS = [1, 3, 5, 8]  # target privacy budgets (thesis grid)
DELTA = 1e-5
GRL_LAMBDAS = [0.5, 1.0]    # gradient-reversal strengths (adversarial gender removal)
GATE_ADV = 0.05             # proposed gate: attacker advantage <= 5 pp ...
GATE_UAR = 0.05             # ... and emotion UAR loss <= 5 points vs undefended FedAvg (pending supervisor confirmation)

random.seed(SEED); np.random.seed(SEED)
for k, p in PATHS.items():
    print(f"{k:16s} {'OK ' if p.exists() else 'MISSING'}  {p}")
print("Outputs ->", OUT_DIR)
if not all(PATHS[k].exists() for k in REQUIRED):
    raise FileNotFoundError(f"{DATASET} files not found under {DATA_ROOT}. Check the Drive folder or edit DATA_ROOT / PATHS above.")

def save_table(df, name):
    df.to_csv(OUT_DIR / "tables" / f"{name}.csv")
    return df

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT_DIR / "figures" / f"{name}.png", dpi=150); plt.show()'''

THREAT_MODEL = r'''## 2. Threat model (applies to every section below)

| Element | Assumption in this notebook |
|---|---|
| **System** | Federated speech emotion recognition (SER). Each **client = one speaker's device** (or a slice of one speaker's data when a corpus has very few speakers). Clients train locally and send model updates; raw audio never leaves the device. |
| **Adversary** | **Honest-but-curious server**: follows the FedAvg protocol correctly but stores and analyses every update it receives. It knows the model architecture, the round number and which client sent which update (unless SecAgg hides individual updates). It may hold a small labelled auxiliary set (here: the other clients' updates, used with cross-validation). |
| **Assets (what leaks)** | Paralinguistic attributes that are *not* the task: **gender**, **age**, **speaker identity** (linkability across rounds) and **membership** (was this utterance used in training?). Under GDPR, voice used to identify a person is biometric data (Art. 9), and emotion is sensitive; the EU AI Act restricts emotion recognition in some settings. |
| **Attacks** | (A1) attribute inference on shared representations / embeddings; (A2) gender inference from client updates; (A3) age inference from client updates; (A4) speaker linkage (re-identifying the same client across rounds); (A5) membership inference on the final global model (loss threshold). |
| **Defences** | DP-FedAvg with client-side (local) Gaussian noise at ε = 1, 3, 5, 8; Secure Aggregation (pairwise masking, server sees only the sum); SecAgg + distributed DP; gradient reversal (adversarial removal of gender from the shared encoder). |
| **Metrics** | Utility: **UAR** (key), macro-F1, accuracy. Privacy: attack balanced accuracy and **advantage over chance/majority**, membership AUC and advantage (max TPR − FPR), age MAE gain over predicting the mean. Efficiency: bytes per client per round, wall time. |
| **Decision gate (P-U-E)** | A defence **passes** if the primary attack advantage is ≤ 5 pp **and** UAR drops by ≤ 5 points versus undefended FedAvg (proposed, pending supervisor confirmation). |

Out of scope here: malicious (active) servers that modify the model, gradient-inversion audio reconstruction, and colluding clients. Sections that the data cannot support (no emotion labels, single-gender corpus, too few speakers) are skipped with a message rather than reporting misleading numbers.'''

LOAD = r'''meta_raw = load_dataset()
meta_raw.to_csv(OUT_DIR / f"{DATASET}_metadata_raw.csv", index=False)
meta = meta_raw.copy()

print(f"Rows (utterances): {meta.shape[0]:,}")
print(f"Columns:           {meta.shape[1]}")
print(f"Total cells:       {meta.size:,}")
print(f"Total missing (NaN) values: {int(meta.isna().sum().sum()):,}  ({100 * meta.isna().sum().sum() / meta.size:.2f}% of cells)")
display(meta.head())'''

INSPECT_COLUMNS = r'''# Column overview: data type, missing values, unique values, an example
overview = pd.DataFrame({
    "dtype": meta.dtypes.astype(str),
    "non_null": meta.notna().sum(),
    "NaN": meta.isna().sum(),
    "NaN_%": (meta.isna().mean() * 100).round(2),
    "unique": meta.nunique(dropna=True),
    "example": [meta[c].dropna().iloc[0] if meta[c].notna().any() else None for c in meta.columns],
})
save_table(overview, "column_overview")'''

INSPECT_DETAIL = r'''# Detailed inspection of every column
for c in meta.columns:
    s = meta[c]
    print(f"\n=== {c}  | dtype={s.dtype} | NaN={s.isna().sum():,} | unique={s.nunique():,}")
    if c == "path":
        sample_paths = s.sample(min(len(s), 200), random_state=SEED)
        print("   example:", s.iloc[0])
        print(f"   files present on disk (checked {len(sample_paths)}): {sample_paths.map(lambda p: Path(p).exists()).mean():.1%}")
        print("   file extensions:", s.map(lambda p: Path(p).suffix.lower()).value_counts().to_dict())
    elif pd.api.types.is_numeric_dtype(s):
        print(s.describe().round(2).to_string())
    else:
        print(s.value_counts(dropna=False).head(15).to_string())

# The 'extra' field holds key=value pairs (session, intensity, sentence, ...): expand it into columns
extra_df = pd.DataFrame([dict(kv.split("=", 1) for kv in str(e).split(";") if "=" in kv) for e in meta.extra.fillna("")],
                        index=meta.index)
if len(extra_df.columns):
    print("\nParsed 'extra' fields:")
    display(extra_df.describe(include="all").T)

print("\nDuplicate paths:", int(meta.path.duplicated().sum()),
      "| duplicate file names:", int(meta.path.map(lambda p: Path(p).name.lower()).duplicated().sum()))'''

SPEAKERS = r'''HAS_EMO = meta.emotion.notna().any()
HAS_GENDER = meta.gender.dropna().nunique() > 1
HAS_AGE = meta.age.notna().any() and meta.drop_duplicates("speaker").age.nunique() > 1
print(f"{DATASET}: {len(meta):,} utterances, {meta.speaker.nunique()} {SPEAKER_LABEL}s")
print("Gender (speakers):", meta.drop_duplicates("speaker").gender.value_counts(dropna=False).to_dict())
if HAS_EMO: print("Emotions:", meta.emotion.value_counts().to_dict())
print(f"Has emotion labels: {HAS_EMO} | both genders: {HAS_GENDER} | age labels: {HAS_AGE}")
speakers = meta.groupby("speaker").agg(gender=("gender", "first"), age=("age", "first"), utterances=("path", "size"))
display(speakers.describe(include="all").T)
save_table(speakers, "speakers")'''

ACOUSTIC = r'''import librosa, librosa.display, soundfile as sf
from IPython.display import Audio
from tqdm.auto import tqdm
from joblib import Parallel, delayed

def sample_meta(df, n, by="speaker"):
    """Sample n rows spreading evenly across speakers (then filling up at random)."""
    if n is None or n >= len(df): return df.reset_index(drop=True)
    per = max(1, n // df[by].nunique())
    s = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in df.groupby(by)])
    if len(s) < n:
        s = pd.concat([s, df.drop(s.index).sample(min(n - len(s), len(df) - len(s)), random_state=SEED)])
    return s.sample(min(n, len(s)), random_state=SEED).reset_index(drop=True)

def acoustic_row(p):
    r = dict(path=p, readable=False)
    try:
        i = sf.info(p); r.update(sr=i.samplerate, channels=i.channels, duration_s=i.duration)
    except Exception:
        try: r.update(sr=librosa.get_samplerate(p), channels=np.nan, duration_s=librosa.get_duration(path=p))
        except Exception: return r
    try:
        y, _ = librosa.load(p, sr=TARGET_SR, mono=True, duration=INSPECT_MAX_S)
    except Exception:
        return r
    r["readable"] = True
    if len(y) < 2048: return r
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    nonsilent = librosa.effects.split(y, top_db=30)
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=TARGET_SR, frame_length=2048, hop_length=512)
    n = min(len(f0), len(rms)); f0v = f0[:n][rms[:n] > 0.1 * rms.max()]   # frames with enough energy ~ voiced
    e = rms ** 2 + 1e-12
    r.update(rms_db=float(20 * np.log10(rms.mean() + 1e-9)), peak=float(np.abs(y).max()),
             clipping_pct=float((np.abs(y) >= 0.99).mean() * 100),
             silence_pct=float(100 * (1 - sum(b - a for a, b in nonsilent) / len(y))),
             zcr=float(librosa.feature.zero_crossing_rate(y)[0].mean()),
             spec_centroid=float(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR)[0].mean()),
             spec_bandwidth=float(librosa.feature.spectral_bandwidth(y=y, sr=TARGET_SR)[0].mean()),
             spec_rolloff=float(librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR)[0].mean()),
             f0_median=float(np.median(f0v)) if len(f0v) else np.nan,
             f0_std=float(np.std(f0v)) if len(f0v) else np.nan,
             snr_db=float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10))))
    return r

insp_meta = sample_meta(meta.drop_duplicates("path"), N_INSPECT)
ac_cache = OUT_DIR / "features" / f"acoustic_n{N_INSPECT}_s{SEED}.csv"
if ac_cache.exists():
    acoustic = pd.read_csv(ac_cache)
else:
    acoustic = pd.DataFrame(Parallel(n_jobs=-1)(delayed(acoustic_row)(p) for p in tqdm(insp_meta.path, desc="acoustic table")))
    acoustic.to_csv(ac_cache, index=False)
df = insp_meta.drop(columns=[c for c in acoustic.columns if c in insp_meta and c != "path"]).merge(acoustic, on="path", how="left")
print(f"Acoustic table: {df.shape[0]:,} rows x {df.shape[1]} columns ({len(insp_meta):,} of {len(meta):,} files analysed)")
display(pd.DataFrame({"dtype": df.dtypes.astype(str), "NaN": df.isna().sum(), "NaN_%": (df.isna().mean() * 100).round(2)}).T)
df.head()'''

DESCRIPTIVE = r'''from scipy import stats
NUM = [c for c in ["duration_s", "sr", "rms_db", "peak", "clipping_pct", "silence_pct", "zcr", "spec_centroid",
                   "spec_bandwidth", "spec_rolloff", "f0_median", "f0_std", "snr_db", "age"]
       if c in df and df[c].notna().any() and df[c].nunique() > 1]
d = df[NUM].astype(float)
desc = pd.DataFrame({
    # central tendency
    "count": d.count(), "mean": d.mean(), "median": d.median(), "mode": d.round(2).mode().iloc[0],
    # dispersion
    "std": d.std(), "variance": d.var(), "min": d.min(), "Q1": d.quantile(.25), "Q3": d.quantile(.75),
    "IQR": d.quantile(.75) - d.quantile(.25), "max": d.max(), "range": d.max() - d.min(),
    "CV_%": 100 * d.std() / d.mean().abs(), "MAD": (d - d.median()).abs().median(),
    # shape
    "skewness": d.skew(), "kurtosis": d.kurt(),
}).round(3)
print("Descriptive statistics of the acoustic / numeric features")
display(save_table(desc, "descriptive_statistics"))

print("Categorical variables (mode and frequency)")
cat = pd.DataFrame({c: {"unique": meta[c].nunique(), "mode": meta[c].mode().iloc[0] if meta[c].notna().any() else None,
                        "mode_freq": int(meta[c].value_counts().iloc[0]) if meta[c].notna().any() else 0}
                    for c in ["emotion", "speaker", "gender", "corpus"] if c in meta}).T
display(cat)

key = [c for c in ["duration_s", "rms_db", "f0_median", "spec_centroid", "snr_db"] if c in NUM]
for by, ok in [("emotion", HAS_EMO), ("gender", HAS_GENDER)]:
    if ok and key:
        print(f"Mean / median / std by {by}")
        display(save_table(df.groupby(by)[key].agg(["mean", "median", "std"]).round(2), f"stats_by_{by}"))'''

OUTLIERS = r'''rows, flags = [], pd.DataFrame(index=df.index)
for c in NUM:
    x = df[c].astype(float); q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (x - x.mean()) / (x.std(ddof=0) + 1e-12)
    flags[c] = (x < lo) | (x > hi)
    rows.append(dict(feature=c, lower_fence=lo, upper_fence=hi, n_outliers_IQR=int(flags[c].sum()),
                     pct_IQR=100 * flags[c].mean(), n_outliers_z3=int((z.abs() > 3).sum())))
out_tab = pd.DataFrame(rows).set_index("feature").round(3)
display(save_table(out_tab, "outliers"))

ncol = 4; nrow = int(np.ceil(len(NUM) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.2 * nrow), squeeze=False)
for ax, c in zip(axes.flat, NUM):
    ax.boxplot(df[c].dropna(), vert=True, showfliers=True); ax.set_title(c); ax.set_xticks([])
for ax in list(axes.flat)[len(NUM):]: ax.axis("off")
savefig("outlier_boxplots")

df["n_outlier_features"] = flags.sum(1)
print("Files that are outliers on the most features (inspect / listen before deciding):")
display(df.sort_values("n_outlier_features", ascending=False)
          .head(10)[["path", "speaker", "emotion", "n_outlier_features"] + [c for c in ["duration_s", "rms_db", "silence_pct", "clipping_pct", "f0_median"] if c in df]])
print("Outliers are only flagged here. Natural variation (e.g. long or loud emotional utterances) is kept; "
      "only clear recording problems are removed in the cleaning step.")'''

CLEANING = r'''clean, log = meta.copy(), []
def drop(step, mask):
    global clean
    mask = mask.fillna(False).astype(bool)
    log.append(dict(step=step, removed=int(mask.sum()), remaining=int((~mask).sum())))
    clean = clean[~mask]

# 1. normalise text labels
clean["gender"] = clean.gender.astype(str).str.strip().str.lower().where(lambda g: g.isin(["female", "male"]))
if HAS_EMO: clean["emotion"] = clean.emotion.astype(str).str.strip().str.lower().replace({"nan": np.nan})
clean["speaker"] = clean.speaker.astype(str).str.strip()
# 2. rows that cannot be used
if HAS_EMO: drop("missing emotion label", clean.emotion.isna())
drop("duplicate path", clean.path.duplicated())
drop("duplicate file name", clean.path.map(lambda p: Path(p).name.lower()).duplicated())
# 3. audio problems (checked for the files in the acoustic table)
A = df.set_index("path")
drop("unreadable / corrupt audio", clean.path.map(A.readable).eq(False))
drop(f"shorter than {MIN_DURATION_S}s", clean.path.map(A.duration_s) < MIN_DURATION_S)
drop("almost silent (>95% silence)", clean.path.map(A.silence_pct) > 95)
drop("heavily clipped (>5% samples)", clean.path.map(A.clipping_pct) > 5)
# 4. classes too rare to evaluate
if HAS_EMO:
    rare = clean.emotion.value_counts()[lambda v: v < MIN_CLASS_COUNT].index
    drop(f"rare emotion classes (<{MIN_CLASS_COUNT} clips): {list(rare)}", clean.emotion.isin(rare))

display(save_table(pd.DataFrame(log), "cleaning_log"))
print(f"Before: {len(meta):,} rows | after: {len(clean):,} rows | removed {len(meta) - len(clean):,}")
if N_INSPECT is not None and len(meta) > N_INSPECT:
    print(f"Note: audio checks covered the {len(df):,} files in the acoustic table; set N_INSPECT = None to check every file.")
if clean.empty:
    raise ValueError("Cleaning removed every row: check the cleaning log above and relax the rules (e.g. MIN_CLASS_COUNT).")
meta = clean.reset_index(drop=True)'''

WRANGLING = r'''from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# Encodings
EMO_CLASSES = sorted(meta.emotion.dropna().unique()) if HAS_EMO else []
meta["emotion_id"] = meta.emotion.map({c: i for i, c in enumerate(EMO_CLASSES)})
meta["gender_id"] = meta.gender.map({"female": 0, "male": 1})

# Train / validation / test split, speaker-independent when there are enough speakers
SPEAKER_INDEPENDENT = meta.speaker.nunique() >= 4
def make_split(d):
    lab = d.emotion if HAS_EMO else d.speaker
    split = pd.Series("train", index=d.index)
    if SPEAKER_INDEPENDENT:
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(d, groups=d.speaker))
        split.iloc[te] = "test"
        rest = d.iloc[tr]
        if rest.speaker.nunique() >= 3:
            a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=SEED).split(rest, groups=rest.speaker))
            split.loc[rest.index[b]] = "val"
    else:
        tr, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED).split(d, lab.fillna("na")))
        split.iloc[te] = "test"
    return split
meta["split"] = make_split(meta)
if not SPEAKER_INDEPENDENT:
    print(f"Only {meta.speaker.nunique()} speakers: using an utterance-level split (NOT speaker-independent).")

tab = pd.crosstab(meta.emotion if HAS_EMO else meta.gender.fillna("unknown"), meta.split, margins=True)
display(save_table(tab, "split_distribution"))
print("Speakers per split:", meta.groupby("split").speaker.nunique().to_dict())
if HAS_EMO:
    cw = (len(meta) / (len(EMO_CLASSES) * meta.emotion.value_counts())).round(3)
    print("Class weights (balanced):", cw.to_dict())
meta.to_csv(OUT_DIR / f"{DATASET}_metadata_clean.csv", index=False)
print("Saved clean metadata with splits ->", OUT_DIR / f"{DATASET}_metadata_clean.csv")'''

PLOTS = r'''panels = [p for p, ok in [("emotion", HAS_EMO), ("speaker", True), ("emo_gender", HAS_EMO and HAS_GENDER), ("age", HAS_AGE)] if ok]
spk = meta.groupby("speaker").agg(gender=("gender", "first"), age=("age", "first"), utterances=("path", "size"))
fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False); axes = axes[0]
for ax, p in zip(axes, panels):
    if p == "emotion":
        meta.emotion.value_counts().plot.bar(ax=ax, color="tab:blue"); ax.set_title("Utterances per emotion")
    elif p == "speaker":
        top = spk.sort_values("utterances", ascending=False).head(40)
        top.utterances.plot.bar(ax=ax, color=top.gender.map({"female": "tab:red", "male": "tab:blue"}).fillna("grey").tolist())
        ax.set_title(f"Utterances per {SPEAKER_LABEL} (red F, blue M, grey unknown)"); ax.set_xticklabels([])
    elif p == "emo_gender":
        pd.crosstab(meta.emotion, meta.gender).plot.bar(ax=ax); ax.set_title("Emotion x gender")
    elif p == "age":
        spk.age.dropna().plot.hist(bins=15, ax=ax); ax.set_title("Speaker age")
    ax.tick_params(axis="x", rotation=45)
savefig("distributions")'''

PLOTS2 = r'''dfc = df[df.path.isin(meta.path)].merge(meta[["path", "split"]], on="path")
by = "emotion" if HAS_EMO else ("gender" if HAS_GENDER else None)
show = [c for c in ["duration_s", "rms_db", "f0_median", "spec_centroid", "silence_pct", "snr_db"] if c in NUM]
if by and show:
    groups = sorted(dfc[by].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), squeeze=False)
    for ax, c in zip(axes.flat, show):
        data = [dfc.loc[dfc[by] == g, c].dropna().values for g in groups]
        keep = [i for i, v in enumerate(data) if len(v) > 1]
        if not keep: ax.axis("off"); continue
        ax.violinplot([data[i] for i in keep], showmedians=True)
        ax.set_xticks(range(1, len(keep) + 1)); ax.set_xticklabels([groups[i] for i in keep], rotation=45); ax.set_title(f"{c} by {by}")
    for ax in list(axes.flat)[len(show):]: ax.axis("off")
    savefig(f"features_by_{by}")

if HAS_GENDER and "f0_median" in NUM:
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, col in [("female", "tab:red"), ("male", "tab:blue")]:
        ax.hist(dfc.loc[dfc.gender == g, "f0_median"].dropna(), bins=40, alpha=.6, color=col, label=g)
    ax.set_xlabel("median F0 (Hz)"); ax.legend(); ax.set_title("Pitch by gender: the main paralinguistic cue an attacker exploits")
    savefig("f0_by_gender")

corr = dfc[NUM].astype(float).corr(method="spearman")
fig, ax = plt.subplots(figsize=(1 + .7 * len(NUM), .8 + .6 * len(NUM)))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(NUM))); ax.set_xticklabels(NUM, rotation=60, ha="right"); ax.set_yticks(range(len(NUM))); ax.set_yticklabels(NUM)
for i in range(len(NUM)):
    for j in range(len(NUM)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("Spearman correlation of acoustic features"); savefig("correlation_heatmap")'''

STATS_TESTS = r'''# Inferential statistics: are labels and acoustic cues related?
res = []
if HAS_EMO and HAS_GENDER:
    ct = pd.crosstab(meta.emotion, meta.gender)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    res.append(dict(test="chi-square emotion x gender", statistic=chi2, p_value=p, effect=f"Cramér's V = {v:.3f}"))
for c in NUM:
    if HAS_EMO:
        g = [x[c].dropna().values for _, x in dfc.groupby("emotion") if x[c].notna().sum() > 2]
        if len(g) > 1:
            h, p = stats.kruskal(*g)
            eps2 = (h - len(g) + 1) / (sum(map(len, g)) - len(g))
            res.append(dict(test=f"Kruskal-Wallis {c} ~ emotion", statistic=h, p_value=p, effect=f"epsilon² = {eps2:.3f}"))
    if HAS_GENDER:
        a, b = dfc.loc[dfc.gender == "female", c].dropna(), dfc.loc[dfc.gender == "male", c].dropna()
        if len(a) > 2 and len(b) > 2:
            u, p = stats.mannwhitneyu(a, b)
            res.append(dict(test=f"Mann-Whitney {c} ~ gender", statistic=u, p_value=p, effect=f"rank-biserial r = {1 - 2 * u / (len(a) * len(b)):.3f}"))
tests = pd.DataFrame(res)
if len(tests):
    tests["significant (p<0.05)"] = tests.p_value < 0.05
    display(save_table(tests.round(4), "statistical_tests"))
print("A feature that differs strongly by gender (large effect) is a likely leakage channel even if it helps emotion recognition.")'''

EXAMPLES = r'''def load_audio(path, sr=TARGET_SR, max_s=None):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_s)
    return y

def show_example(row):
    y = load_audio(row.path, max_s=10)
    fig, ax = plt.subplots(1, 2, figsize=(14, 3))
    librosa.display.waveshow(y, sr=TARGET_SR, ax=ax[0]); ax[0].set_title(f"{row.emotion} | {row.speaker} | {row.gender}")
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64), ref=np.max)
    librosa.display.specshow(S, sr=TARGET_SR, x_axis="time", y_axis="mel", ax=ax[1]); ax[1].set_title("log-mel")
    plt.show(); display(Audio(y, rate=TARGET_SR))

ex_by = "emotion" if HAS_EMO else "speaker"
for _, row in meta.dropna(subset=[ex_by]).groupby(ex_by).sample(1, random_state=SEED).head(8).iterrows():
    show_example(row)'''

FEATURES = r'''work = sample_meta(meta, N_SAMPLE)
print(len(work), "utterances,", work.speaker.nunique(), f"{SPEAKER_LABEL}s; split:", work.split.value_counts().to_dict())

def preprocess(path):
    y = load_audio(path)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = y / (np.abs(y).max() + 1e-8)
    return y[: int(MAX_SECONDS * TARGET_SR)]

def handcrafted(path):
    y = preprocess(path)
    mf = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=40)
    mf = np.vstack([mf, librosa.feature.delta(mf)])
    lm = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=64))
    return np.concatenate([mf.mean(1), mf.std(1)]), np.concatenate([lm.mean(1), lm.std(1)])

tag = f"n{N_SAMPLE}_s{SEED}_c{len(meta)}"
f_cache = OUT_DIR / "features" / f"handcrafted_{tag}.npz"
if f_cache.exists():
    z = np.load(f_cache); X_mfcc, X_mel = z["mfcc"], z["mel"]
else:
    out = Parallel(n_jobs=-1)(delayed(handcrafted)(p) for p in tqdm(work.path, desc="MFCC/log-mel"))
    X_mfcc, X_mel = np.array([a for a, _ in out]), np.array([b for _, b in out])
    np.savez(f_cache, mfcc=X_mfcc, mel=X_mel)
work.to_csv(OUT_DIR / "features" / f"work_meta_{tag}.csv", index=False)
FEATS = {"MFCC": X_mfcc, "log-mel": X_mel}
{k: v.shape for k, v in FEATS.items()}'''

SSL = r'''try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
if RUN_SSL and HAS_TORCH:
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
    print("SSL embeddings skipped (RUN_SSL = False or PyTorch missing): using handcrafted features only")'''

EMBED_VIZ = r'''from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
fname = list(FEATS)[-1]
P = PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(FEATS[fname]))
Z = P.transform(StandardScaler().fit_transform(FEATS[fname]))
cols = [c for c, ok in [("emotion", HAS_EMO), ("gender", HAS_GENDER), ("speaker", True)] if ok]
fig, axes = plt.subplots(1, len(cols), figsize=(6.5 * len(cols), 5), squeeze=False)
for ax, c in zip(axes[0], cols):
    labs = work[c].fillna("unknown").astype(str); u = labs.unique()
    cmap = plt.get_cmap("tab20" if len(u) > 10 else "tab10")
    for i, l in enumerate(u[:20]):
        m = (labs == l).values; ax.scatter(Z[m, 0], Z[m, 1], s=8, color=cmap(i % 20), label=l)
    ax.set_title(f"PCA of {fname} coloured by {c}"); ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]:.0%})")
    if len(u) <= 12: ax.legend(markerscale=2, fontsize=8)
savefig("pca_embeddings")
print("If gender or speaker clusters are clearer than emotion clusters, the representation leaks more identity than task information.")'''

ML = r'''from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, accuracy_score, mean_absolute_error,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)

MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
    "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC(C=3, gamma="scale", class_weight="balanced")),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=SEED)),
}
if HAS_EMO:
    tr = work.split.isin(["train", "val"]).values; te = (work.split == "test").values
    # integer labels: some sklearn versions (e.g. MLP with early stopping) fail on string labels
    y = work.emotion.map({c: i for i, c in enumerate(EMO_CLASSES)}).astype(int).values
    res, best = [], None
    for fname, X in FEATS.items():
        for mname, mk in MODELS.items():
            t0 = time.time(); m = mk().fit(X[tr], y[tr]); p = m.predict(X[te])
            r = dict(features=fname, model=mname, UAR=balanced_accuracy_score(y[te], p), macro_F1=f1_score(y[te], p, average="macro"),
                     accuracy=accuracy_score(y[te], p), train_s=time.time() - t0)
            res.append(r)
            if best is None or r["UAR"] > best[0]["UAR"]: best = (r, p)
    ser = pd.DataFrame(res).round(3)
    display(save_table(ser.pivot(index="model", columns="features", values="UAR"), "ser_uar_models"))
    display(ser.sort_values("UAR", ascending=False).head(8))
    r, p = best
    print(f"Best: {r['model']} on {r['features']}  UAR={r['UAR']:.3f}  macro-F1={r['macro_F1']:.3f}  "
          f"({'speaker-independent' if SPEAKER_INDEPENDENT else 'utterance-level'} test, chance UAR = {1/len(EMO_CLASSES):.3f})")
    K = list(range(len(EMO_CLASSES)))
    print(classification_report(y[te], p, labels=K, target_names=EMO_CLASSES, zero_division=0))
    cm = np.nan_to_num(confusion_matrix(y[te], p, labels=K, normalize="true"))
    fig, ax = plt.subplots(figsize=(1.2 + .8 * len(EMO_CLASSES), 1 + .7 * len(EMO_CLASSES)))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMO_CLASSES))); ax.set_xticklabels(EMO_CLASSES, rotation=45); ax.set_yticks(range(len(EMO_CLASSES))); ax.set_yticklabels(EMO_CLASSES)
    for i in range(len(EMO_CLASSES)):
        for j in range(len(EMO_CLASSES)): ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion matrix (recall), {r['model']} / {r['features']}")
    savefig("confusion_matrix")
else:
    print("No emotion labels in this dataset: SER models skipped.")'''

ATTACK_REPR = r'''# A1. Attribute inference on shared representations: an attacker who sees embeddings tries to recover gender / speaker / age
def clf(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))

def split_idx(y, groups, speaker_independent, test_size=0.25):
    """Split with >= 2 classes on both sides and test classes seen in training (tries several seeds); None if impossible."""
    for s in range(SEED, SEED + 30):
        if speaker_independent:
            a, b = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(y, y, groups))
        else:
            a, b = next(StratifiedShuffleSplit(1, test_size=test_size, random_state=s).split(y, y))
        if len(set(y[a])) > 1 and len(set(y[b])) > 1 and set(y[b]) <= set(y[a]):
            return a, b
    return None

def eval_task(X, y, groups, speaker_independent=True):
    y = np.asarray(y); sp = split_idx(y, np.asarray(groups), speaker_independent)
    if sp is None: return None
    a, b = sp; p = clf().fit(X[a], y[a]).predict(X[b])
    maj = pd.Series(y[a]).mode()[0]; acc, base = accuracy_score(y[b], p), accuracy_score(y[b], np.full(len(b), maj))
    uar = balanced_accuracy_score(y[b], p)
    return dict(UAR=uar, chance_UAR=1 / len(set(y[b])), advantage_UAR=uar - 1 / len(set(y[b])), acc=acc, majority_acc=base, advantage_acc=acc - base)

res = []
for fname, X in FEATS.items():
    tasks = []
    if HAS_GENDER:
        ok = work.gender.notna().values; tasks.append(("gender (unseen speakers)", ok, work.gender.values, True))
    counts = work.speaker.map(work.speaker.value_counts())
    if work.speaker.nunique() > 1:
        tasks.append((f"{SPEAKER_LABEL} ID (closed set)", (counts >= 4).values, work.speaker.values, False))
    for task, ok, y, si in tasks:
        r = eval_task(X[ok], y[ok], work.speaker.values[ok], si)
        if r is None:
            print(f"skip {task} on {fname}: not enough speakers/classes for a valid split"); continue
        r.update(attack=task, features=fname); res.append(r)
    ok = work.age.notna().values
    if HAS_AGE and work[ok].speaker.nunique() >= 4:
        g = work[ok]; yv = g.age.values
        a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(g, groups=g.speaker))
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(X[ok][a], yv[a])
        mae, base = mean_absolute_error(yv[b], m.predict(X[ok][b])), mean_absolute_error(yv[b], np.full(len(b), yv[a].mean()))
        res.append(dict(attack="age (unseen speakers, MAE years)", features=fname, MAE=mae, baseline_MAE=base, MAE_gain=base - mae))
probe = pd.DataFrame(res)
cols = [c for c in ["attack", "features", "UAR", "chance_UAR", "advantage_UAR", "acc", "majority_acc", "advantage_acc", "MAE", "baseline_MAE", "MAE_gain"] if c in probe]
display(save_table(probe[cols].round(3), "attack_representations") if len(probe) else probe)'''

FL_SETUP = r'''# Federated setup: clients = speakers, frozen features -> small MLP trained with FedAvg
FL_OK = HAS_EMO and HAS_TORCH
if not FL_OK:
    print("Federated learning sections skipped:", "no emotion labels" if not HAS_EMO else "PyTorch not installed")
else:
    import torch, torch.nn as nn, copy
    torch.manual_seed(SEED)
    FL_FEAT = list(FEATS)[-1]                                  # SSL embedding if available, else log-mel
    Xn = FEATS[FL_FEAT].astype(np.float32)
    trmask = work.split.isin(["train", "val"]).values
    mu, sd = Xn[trmask].mean(0), Xn[trmask].std(0) + 1e-6
    XT = torch.tensor((Xn - mu) / sd)
    YT = torch.tensor(work.emotion.map({c: i for i, c in enumerate(EMO_CLASSES)}).values)
    GT = torch.tensor(work.gender.map({"female": 0, "male": 1}).fillna(-1).astype(int).values)
    test_idx = np.where(work.split == "test")[0]

    # each training speaker keeps 20% of its clips out of training: these are the NON-members for membership inference
    rng = np.random.default_rng(SEED); clients, nonmember_idx = [], []
    train_spk = sorted(work.loc[trmask, "speaker"].unique())
    chunks = max(1, int(np.ceil(MIN_CLIENTS / max(1, len(train_spk)))))
    for s in train_spk:
        idx = np.where((work.speaker == s).values & trmask)[0]; rng.shuffle(idx)
        k = max(1, int(round(0.2 * len(idx)))) if len(idx) >= 5 else 0
        nonmember_idx += list(idx[:k]); mem = idx[k:]
        for j, part in enumerate(np.array_split(mem, min(chunks, max(1, len(mem) // 4)))):
            if len(part):
                clients.append(dict(cid=f"{s}#{j}", speaker=s, gender=work.gender.iloc[part[0]], age=work.age.iloc[part[0]], idx=part))
    member_idx = np.concatenate([c["idx"] for c in clients])
    nonmember_idx = np.array(nonmember_idx, dtype=int)
    print(f"FL features: {FL_FEAT} {tuple(XT.shape)} | {len(clients)} clients from {len(train_spk)} training speakers "
          f"| test clips ({'unseen speakers' if SPEAKER_INDEPENDENT else 'utterance-level split'}): {len(test_idx)} | members {len(member_idx)} / non-members {len(nonmember_idx)}")

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
        @staticmethod
        def backward(ctx, g): return -ctx.lam * g, None

    class Net(nn.Module):
        def __init__(self, d, k, h=128):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1))
            self.head = nn.Linear(h, k)          # emotion classifier
            self.adv = nn.Linear(h, 2)           # gender adversary (only trained with gradient reversal)
        def forward(self, x): z = self.enc(x); return self.head(z), z

    def flat(m): return torch.cat([p.data.view(-1) for p in m.parameters()])
    def unflat(m, v): torch.nn.utils.vector_to_parameters(v, m.parameters())
    base_model = Net(XT.shape[1], len(EMO_CLASSES))
    N_PARAMS = flat(base_model).numel()
    # the server-side attacks look at the emotion-head + encoder-bias part of each update (small, fast)
    _sizes = [p.numel() for p in base_model.parameters()]
    _off = np.cumsum([0] + _sizes)
    ATTACK_SLICE = np.r_[_off[1]:_off[2], _off[2]:_off[4]]          # enc bias, head weight, head bias
    print(f"Model parameters: {N_PARAMS:,}; attack features per update: {len(ATTACK_SLICE):,}")'''

PRIVACY_ACCOUNTING = r'''# Privacy accounting (Rényi DP of the Gaussian mechanism, composed over rounds, converted to (epsilon, delta))
def eps_gaussian(z, rounds, delta=DELTA, sens=1.0):
    """epsilon after `rounds` Gaussian mechanisms with noise multiplier z (std = z * clip) and L2 sensitivity sens * clip."""
    alphas = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 512)])
    rdp = rounds * alphas * sens ** 2 / (2 * z ** 2)
    return float(np.min(rdp + np.log(1 / delta) / (alphas - 1)))

def z_for_eps(eps, rounds, sens):
    lo, hi = 1e-3, 1e4
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if eps_gaussian(mid, rounds, sens=sens) > eps else (lo, mid)
    return hi

if FL_OK:
    # Local DP (server sees each noisy update): replacing a client's data can move its clipped update by 2*CLIP.
    # Worst case: the client takes part in every round.
    Z_LOCAL = {e: z_for_eps(e, ROUNDS, sens=2.0) for e in DP_EPSILONS}
    # Distributed DP with SecAgg (server sees only the noisy sum): adding/removing one client moves the sum by CLIP.
    Z_DIST = {e: z_for_eps(e, ROUNDS, sens=1.0) for e in DP_EPSILONS}
    display(pd.DataFrame({"local DP noise multiplier z": Z_LOCAL, "distributed DP (SecAgg) z on the sum": Z_DIST}).rename_axis("target ε").round(2))
    print("Local DP needs far more noise for the same ε because every single update is exposed to the server; "
          "SecAgg lets the noise be shared across clients. (No subsampling amplification is claimed, so these ε are conservative.)")'''

FL_RUN = r'''def run_fl(defence="none", eps=None, lam=0.0, seed=SEED, verbose=False):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = copy.deepcopy(base_model); hist, records = [], []
    t0 = time.time(); bytes_up = 0
    for rnd in range(ROUNDS):
        chosen = rng.choice(len(clients), max(2, int(CLIENT_FRAC * len(clients))), replace=False)
        w0 = flat(model).clone(); sent, sizes = [], []
        for ci in chosen:
            c = clients[ci]; idx = torch.tensor(c["idx"])
            local = copy.deepcopy(model); local.train()
            opt = torch.optim.SGD(local.parameters(), lr=LR, momentum=0.9)
            for _ in range(LOCAL_EPOCHS):
                opt.zero_grad(); logits, z = local(XT[idx])
                loss = nn.functional.cross_entropy(logits, YT[idx])
                if defence == "grl" and (GT[idx] >= 0).all():
                    loss = loss + nn.functional.cross_entropy(local.adv(GradReverse.apply(z, lam)), GT[idx])
                loss.backward(); opt.step()
            d = flat(local) - w0
            if defence in ("ldp", "secagg_ddp"):
                d = d * min(1.0, CLIP / (d.norm().item() + 1e-12))
            if defence == "ldp":
                d = d + torch.randn_like(d) * Z_LOCAL[eps] * CLIP
            if defence == "secagg_ddp":
                d = d + torch.randn_like(d) * Z_DIST[eps] * CLIP / np.sqrt(len(chosen))   # shares sum to std z*CLIP
            sent.append(d); sizes.append(len(c["idx"]))
        # weighting: FedAvg weights by data size; DP variants use a plain mean so the sensitivity analysis holds
        weights = np.array(sizes, float) / sum(sizes) if defence in ("none", "grl", "secagg") else np.full(len(sent), 1 / len(sent))
        contrib = [d * w for d, w in zip(sent, weights)]
        if defence.startswith("secagg"):
            # pairwise additive masks: client i adds m_ij, client j subtracts it -> masks cancel in the sum
            masked = [v.clone() for v in contrib]
            for a in range(len(masked)):
                for b in range(a + 1, len(masked)):
                    g = torch.Generator().manual_seed(hash((seed, rnd, a, b)) % (2 ** 31))
                    m = torch.randn(N_PARAMS, generator=g) * 10.0
                    masked[a] += m; masked[b] -= m
            agg = sum(masked)
            assert torch.allclose(agg, sum(contrib), atol=1e-3), "SecAgg masks did not cancel"
            observed = masked
            bytes_up += len(chosen) * (N_PARAMS * 4 + 32 * (len(chosen) - 1))   # update + one key/seed per peer
        else:
            agg = sum(contrib); observed = sent
            bytes_up += len(chosen) * N_PARAMS * 4
        for ci, v in zip(chosen, observed):     # what the honest-but-curious server can store
            c = clients[ci]
            records.append(dict(round=rnd, cid=c["cid"], speaker=c["speaker"], gender=c["gender"], age=c["age"],
                                u=v.numpy()[ATTACK_SLICE].copy()))
        unflat(model, w0 + agg)
        model.eval()
        with torch.no_grad(): p = model(XT[test_idx])[0].argmax(1).numpy()
        hist.append(balanced_accuracy_score(YT[test_idx].numpy(), p))
    with torch.no_grad(): p = model(XT[test_idx])[0].argmax(1).numpy()
    yt = YT[test_idx].numpy()
    util = dict(UAR=balanced_accuracy_score(yt, p), macro_F1=f1_score(yt, p, average="macro"), accuracy=accuracy_score(yt, p),
                bytes_per_client_round=bytes_up / (ROUNDS * max(2, int(CLIENT_FRAC * len(clients)))), time_s=time.time() - t0)
    return model, hist, records, util

if FL_OK:
    fl_model, fl_hist, fl_records, fl_util = run_fl("none")
    print("Undefended FedAvg:", {k: round(v, 3) for k, v in fl_util.items()})
    plt.plot(fl_hist); plt.xlabel("round"); plt.ylabel("emotion UAR (test speakers)"); plt.title(f"FedAvg on {DATASET}")
    savefig("fedavg_curve")'''

FL_ATTACKS = r'''from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

def attack_suite(records, model):
    """Paralinguistic inference attacks run by the honest-but-curious server."""
    R = pd.DataFrame(records); U = np.stack(R.u.values); out = {}
    # A2. gender inference from individual updates, generalising to unseen speakers (grouped CV)
    g = R.gender.values; okg = pd.notna(g)
    per_gender = R[okg].groupby("gender").speaker.nunique()
    if HAS_GENDER and len(per_gender) == 2 and per_gender.min() >= 2:
        k = min(5, R[okg].speaker.nunique())
        p = cross_val_predict(clf(), U[okg], g[okg], groups=R.speaker.values[okg], cv=GroupKFold(k))
        out["gender_UAR"] = balanced_accuracy_score(g[okg], p); out["gender_adv"] = out["gender_UAR"] - 0.5
    # A3. age inference from updates (regression, grouped CV)
    oka = R.age.notna().values
    if HAS_AGE and R[oka].speaker.nunique() >= 4:
        k = min(5, R[oka].speaker.nunique()); ya = R.age.values[oka].astype(float)
        p = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=10)), U[oka], ya, groups=R.speaker.values[oka], cv=GroupKFold(k))
        base = np.abs(ya - ya.mean()).mean(); out["age_MAE"] = mean_absolute_error(ya, p); out["age_MAE_gain"] = base - out["age_MAE"]
    # A4. speaker linkage: link a new update to a client seen in earlier rounds (closed set)
    vc = R.cid.value_counts(); okl = R.cid.isin(vc[vc >= 3].index).values
    if okl.sum() > 10 and R.cid[okl].nunique() > 1:
        yl = R.cid.values[okl]
        p = cross_val_predict(clf(), U[okl], yl, cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
        out["linkage_acc"] = accuracy_score(yl, p)
        out["linkage_adv"] = out["linkage_acc"] - pd.Series(yl).value_counts(normalize=True).max()
    # A5. membership inference on the final global model (loss-threshold attack)
    if len(nonmember_idx) >= 5:
        model.eval()
        with torch.no_grad():
            def losses(ix): return nn.functional.cross_entropy(model(XT[ix])[0], YT[ix], reduction="none").numpy()
            lm, ln = losses(torch.tensor(member_idx)), losses(torch.tensor(nonmember_idx))
        score = -np.concatenate([lm, ln]); lab = np.r_[np.ones(len(lm)), np.zeros(len(ln))]
        fpr, tpr, _ = roc_curve(lab, score)
        out["MIA_AUC"] = roc_auc_score(lab, score); out["MIA_adv"] = float(np.max(tpr - fpr))
    return out

if FL_OK:
    base_attacks = attack_suite(fl_records, fl_model)
    print("Attacks on undefended FedAvg (honest-but-curious server):")
    for k, v in base_attacks.items(): print(f"  {k:14s} {v:.3f}")
    print("Advantage = attack score minus chance/majority. MIA_AUC 0.5 = no membership leakage.")'''

DEFENCES = r'''if FL_OK:
    configs = [("FedAvg (no defence)", "none", None, 0.0)]
    configs += [(f"Local DP ε={e}", "ldp", e, 0.0) for e in DP_EPSILONS]
    configs += [("SecAgg", "secagg", None, 0.0)]
    configs += [(f"SecAgg + distributed DP ε={e}", "secagg_ddp", e, 0.0) for e in DP_EPSILONS]
    if HAS_GENDER:
        configs += [(f"Gradient reversal λ={l}", "grl", None, l) for l in GRL_LAMBDAS]
    rows, curves = [], {}
    for name, dfn, eps, lam in tqdm(configs, desc="FL configurations"):
        if dfn == "none":
            model, hist, recs, util = fl_model, fl_hist, fl_records, fl_util
        else:
            model, hist, recs, util = run_fl(dfn, eps, lam)
        att = attack_suite(recs, model)
        rows.append(dict(config=name, defence=dfn, epsilon=eps if eps else np.inf, lam=lam, **util, **att)); curves[name] = hist
    pue = pd.DataFrame(rows).set_index("config")
    base_uar = pue.loc["FedAvg (no defence)", "UAR"]
    pue["UAR_drop"] = base_uar - pue.UAR
    PRIMARY = "gender_adv" if "gender_adv" in pue else ("linkage_adv" if "linkage_adv" in pue else "MIA_adv")
    if PRIMARY not in pue: pue[PRIMARY] = np.nan
    pue["primary_attack"] = PRIMARY
    pue["gate_pass"] = (pue[PRIMARY].fillna(0) <= GATE_ADV) & (pue.UAR_drop <= GATE_UAR)
    cols = [c for c in ["epsilon", "UAR", "macro_F1", "UAR_drop", "gender_UAR", "gender_adv", "age_MAE_gain", "linkage_acc", "linkage_adv",
                        "MIA_AUC", "MIA_adv", "bytes_per_client_round", "time_s", "gate_pass"] if c in pue]
    print(f"Privacy-Utility-Efficiency table. Primary attack for the gate: {PRIMARY}. "
          f"Gate: advantage <= {GATE_ADV:.0%} and UAR drop <= {GATE_UAR:.0%}.")
    display(save_table(pue[cols].round(3), "pue_table"))'''

PUE_PLOTS = r'''if FL_OK:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for name, h in curves.items(): ax[0].plot(h, label=name)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("emotion UAR"); ax[0].set_title("Utility over rounds"); ax[0].legend(fontsize=7)
    for name, r in pue.iterrows():
        ax[1].scatter(r[PRIMARY] if pd.notna(r[PRIMARY]) else 0, r.UAR, s=60, color="tab:green" if r.gate_pass else "tab:red")
        ax[1].annotate(name, (r[PRIMARY] if pd.notna(r[PRIMARY]) else 0, r.UAR), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax[1].axvline(GATE_ADV, ls="--", c="grey"); ax[1].axhline(base_uar - GATE_UAR, ls="--", c="grey")
    ax[1].set_xlabel(f"{PRIMARY} (lower = more private)"); ax[1].set_ylabel("emotion UAR (higher = more useful)")
    ax[1].set_title("Privacy-utility trade-off (green = passes the gate)")
    savefig("privacy_utility")

    print("Summary")
    print(f"  Undefended FedAvg UAR = {base_uar:.3f}; {PRIMARY} = {pue.loc['FedAvg (no defence)', PRIMARY]:.3f}")
    passed = pue[pue.gate_pass & (pue.defence != "none")]
    print("  Defences passing the gate:", ", ".join(passed.index) if len(passed) else "none at these settings")
    print("  Note: SecAgg hides individual updates (per-client attacks fall to chance) but not the aggregate or the final model;"
          " membership leakage is reduced only by DP noise.")'''

NEXT = r'''## Notes and next steps

* Everything is saved in `OUT_DIR`: `tables/` (column overview, descriptive statistics, outliers, cleaning log, statistical tests, model results, attack results, P-U-E table), `figures/`, cached `features/`, and the raw and clean metadata CSVs (same columns across datasets, so they can be concatenated).
* Start with the default `N_SAMPLE` / `N_INSPECT`; raise them (or set `None`) once everything runs. Cached features are reused.
* Results on very small or single-gender corpora (SAVEE, TESS, MSP-IMPROV session 1) are indicative only; several sections skip themselves when the data cannot support them.
* The ε values are conservative (no subsampling amplification). For a tighter accountant, replace `eps_gaussian` with Opacus' RDP accountant.
* Not yet covered: PEFT (LoRA / adapters) on the SSL encoder, gradient-inversion reconstruction, active (malicious) servers, and the multimodal arm (RQ3). For cross-corpus experiments use the combined notebook `notebooks/SER_Privacy_Dataset_Prep.ipynb`.'''
