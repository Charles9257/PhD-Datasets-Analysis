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

"MSP-PODCAST": dict(
    title="MSP-Podcast",
    blurb="MSP-Podcast: natural, spontaneous emotional speech segmented from podcast recordings (UT Dallas). Your folder `MSP-POSCAST/Audios/Audios` holds 58,899 wav segments named `MSP-PODCAST_<podcast>_<segment>.wav`.",
    caveats="* **No label files are in the folder yet.** Without them the notebook runs in *audio-only* mode: durations, examples and features work, but emotion, gender and true speaker probes are skipped. The podcast number from the file name is used as a **proxy grouping** (one podcast can contain several speakers, and a speaker can appear in several podcasts), so the closed-set probe is labelled *podcast source*, not speaker.\n* When you get the labels from UT Dallas, put `labels_consensus.csv` (columns `FileName, EmoClass, EmoAct, EmoVal, EmoDom, SpkrID, Gender, Split_Set`) in `MSP-POSCAST/Labels/`. The notebook then picks up emotion (A, S, H, U, F, D, C, N, O; `X` = no agreement, dropped), gender and speaker IDs automatically. Utterances with unknown speaker are grouped per podcast.\n* Large corpus: listing 58k files on Drive takes a minute or two; keep `N_SAMPLE` small at first.\n* Arousal / valence / dominance scores are kept in `extra` when labels are present.",
    paths='''MSPP_ROOT = next((DATA_ROOT / n for n in ["MSP-POSCAST", "MSP-PODCAST", "MSP-Podcast"] if (DATA_ROOT / n).exists()),
                 DATA_ROOT / "MSP-POSCAST")
LABEL_CANDIDATES = [MSPP_ROOT / "Labels" / "labels_consensus.csv", MSPP_ROOT / "labels_consensus.csv",
                    MSPP_ROOT / "Labels" / "Labels" / "labels_consensus.csv"]
PATHS = {"MSPP_ROOT": MSPP_ROOT,
         "MSPP_LABELS": next((p for p in LABEL_CANDIDATES if p.exists()), LABEL_CANDIDATES[0])}
REQUIRED = ["MSPP_ROOT"]''',
    loader='''def load_dataset():
    global SPEAKER_LABEL
    wavs = {f.name: f for f in PATHS["MSPP_ROOT"].rglob("*.wav")}
    print(f"{len(wavs):,} wav files found")
    if PATHS["MSPP_LABELS"].exists():
        lab = pd.read_csv(PATHS["MSPP_LABELS"])
        m = {"A":"angry","S":"sad","H":"happy","U":"surprise","F":"fear","D":"disgust","C":"contempt","N":"neutral","O":"other"}
        lab = lab[lab.FileName.isin(wavs)]
        pod = lab.FileName.str.split("_").str[1]
        spk = lab.SpkrID.astype(str)
        unknown = spk.str.lower().isin(["unknown", "nan", "-1", ""])
        g = lab.Gender.astype(str).str.lower().map({"female": "female", "male": "male"})
        out = pd.DataFrame(dict(path=lab.FileName.map(lambda n: str(wavs[n])), corpus="MSP-Podcast",
                                emotion=lab.EmoClass.map(m), emotion_raw=lab.EmoClass,
                                speaker=np.where(unknown, "MSPP_unk_pod" + pod, "MSPP_" + spk), gender=g, age=np.nan,
                                extra="split=" + lab.Split_Set.astype(str) + ";act=" + lab.EmoAct.astype(str)
                                      + ";val=" + lab.EmoVal.astype(str) + ";dom=" + lab.EmoDom.astype(str)))
        print("Labels found:", PATHS["MSPP_LABELS"])
        return out.dropna(subset=["emotion"]).reset_index(drop=True)
    print("No labels_consensus.csv found: audio-only mode, grouping by podcast number (a proxy, not speaker identity).")
    SPEAKER_LABEL = "podcast source"
    names = sorted(wavs)
    return pd.DataFrame(dict(path=[str(wavs[n]) for n in names], corpus="MSP-Podcast", emotion=np.nan, emotion_raw=np.nan,
                             speaker=["MSPP_pod" + n.split("_")[1] for n in names], gender=np.nan, age=np.nan,
                             extra=["segment=" + n.split("_")[-1].split(".")[0] for n in names]))'''),

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

from nb_cells import *  # noqa: E402,F401  (shared notebook cells)


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)}


def colab_url(rel):
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel}"


def build(name, d):
    folder = name.replace(" ", "_")
    rel = f"notebooks/{folder}/{folder}_Explore.ipynb"
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(rel)})"
    cells = [
        md(f"{badge}\n\n# {d['title']}: full data analysis, SER models, federated learning, attacks and defences\n\n{d['blurb']}\n\n"
           f"**Dataset notes**\n{d['caveats']}\n\n"
           "**What this notebook covers** (this dataset only)\n"
           "1. Setup and configuration\n2. Threat model\n"
           "3. Data inspection: rows, columns, data types, missing values (NaN), detailed look at every column, duplicates\n"
           "4. Acoustic feature table: duration, sample rate, loudness, clipping, silence, pitch (F0), spectral shape, SNR\n"
           "5. Descriptive statistics: central tendency (mean, median, mode), dispersion (std, variance, IQR, range, CV, MAD), skewness, kurtosis\n"
           "6. Outliers: IQR and z-score rules, box plots, most extreme files\n"
           "7. Data cleaning (logged) and 8. data wrangling: label encoding, speaker-independent train/val/test split, class weights\n"
           "9. Analysis and visualisation: distributions, features by emotion and gender, correlations, statistical tests, audio examples\n"
           "10. Features (MFCC, log-mel, wav2vec2) and embedding visualisation\n"
           "11. Machine learning models: logistic regression, SVM, random forest, MLP (UAR, macro-F1, confusion matrix)\n"
           "12. Paralinguistic inference attacks on representations (gender, speaker, age)\n"
           "13. Federated learning (FedAvg, one client per speaker) and attacks by an honest-but-curious server: gender, age, speaker linkage, membership inference\n"
           "14. Defences: local DP (ε = 1, 3, 5, 8), Secure Aggregation, SecAgg + distributed DP, gradient reversal\n"
           "15. Evaluation: privacy-utility-efficiency (P-U-E) table and the 5 pp / 5 pt decision gate\n\n"
           "**Source data is never modified.** Outputs go to `OUT_DIR` (on Colab: `MyDrive/SER_outputs/"
           f"{name}`).\n\n**On Google Colab:** Runtime > Change runtime type > **T4 GPU**, then Runtime > Run all."),
        md("## 0. Environment setup (Colab or local)\n\nMounts Google Drive on Colab, installs missing packages and checks for a GPU."),
        code(SETUP),
        md("## 1. Configuration\n\n`DATA_ROOT` should mirror your local `Datasets` folder (default on Colab: `MyDrive/Datasets`). "
           "All knobs (sample sizes, FL rounds, ε grid, gate thresholds) are here."),
        code(CONFIG.replace("__NAME__", name).replace("__PATHS__", d["paths"])),
        md(THREAT_MODEL),
        md("## 3. Data inspection\n\n### 3.1 Load the metadata table\nOne row per utterance: `path, corpus, emotion, emotion_raw, speaker, gender, age, extra`."),
        code(d["loader"]),
        code(LOAD),
        md("### 3.2 Columns: data types, missing values, unique values"),
        code(INSPECT_COLUMNS),
        md("### 3.3 Detailed inspection of each column"),
        code(INSPECT_DETAIL),
        md("### 3.4 Speakers and available labels"),
        code(SPEAKERS),
        md("## 4. Acoustic feature table\n\nReads each audio file (up to `N_INSPECT` files, spread across speakers) and measures duration, sample rate, channels, "
           "loudness (RMS dB), peak, clipping %, silence %, zero-crossing rate, spectral centroid / bandwidth / roll-off, median and spread of pitch (F0) and an SNR estimate. "
           "Cached in `OUT_DIR/features`."),
        code(ACOUSTIC),
        md("## 5. Descriptive statistics\n\nMeasures of **central tendency** (mean, median, mode), **dispersion** (standard deviation, variance, quartiles, IQR, range, "
           "coefficient of variation, median absolute deviation) and **shape** (skewness, kurtosis), overall and by emotion and gender."),
        code(DESCRIPTIVE),
        md("## 6. Outliers\n\nTukey fences (below Q1 − 1.5·IQR or above Q3 + 1.5·IQR) and |z| > 3, per feature."),
        code(OUTLIERS),
        md("## 7. Data cleaning\n\nEvery rule is logged with the number of rows it removes. Source files are never touched; the clean table is saved separately."),
        code(CLEANING),
        md("## 8. Data wrangling\n\nLabel encoding, a **speaker-independent** train / validation / test split (no speaker in more than one split), class balance and class weights."),
        code(WRANGLING),
        md("## 9. Analysis and visualisation"),
        code(PLOTS),
        code(PLOTS2),
        md("### 9.1 Statistical tests\nChi-square (emotion × gender, Cramér's V), Kruskal-Wallis (feature ~ emotion, ε²) and Mann-Whitney (feature ~ gender, rank-biserial r)."),
        code(STATS_TESTS),
        md("### 9.2 Listen: one example per class"),
        code(EXAMPLES),
        md("## 10. Features\n\n16 kHz mono, silence trim (`top_db=30`), peak normalisation, crop to `MAX_SECONDS`. MFCC (40 + deltas, mean/std, 160-d), "
           "log-mel (64 bands, mean/std, 128-d) and wav2vec2 mean-pooled embeddings (all layers cached, last layer used)."),
        code(FEATURES),
        code(SSL),
        code(EMBED_VIZ),
        md("## 11. Machine learning models (centralised SER baseline)\n\nTrained on train + validation speakers, evaluated on unseen test speakers. "
           "UAR (balanced accuracy) is the key metric."),
        code(ML),
        md("## 12. Paralinguistic inference attacks on representations (A1)\n\nIf embeddings are shared (feature-sharing / split learning), how much gender, speaker identity and age can an attacker recover? "
           "Gender and age are tested on **unseen speakers**; speaker ID is closed-set by definition."),
        code(ATTACK_REPR),
        md("## 13. Federated learning and attacks by an honest-but-curious server\n\n"
           "Each client is one speaker (split into several clients when a corpus has few speakers). Clients train a small MLP emotion head on frozen features; the server averages updates (FedAvg). "
           "20% of each training speaker's clips are kept out of training as **non-members** for the membership attack."),
        code(FL_SETUP),
        md("### 13.1 Privacy accounting for the DP defences"),
        code(PRIVACY_ACCOUNTING),
        md("### 13.2 FedAvg training loop (with pluggable defences)"),
        code(FL_RUN),
        md("### 13.3 Attacks on the undefended system\n"
           "* **A2 gender** inference from each client's update (grouped CV, so the attacker generalises to new speakers)\n"
           "* **A3 age** inference from updates (where age labels exist)\n"
           "* **A4 speaker linkage**: recognise which client sent an update from its earlier updates\n"
           "* **A5 membership** inference on the final model (loss threshold; AUC and max TPR − FPR)"),
        code(FL_ATTACKS),
        md("## 14. Defences\n\n"
           "* **Local DP**: each client clips its update to `CLIP` and adds Gaussian noise before sending (protects every single update; costly in utility).\n"
           "* **Secure Aggregation**: pairwise masks that cancel in the sum, so the server sees only the aggregate.\n"
           "* **SecAgg + distributed DP**: each client adds a share of the noise; the sum carries the DP guarantee.\n"
           "* **Gradient reversal**: an adversarial gender head is trained through a gradient-reversal layer so the shared encoder forgets gender."),
        code(DEFENCES),
        md("## 15. Evaluation: privacy-utility-efficiency and the decision gate"),
        code(PUE_PLOTS),
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
