# Datasets folder survey (read-only), 17 Nov 2026
Folder: C:\Desktop\RGU PhD\New folder\Datasets. Nothing was modified or extracted.

| Folder | What it is | Files / format | Emotion | Speaker ID | Gender | Age |
|---|---|---|---|---|---|---|
| IEMOCAP_full_release | IEMOCAP, all 5 sessions (wav, MOCAP, avi, transcripts, EmoEvaluation) | 10,039 utterance wavs + `iemocap_full_dataset.csv` | Yes (neu/fru/ang/sad/exc/hap...; 2,507 rows `xxx` = no agreement) | From filename (10 speakers) | Yes (M 5,098 / F 4,941) | No |
| Datasets/IEMOCAP | IEMOCAP again as 5 HF parquet files (~790 MB) | 5 .parquet | Likely | Likely | Likely | No |
| CREMA-D-master | CREMA-D repo + demographics + vote tables | 6,211 mp3 (76 of 91 actors only) | Yes (filename + crowd votes) | Yes (91) | Yes (48 M / 43 F) | Yes, 20 to 74 (+ race, ethnicity) |
| Datasets/Crema | CREMA-D full audio | 7,442 wav (all 91 actors) | Yes | Yes | via VideoDemographics.csv | via VideoDemographics.csv |
| Datasets/EmoDB | EmoDB | 535 wav + `emoDB demographics` csv | Yes (filename) | Yes (10) | Yes (5/5) | Yes, 21 to 35 |
| Datasets/Savee + AudioData | SAVEE (twice: 480 + 960 wav) | wav 44.1 kHz | Yes (7 classes) | Yes (4) | All male | No |
| Datasets/Ravdess, Datasets/RAVDES | RAVDESS audio speech (same 1,440 files, RAVDES holds them twice) | wav | Yes (filename) | Yes (24) | Yes (odd = M, even = F) | No |
| RAVDESS | RAVDESS video speech, 24 actors | 2,880 mp4 | Yes | Yes | Yes | No |
| RAVESS | RAVDESS original zips (audio speech/song, video speech/song) | 49 zips (~25 GB); Video_Song_Actor_18 missing (Actor 18 has no song in RAVDESS, so expected) | | | | |
| TESS | Toronto Emotional Speech Set | 2,800 wav, stored twice (5,600) | Yes (7) | 2 speakers | Both female | Two ages (young/old) |
| MSP-IMPROV | Session 1 only (of 6) | 989 wav + 989 avi, 8,438 transcripts, TextGrids | No label file; only target scenario (A/H/N/S) in folder names | Yes (filename) | Yes (filename) | No |
| MSP-POSCAST | MSP-Podcast audio only | 58,899 wav | No labels file present | No | No | No |
| MSP_Face-master | MSP-Face code repo + `label_consensus.json` (emotion, gender, speaker, split); videos need download script | 83 files | Yes | Yes | Yes | No |
| Raw | Looks like CMU-MOSI (93 YouTube videos, 2,199 segments, COVAREP) | wav 11k/16k, mp4, transcripts | Sentiment, no label file seen | No | No | No |
| mcv-sps-st-09-2025.tar.gz | Common Voice Spontaneous Speech 1.0 (Sep 2025), 21 languages, no English, 4.6 GB | 38,531 mp3 + per-language TSV | No | Yes (client_id, ~380 speakers) | Only 4.6% labelled (mainly Western Penan) | Only 4.6% labelled |
| pollution-data | US air pollution CSV, unrelated to the PhD | 1 csv | | | | |

## Gaps for attribute inference
1. **Age labels exist only for CREMA-D (good spread, 20 to 74) and EmoDB (narrow, 21 to 35).** TESS gives a coarse young/old split. Age attacks should centre on CREMA-D.
2. **MSP-IMPROV is incomplete**: session 1 only and no `Evaluation.txt` emotion labels. Request sessions 2 to 6 and the label file.
3. **MSP-Podcast has no labels or speaker/gender metadata** in the folder (needs `labels_consensus.csv` and the speaker ID file). It is also not in the planned corpus list.
4. **Standard CommonVoice is missing** (planned for gender/age). Only the Spontaneous Speech release is present, and it is mostly unlabelled for gender/age.
5. SAVEE (all male) and TESS (all female, 2 speakers) cannot support gender inference alone and give few speakers for speaker ID.
6. Duplicates to note: RAVDESS speech in 2 places (RAVDES contains it twice), SAVEE twice, TESS twice, IEMOCAP as wav and parquet, CREMA-D as partial mp3 and full wav. Use Datasets/Crema (complete) rather than AudioMP3.
7. IEMOCAP: 2,507 utterances lack an agreed emotion; standard 4-class setup (ang/hap+exc/neu/sad) gives ~5,531.

## Addendum: mcv-sps-st-09-2025.tar.gz (read without extracting into the folder)
Mozilla Common Voice **Spontaneous Speech** 1.0 (2025-09-05), 4.6 GB. 21 language subfolders `sps-corpus-1.0-2025-09-05-<lang>/`, each with `audios/` (mp3), `ss-corpus-<lang>.tsv` and `ss-reported-audios-<lang>.tsv`. 38,531 mp3 clips, 38,901 TSV rows, about 239 hours, about 380 speakers.
Languages: koo, lth, ttj, ruc, cgg, bxk, led, lke, kcn, pne, rwm, ukv (mostly Ugandan/East African), aln (Gheg Albanian), hch, tob, meh, mmc, top (Latin American indigenous), bew (Betawi), cy (Welsh), sco (Scots). **No English.**
TSV columns: client_id, audio_id, audio_file, duration_ms, prompt_id, prompt, transcription, votes, age, gender, language, split, char_per_sec, quality_tags. **No accent column, no emotion labels.**
- Speaker ID: yes, `client_id` for every clip (8 to 26 speakers per language).
- Gender/age: only 1,798 of 38,901 clips (4.6%). Only Western Penan (pne) has both genders (885 F, 554 M) and an age spread (teens to fifties). cgg, kcn, meh, ruc each have one male speaker labelled; the other 16 languages have none.
Fit: useful for speaker-ID attacks and cross-lingual robustness, but not for gender/age attacks or SER. The standard Common Voice (read speech, e.g. English) release would be needed for gender/age at scale.
