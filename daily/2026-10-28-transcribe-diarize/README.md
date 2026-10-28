# Speaker-attributed transcription (Nemotron 3.5 Transcribe-style toy)

Daily experiment, 28 October 2026. Trending item: NVIDIA released
**Nemotron 3.5 Transcribe** on 28 October 2026, an open-weight speech
model for transcription, speaker attribution, translation and audio
question answering in one checkpoint, alongside Granary v2, its open
speech dataset. The figures in this paragraph are NVIDIA's / reporters'
claims, not measurements from this repo:

* 30B-parameter mixture-of-experts model with about 4B active
  parameters per token (TechCrunch, Unite.AI).
* Transcription in 40 languages, translation for 31 languages
  to/from English, speaker attribution for up to 8 speakers, and
  recordings up to one hour long (Unite.AI).
* Granary v2: 463,062 hours across 48 languages, up from 25 languages
  in v1 (SQ Magazine).
* Reported benchmark positions: first among open-weight models on the
  Open ASR Leaderboard English score (NVIDIA's chart leaves API models
  out, per SQ Magazine), 2.8% on Artificial Analysis' AA-WER index
  (tied first among open-weight models), 18.26% diarization error rate
  on David AI's multilingual track, and 32.4% cpWER on VoiceArena's
  English benchmark (SQ Magazine).

Sources: TechCrunch, 28 Oct 2026
<https://techcrunch.com/2026/10/28/voice-ai-in-vogue-nvidia-releases-new-open-transcription-model/>;
Unite.AI, 28 Oct 2026
<https://www.unite.ai/nvidia-launches-open-weight-speech-model-and-463-062-hour-audio-dataset/>;
SQ Magazine
<https://sqmagazine.co.uk/?p=33887>.

## What this repo actually does

This is **not** the Nemotron model and it decodes **no audio**.
It is a stdlib-only Python implementation of the pipeline around such
a model, using synthetic segments with hand-made voice embeddings, so
the diarization and scoring logic can be read and tested:

* `transcribe.py` — speaker clustering (cosine similarity to cluster
  centroids, 8-speaker cap), WER, cpWER (concatenated
  minimum-permutation WER, so labels are scored up to permutation),
  diarization error rate (missed + false alarm + confusion), prompt
  glossary correction, decision extraction, transcript formatting.
* `demo.py` — an 8-turn, 3-speaker synthetic meeting. The raw
  recogniser output misspells two glossary terms; a meeting prompt
  supplying `["Kubernetes", "PostgreSQL"]` corrects both.
* `test_transcribe.py` — plain-Python test runner (no pytest needed).

## Demo results (toy numbers, synthetic data only)

```
clustered labels match ground truth exactly (3 speakers, 8 turns)
raw WER vs corrected truth: 2 errors / 76 words = 0.0263
prompt-corrected WER      : 0 errors / 76 words = 0.0000
cpWER clustered           : 0 errors / 76 words = 0.0000
cpWER all-one-speaker     : 92 errors / 76 words = 1.2105
DER clustered             : 0.0s missed, 0.0s false alarm, 0.0s confusion / 48.0s = 0.0000
DER all-one-speaker       : 30.0s confusion / 48.0s = 0.6250
```

The contrast is the point of the demo: perfect words with flat
speaker labels still score terribly on speaker-attributed metrics.
With one label for everyone, cpWER exceeds 1.0 because misplaced
words count as insertions under the wrong speaker and deletions from
their true speaker, and DER hits 0.625 because 30 of the 48 seconds
belong to speakers other than the one the flat label maps to.

## Run it

```
python3 demo.py
python3 test_transcribe.py   # 10/10 tests passing
```

## Honest limits

* Embeddings are synthetic axis vectors with tiny jitter, not voice
  prints from audio; real diarization faces overlap, noise and
  same-speaker drift that this toy does not model.
* Prompt correction only fixes a word within edit distance 2 of
  exactly one glossary term; ambiguous near-misses are left alone
  rather than guessed.
* No translation, audio question answering, or streaming is
  implemented; those are Transcribe capabilities listed above, not
  demonstrated here.
