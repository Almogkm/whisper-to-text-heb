# Long-Form Runbook

Pipeline for one long Hebrew recording (e.g. a 90-minute interview) with speaker attribution, tuned for quality over speed.

Entry point: `transcribe_long.py` → `src/whisper_md_transcriber/longform.py`.

## Pipeline

1. **prepare**: ffmpeg → 16 kHz mono WAV, EBU R128 loudness normalization (`--no-loudnorm` to skip).
2. **asr**: faster-whisper, one pass per model in `--asr-models`:
   - primary: `ivrit-ai/whisper-large-v3-ct2` (Hebrew fine-tune of large-v3, ~4.7k h of Knesset + crowd data)
   - alternative: `Systran/faster-whisper-large-v3` (vanilla large-v3)
   - beam 5, Silero VAD, word timestamps, `hallucination_silence_threshold=2.0`
   - glossary: context → `initial_prompt`, names/terms → `hotwords` (injected into every 30 s window)
   - float32 on CPU by default (`--compute-type int8` is ~3× faster with a small quality cost)
3. **diarize**: pyannote with `--num-speakers 2` fixed.
   - with a Hugging Face token: `pyannote/speaker-diarization-community-1` (newer, better)
   - without: `ivrit-ai/pyannote-speaker-diarization-3.1` (ungated mirror of 3.1)
4. **align**: each primary word gets the speaker whose diarization turn covers its midpoint; one-word flips shorter than 0.4 s are smoothed; words merge into speaker turns (split at speaker change, 1.5 s pause, or 600 chars). The alternative ASR words are attached to the same turns by time.
5. **cleanup**: DICTA 12B rewrites turns in blocks of ~2500 chars. Each turn carries a `<<id|speaker>>` marker; the block is rejected unless every id comes back, then retried turn by turn. A rewrite whose length drifts outside 0.5–1.8× of the source is discarded.
6. **review**: Qwen 3.6 35B-A3B sees, per turn, raw A (ivrit), raw B (vanilla) and the DICTA text, plus the glossary, in blocks of ~1500 chars. Same id and length guards. `--review-think` enables thinking (much slower).
7. **render**: `<stem>.speakers.md` (RTL, timestamps, ⚠️ on flagged turns, plus a "segments to check" list), `<stem>.speakers.srt`, `<stem>.speakers.json` (all stages per turn).

A turn is flagged when: any word has probability < 0.3 (or mean < 0.6), the two ASR models disagree on > 35% of words, review changed > 30% of words, or an LLM stage fell back to its input.

## Measured speed (this machine, CPU only, 2026-10-12)

Smoke test: 48 s, 2 synthetic speakers (one voice note with every other 12 s pitch-shifted).

| Stage | 48 s test | Extrapolated to 90 min |
|---|---|---|
| ASR ivrit large-v3, float32, beam 5 | 54 s (incl. model load) | ~80 min |
| ASR vanilla large-v3, float32, beam 5 | 47 s | ~75 min |
| Diarization (3.1 mirror) | 19 s (incl. load) | ~20 min |
| DICTA cleanup | ~41 s / 640 chars | ~70–80 min |
| Qwen review | ~42 s / 640 chars | ~70–80 min |
| **Total** | **~3.5 min** | **~5.5–6.5 h** |

Smoke-test observations: with the glossary, ivrit large-v3 got "אלמוג" and "אשתך" right where `medium` produced "מוג" and "אשלך"; the vanilla model was clearly worse ("אני אצליח", "להשכיבת"), so it is only an alternative reading. DICTA changed little on ivrit output; Qwen fixed the remaining error ("צורך. בין דרכונים" → "צורך בדרכונים").

## Checkpoints and resume

Everything lands in `work/<stem>/`:

| File | Content |
|---|---|
| `audio.wav`, `prepare.json` | normalized audio, duration |
| `asr-<model>.json` | segments + words per ASR model |
| `diarization.json` | speaker turns from pyannote |
| `turns.json` | aligned turns |
| `cleanup.jsonl`, `review.jsonl` | per-turn LLM results, appended after every block |

Re-running the same command skips finished stages and resumes LLM stages from the last finished block. A checkpoint is reused only when its parameters (model, glossary, upstream results) match, so changing the glossary re-runs ASR and everything after it. `--fresh` discards all checkpoints.

## Recommended workflow for the real recording

```bash
conda activate whisper

# 0. (optional, better diarization) accept terms at
#    https://huggingface.co/pyannote/speaker-diarization-community-1, then:
hf auth login

# 1. ASR + diarization + alignment only (~2–3 h for 90 min on this machine)
python transcribe_long.py /path/to/recording.m4a --glossary glossary.json --stop-after align

# 2. Look at work/<stem>/turns.json (or run step 3 with --skip-cleanup for a quick .md)
#    to see who "speaker 1" and "speaker 2" are, in order of first appearance.

# 3. Full run with names (resumes from checkpoints, LLM stages ~3–4 h)
python transcribe_long.py /path/to/recording.m4a --glossary glossary.json --speaker-names "רחל,דוד"
```

Names are passed to the LLM stages (they help with gender agreement), so set them before the LLM stages run. Changing names afterwards re-runs cleanup and review.

Then open `<stem>.speakers.md`, jump to the flagged timestamps in "קטעים לבדיקה", and listen to those spots.

## Glossary

`docs/glossary-questionnaire.html` is a Hebrew, RTL questionnaire that any Hebrew speaker can fill in a browser (works offline, autosaves locally). "שמירת קובץ" downloads `glossary.json`, which is the `--glossary` input. Speakers' genders feed the LLM prompts; every name and term feeds Whisper hotwords and both LLM prompts.

## Installed for this pipeline

- `faster-whisper==1.2.1`, `pyannote.audio==4.0.7` in the `whisper` conda env (torch 2.13 untouched)
- Hugging Face cache: `ivrit-ai/whisper-large-v3-ct2`, `ivrit-ai/whisper-large-v3-turbo-ct2` (fast drafts), `Systran/faster-whisper-large-v3`, `ivrit-ai/pyannote-speaker-diarization-3.1`, `ivrit-ai/pyannote-segmentation-3.0`, `pyannote/wespeaker-voxceleb-resnet34-LM`
- Ollama: `dicta-hebrew-cleaner`, `qwen36-whisper-reviewer-16k` (already present)

pyannote 4 always tries to load PLDA weights from the gated community-1 repo, even for 3.x pipelines that never use them; `run_diarization` tolerates that gate, and pyannote prints a harmless "Could not download xvec_transform.npz" message.
