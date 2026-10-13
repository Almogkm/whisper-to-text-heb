# whisper-md-transcriber

Basic local speech-to-text app built on OpenAI's open source Whisper model.

It is tuned for the request here:

- local transcription on Ubuntu
- `.ogg` support first, with other common audio formats also accepted
- Hebrew-first defaults
- Markdown output files

## What it does

Given one audio file or a directory of audio files, it produces one `.md` file per audio file.

By default it:

- transcribes, not translates
- assumes Hebrew input with `--language he`
- writes UTF-8 Markdown
- writes output into the project directory by default
- runs a first cleanup pass through local Ollama with a Hebrew-specific DICTA model
- runs a second reconciliation pass that reviews the DICTA output against the raw Whisper text
- wraps the transcript in `<div dir="rtl">` for better right-to-left rendering

## Important environment note

This project is now set up around the dedicated conda environment `whisper`.

The intended runtime is:

- conda environment for Python and `ffmpeg`
- `openai-whisper` installed with `pip`
- local Ollama running on `http://127.0.0.1:11434`
- DICTA cleanup model installed in Ollama
- at least one review model installed in Ollama; the default is `qwen36-whisper-reviewer-16k`
- UTF-8 Markdown output with Hebrew defaults

## Setup

From this directory:

```bash
conda env update -n whisper -f environment.yml
conda activate whisper
pip install openai-whisper
```

Install the default Hebrew cleanup model in Ollama:

```bash
ollama create dicta-hebrew-cleaner -f ollama/DictaHebrewCleaner.Modelfile
```

Install the default project-specific review model in Ollama:

```bash
ollama create qwen36-whisper-reviewer-16k -f ollama/Qwen36WhisperReviewer16k.Modelfile
```

That local alias points at the official DICTA GGUF model:

```text
hf.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF:Q4_K_M
```

Optional but useful on Ubuntu for better Hebrew rendering in GUI apps:

```bash
sudo apt install -y fonts-noto-core
```

## Usage

Transcribe one file:

```bash
python transcribe_to_md.py /path/to/audio.ogg
```

This writes `/home/almogked/Projects/whisper/<audio-stem>.md` unless you pass `--output-dir`.
The output file includes the final reviewed transcript, the intermediate DICTA transcript, and the raw Whisper transcript.

Transcribe a directory recursively into a separate output directory:

```bash
python transcribe_to_md.py /path/to/audio-dir --recursive --output-dir /path/to/transcripts
```

Use a smaller Whisper model only when you need to trade accuracy for speed:

```bash
python transcribe_to_md.py /path/to/audio.ogg --model small
```

Skip the Ollama cleanup layer:

```bash
python transcribe_to_md.py /path/to/audio.ogg --skip-cleanup
```

Skip only the second review layer and keep the DICTA result as the final transcript:

```bash
python transcribe_to_md.py /path/to/audio.ogg --skip-review
```

## Long recordings with speakers

For long Hebrew recordings (interviews, conversations) with speaker attribution, use the long-form pipeline:

```bash
conda run -n whisper python transcribe_long.py /path/to/recording.m4a --glossary glossary.json
```

It runs two faster-whisper large-v3 passes (ivrit.ai Hebrew fine-tune + vanilla), pyannote diarization, then DICTA cleanup and Qwen review per speaker turn, and writes `<stem>.speakers.md`, `.srt` and `.json`.
Every stage is checkpointed under `work/<stem>/`, so an interrupted run resumes where it stopped.
The glossary comes from the Hebrew questionnaire at `docs/glossary-questionnaire.html`.
See [docs/takeoff/LONG-FORM.md](docs/takeoff/LONG-FORM.md) for the full runbook.

## Common options

- `--language he` keeps Hebrew transcription as Hebrew
- `--model medium` is the default balance for this project
- `--model small` is faster, but usually worse on Hebrew
- `--cleanup-model dicta-hebrew-cleaner` overrides the local cleanup model
- `--review-model qwen36-whisper-reviewer-16k` overrides the second-pass review model
- `--overwrite` allows replacing existing `.md` files
- `--prompt "..."` lets you provide Hebrew context words or names

## Supported input formats

Supported extensions:

- `.ogg`
- `.wav`
- `.mp3`
- `.m4a`
- `.mp4`
- `.mpeg`
- `.mpga`
- `.webm`
- `.flac`

## Output format

Each output file contains:

- source filename
- transcription timestamp
- Whisper model and language
- final reviewed transcript when both Ollama passes succeed
- intermediate DICTA transcript when the second pass succeeds
- raw Whisper transcript for reference

## Review Benchmark

The repo includes a small judged benchmark for the second-pass reviewer:

```bash
conda run -n whisper python benchmarks/review_model_benchmark.py \
  --output-dir benchmarks/review-runs/manual \
  qwen36-reader-16k qwen36-whisper-reviewer-16k
```

This keeps the raw Whisper and DICTA first-pass text fixed, then scores each review model by how many judged word-level corrections it recovers.
The project-specific reviewer path also applies a narrow final Hebrew normalization pass for recurring ASR leftovers after the Qwen review step.

## Notes

- The first run for a given model downloads model weights.
- On this CPU-only machine, `medium` is slower than `small`, but it is now the default because it produces materially better Hebrew transcripts.
- If `ffmpeg` is missing from the conda env, Whisper will fail to open audio files.
- The default cleanup target is `dicta-hebrew-cleaner`, which points at the official DICTA 12B Hebrew GGUF model in Ollama.
- If the DICTA model is not installed yet, the tool prints the exact `ollama create` command to run.
- The default second-pass review target is `qwen36-whisper-reviewer-16k`, a repo-local Qwen alias tuned for Hebrew transcript reconciliation.
