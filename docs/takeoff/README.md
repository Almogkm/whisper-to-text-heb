# Project Takeoff

This directory is the engineering handoff for the current state of `whisper-md-transcriber` as of 2026-09-03.

Read these files in order:

1. [README.md](../../README.md) for setup and basic usage
2. [ARCHITECTURE.md](./ARCHITECTURE.md) for runtime structure and module seams
3. [OPERATIONS.md](./OPERATIONS.md) for environment, model setup, and troubleshooting
4. [QUALITY-AND-BENCHMARKS.md](./QUALITY-AND-BENCHMARKS.md) for accuracy history and performance numbers
5. [NEXT-STEPS.md](./NEXT-STEPS.md) for the most valuable follow-on work
6. [APP-SHARING.md](./APP-SHARING.md) for turning the local tool into a shareable app
7. [WHATSAPP-BOT.md](./WHATSAPP-BOT.md) for turning the local tool into a WhatsApp voice-note bot
8. [LONG-FORM.md](./LONG-FORM.md) for long recordings with speaker attribution
9. [LONG-FORM-STATUS.md](./LONG-FORM-STATUS.md) for the current state of the long-form branch and what is left

## What this project does

The tool converts local audio files into Markdown transcript packages. It is tuned for Hebrew speech and currently favors quality over minimal latency by using a three-stage local pipeline:

1. Whisper transcribes speech into raw text
2. DICTA performs a Hebrew-first cleanup pass
3. A second local reviewer reconciles the DICTA output against the raw Whisper text

## Current code shape

- Python package: `src/whisper_md_transcriber/`
- Main runtime entrypoint: `whisper_md_transcriber.cli`
- Thin launcher: `transcribe_to_md.py`
- Project metadata: `pyproject.toml`
- Conda environment: `environment.yml`
- Ollama model wrapper: `ollama/DictaHebrewCleaner.Modelfile`

## Current repo milestones

- `731c408` — initial local backup before DICTA migration
- `3cba7d6` — switched first cleanup stage from Qwen to DICTA
- `eb54ad0` — added a second-pass review layer after DICTA

## Current behavior

For a successful full run, each output Markdown file contains:

- job metadata
- final reviewed transcript
- intermediate DICTA transcript
- raw Whisper transcript

That format is intentional. It keeps the output auditable and makes model regressions visible.

## Practical status

On the reference WhatsApp OGG used during development, the current pipeline moved from roughly 60-70/100 quality into the high-90s. The strongest remaining errors are context-specific words that all stages can still hear plausibly but interpret incorrectly, especially names and kinship terms.
