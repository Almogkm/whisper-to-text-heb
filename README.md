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
- runs a second cleanup pass through local Ollama
- wraps the transcript in `<div dir="rtl">` for better right-to-left rendering

## Important environment note

This project is now set up around the dedicated conda environment `whisper`.

The intended runtime is:

- conda environment for Python and `ffmpeg`
- `openai-whisper` installed with `pip`
- local Ollama running on `http://127.0.0.1:11434`
- UTF-8 Markdown output with Hebrew defaults

## Setup

From this directory:

```bash
conda env update -n whisper -f environment.yml
conda activate whisper
pip install openai-whisper
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
The output file includes both a cleaned transcript and the raw Whisper transcript.

Transcribe a directory recursively into a separate output directory:

```bash
python transcribe_to_md.py /path/to/audio-dir --recursive --output-dir /path/to/transcripts
```

Use a larger model for better Hebrew accuracy:

```bash
python transcribe_to_md.py /path/to/audio.ogg --model medium
```

Skip the Ollama cleanup layer:

```bash
python transcribe_to_md.py /path/to/audio.ogg --skip-cleanup
```

## Common options

- `--language he` keeps Hebrew transcription as Hebrew
- `--model small` is the default balance between speed and quality on CPU
- `--model medium` usually improves Hebrew accuracy, but is slower
- `--cleanup-model qwen36-reader-16k` overrides the local cleanup model
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
- cleaned transcript when Ollama cleanup succeeds
- raw Whisper transcript for reference

## Notes

- The first run for a given model downloads model weights.
- On this CPU-only machine, `small` is the safest default.
- If `ffmpeg` is missing from the conda env, Whisper will fail to open audio files.
- The default cleanup target is `qwen36-reader-16k`. On September 3, 2026, `qwen36-reader-32k` was installed locally but failed on this machine with `model requires more system memory (24.9 GiB) than is available (22.0 GiB)`.
