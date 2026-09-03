# Operations

## Environment

Current expected runtime:

- Ubuntu
- conda environment: `whisper`
- Python 3.10
- `ffmpeg` available in the conda env or on `PATH`
- local Ollama server on `http://127.0.0.1:11434`
- UTF-8 output

## Required local models

### Whisper

- `small` is available and was the original default
- `medium` is installed locally and is the practical quality default used in testing
- `large` resolves to `large-v3` in the installed Whisper package

### Ollama

- first-pass cleanup: `dicta-hebrew-cleaner`
- second-pass review: `qwen36-reader-16k`

The DICTA alias is defined in `ollama/DictaHebrewCleaner.Modelfile`.

## Setup sequence

```bash
conda env update -n whisper -f environment.yml
conda activate whisper
pip install openai-whisper
ollama create dicta-hebrew-cleaner -f ollama/DictaHebrewCleaner.Modelfile
```

Optional Ubuntu package:

```bash
sudo apt install -y fonts-noto-core
```

## Common commands

Transcribe one file with the current quality-oriented path:

```bash
python transcribe_to_md.py /path/to/file.ogg --model medium
```

Skip the review stage:

```bash
python transcribe_to_md.py /path/to/file.ogg --model medium --skip-review
```

Benchmark raw Whisper only:

```bash
python transcribe_to_md.py /path/to/file.ogg --model medium --skip-cleanup
```

## Operational behavior

- output defaults to the project directory
- one input file produces one Markdown transcript package
- directory inputs preserve relative paths inside the chosen output directory
- post-processing is fail-soft; the run still writes output if later stages fail

## Known runtime constraints

### CPU-only behavior

This environment has been operated from CPU, not CUDA. That makes:

- Whisper `medium` practical
- Whisper `large` usable but slower
- large-context Qwen variants memory-sensitive

### Memory pressure

Observed during development:

- `qwen36-reader-32k` did not fit this machine cleanly in earlier tests
- `dicta-hebrew-cleaner` fits and runs on CPU
- `qwen36-reader-16k` fits for the review pass but is materially heavier than DICTA

## Troubleshooting

### `ffmpeg` not found

Install `ffmpeg` into the conda env or system and retry.

### Ollama model missing

If cleanup fails because the DICTA alias is missing:

```bash
ollama create dicta-hebrew-cleaner -f ollama/DictaHebrewCleaner.Modelfile
```

### DICTA emits literal chat control tokens

The Modelfile must include stop tokens for:

- `<|im_start|>`
- `<|im_end|>`
- `<|endoftext|>`

That fix is already present in the checked-in Modelfile.

### Review stage returns `done_reason=length`

This happened during development when the second pass was allowed to think freely and over-generate. The current code mitigates that by:

- forcing `think=False`
- capping review generation length based on transcript size

## Artifact locations

- main outputs: project root by default
- benchmark artifacts: `benchmarks/`
- takeoff docs: `docs/takeoff/`
- model wrapper: `ollama/`

## Safety and privacy posture

The current design is fully local:

- no remote transcription API
- no remote cleanup API
- transcript text remains on the local machine unless the user exports it elsewhere
