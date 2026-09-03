# Architecture

## Runtime shape

The current implementation is a single Python package with one large runtime module. The external seam is the CLI; everything else is currently behind that seam.

```text
CLI
  -> input discovery
  -> Whisper transcription
  -> DICTA cleanup
  -> review-stage reconciliation
  -> Markdown rendering
  -> file write
```

## Module map

| Module / file | Role | Notes |
| --- | --- | --- |
| `transcribe_to_md.py` | Thin launcher | Adds `src/` to `sys.path` and forwards to package main |
| `src/whisper_md_transcriber/__main__.py` | Package launcher | Lets the package run as a module |
| `src/whisper_md_transcriber/cli.py` | Main runtime module | Currently owns argument parsing, orchestration, stage prompts, Ollama calls, and Markdown rendering |
| `ollama/DictaHebrewCleaner.Modelfile` | DICTA wrapper | Pins the Hebrew cleanup alias and its stop-token behavior |

## Execution flow

1. Parse CLI arguments
2. Normalize paths and discover audio jobs
3. Resolve device and load Whisper
4. Transcribe one audio source into a raw transcript
5. Run first-pass cleanup through Ollama with the DICTA alias
6. Run second-pass review through Ollama against both raw and cleaned text
7. Render a multi-stage Markdown transcript package
8. Write the result to the destination path

## Current design strengths

- Small external interface: one CLI entrypoint with sensible defaults
- Locality: the whole workflow is easy to inspect because it lives in one file
- Auditable outputs: raw, intermediate, and final transcripts are preserved
- Fail-soft post-processing: if a later stage fails, earlier text is still emitted instead of losing the run

## Current design liabilities

- `cli.py` is doing too much. It currently mixes interface, orchestration, prompt construction, adapter code, and rendering.
- Stage-specific behavior is implicit rather than modeled. Cleanup and review are similar but not represented by a shared abstraction.
- Prompt definitions live inline with orchestration code, which makes systematic prompt iteration awkward.
- Rendering and orchestration are coupled, which makes alternative output formats harder to add.

## Deep-module read of the codebase

Using the codebase-design vocabulary:

- The current **module** is effectively `cli.py`
- Its **interface** is the CLI flags plus the `main()` function
- Its **implementation** is deeper than the interface, but still not as deep as it could be because multiple concerns are exposed through branching arguments instead of being hidden behind smaller internal seams

## Clean seams that already exist

- **Whisper adapter seam**: `load_model()` and `model.transcribe(...)`
- **Ollama adapter seam**: `load_ollama_models()` and `ollama_chat()`
- **Rendering seam**: `render_markdown()`
- **Job discovery seam**: `discover_jobs()`

## Best refactor candidates

### 1. Extract a pipeline module

Move orchestration out of `cli.py` into a dedicated pipeline module with an interface close to:

```text
run_transcription_job(config, audio_job) -> transcript_package
```

That would reduce CLI knowledge and concentrate workflow logic in one reusable place.

### 2. Model stages explicitly

Introduce a small stage abstraction so cleanup and review are siblings rather than one-off functions. That would make it easier to add:

- alternate review models
- optional review retries
- benchmark harnesses
- future stages such as glossary repair or timestamp alignment

### 3. Separate rendering from package data

Return a structured transcript package first, then render Markdown from that package. This would make JSON export, HTML export, or diff tooling straightforward.

## Why the current architecture still makes sense

The project is small and still actively changing. Keeping the logic in one module has been reasonable up to now because prompt iteration and model experiments have been the dominant work, not broad code reuse. The next refactor should happen only when the experimentation surface stabilizes enough that repeated edits to `cli.py` become a drag.
