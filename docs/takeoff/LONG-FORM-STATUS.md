# Status: long-form diarized transcription

Living status doc for the long-form work: what is done and what is left. Update it in place instead of writing a new dated handoff. The glossary lives in [CONTEXT.md](../../CONTEXT.md), the runbook in [LONG-FORM.md](./LONG-FORM.md).

Last updated: 2026-10-13. Branch: `feat/diarized-long-form` (created from `main` at `f270f18`), committed and pushed to `origin`.

## Goal

Transcribe one ~90-minute Hebrew-only recording with 2 speakers at the best achievable quality, time not a constraint (a weekend). Runs on this Linux machine (CPU only, 32 threads, 60 GB RAM), conda env `whisper`.

## Done

### Environment (conda env `whisper`)
- Installed `faster-whisper==1.2.1`, `pyannote.audio==4.0.7` (+ deps: torchaudio 2.11, torchcodec 0.17, ctranslate2 4.8.2). torch 2.13 untouched. Imports verified.
- Added both to `pyproject.toml` and `environment.yml`; new script entry `whisper-md-transcriber-long`.

### Models in the Hugging Face cache (`~/.cache/huggingface/hub`)
- `ivrit-ai/whisper-large-v3-ct2`: primary ASR (Hebrew fine-tune of large-v3)
- `Systran/faster-whisper-large-v3`: alternative reading for the review stage
- `ivrit-ai/whisper-large-v3-turbo-ct2`: fast drafts (not used by default)
- `pyannote/speaker-diarization-community-1`: gated diarization, the default whenever a Hugging Face login exists (terms accepted and `hf auth login` done on 2026-10-13)
- `ivrit-ai/pyannote-speaker-diarization-3.1` + `ivrit-ai/pyannote-segmentation-3.0` + `pyannote/wespeaker-voxceleb-resnet34-LM`: ungated fallback, used without a token
- Ollama models were already present: `dicta-hebrew-cleaner`, `qwen36-whisper-reviewer-16k`

### Code
- `src/whisper_md_transcriber/longform.py` + launcher `transcribe_long.py`: the pipeline prepare → asr (×2 models) → diarize → align → cleanup (DICTA) → review (Qwen) → render (`<stem>.speakers.md/.srt/.json`). Every stage saves checkpoints under `work/<stem>/`, and LLM stages resume per block. Details: [LONG-FORM.md](./LONG-FORM.md).
- `cli.py`: `ollama_chat` gained an optional `num_ctx` (backward compatible).
- `.gitignore`: `work/`.

### Glossary questionnaire
- `docs/glossary-questionnaire.html`: Hebrew RTL form, offline, autosaves in the browser, exports `glossary.json` (schema read by `load_glossary`). Checked in a real browser by the user (2026-10-13).

### Docs
- `docs/takeoff/LONG-FORM.md` (runbook + measured speed), README section, 3 new terms in `CONTEXT.md` (speaker turn, glossary, alternative reading).

### Unit tests
`tests/test_longform.py` (stdlib `unittest`, no extra deps): speaker assignment, smoothing, turn building, alternative-reading alignment, block packing, marker parsing, rewrite guards, flags, glossary loading. Run: `python -m unittest discover -s tests`.

### Verified (smoke test, open diarization model, 2026-10-12)
48 s voice note made into 2 synthetic speakers (every other 12 s pitch-shifted); test files are in the session scratchpad, not the repo.
- End to end in ~3 min; resuming from checkpoints works.
- ivrit + glossary fixed "אלמוג" / "אשתך" vs `medium`; Qwen fixed the remaining error.
- Measured speed: ~1× real time per ASR model (float32, beam 5); expect **~5.5–6.5 h for 90 min**.

### Verified (smoke test, `community-1`, 2026-10-13)
A different 48 s voice note, same synthetic setup (speaker changes at 12, 24 and 36 s).
- End to end in 4.1 min; 107 words → 5 turns, 1 flagged. No "xvec_transform.npz" warning any more.
- Speakers alternate correctly. Each change landed 0.4–0.8 s early (11.2 / 23.7 / 35.4 s), i.e. about one boundary word went to the next speaker. Same pattern as with the open model, so the real-audio trial still needs to check boundaries.
- DICTA: 3 of 5 turns failed the block round trip and passed when retried one by one.

### Bugs found and fixed during testing
1. pyannote 4 always loads PLDA from the gated `community-1` repo → `run_diarization` ignores that gate (3.x clustering never uses PLDA). Without a login pyannote still prints a harmless "Could not download xvec_transform.npz".
2. Words snapping to 0.02 s diarization slivers → speaker chosen by time overlap, slivers < 0.15 s ignored.
3. LLM checkpoints reused after the turns changed → keyed to a fingerprint of the turn contents.
4. DICTA sometimes merges turns and renumbers them → per-turn guards (length 0.5–1.8×, ≤ 60% word change); only the rejected turns are retried one by one.

## Left to do

### Needs the user
1. **Recording** (path, format).
2. **Glossary**: someone who knows the recording fills `docs/glossary-questionnaire.html` → `glossary.json`.

### Run plan for the real file
1. `python transcribe_long.py REC --glossary glossary.json --stop-after align` (~2.5–3 h)
2. Identify speaker 1/2 (order of first appearance) from `work/<stem>/turns.json`.
3. `python transcribe_long.py REC --glossary glossary.json --speaker-names "A,B"` (LLM stages ~2.5–3 h, resumes from checkpoints)
4. Listen to the ⚠️ turns listed under "קטעים לבדיקה".

### Recommended before the real run
- **10-minute real-audio trial** first: check diarization quality on real voices (the smoke test used synthetic voices; one boundary word was misassigned by pyannote itself), the flag rate, and that DICTA's block mode holds up on long blocks.

### Open questions / possible improvements
- DICTA changed almost nothing on ivrit output. After the trial, consider `--skip-cleanup` (saves ~1.25 h) or keeping it only as a source for the review stage.
- The review stage may move a word across turns (it moved "מיכאל" into the right turn in the test). Currently allowed; watch for duplicated or dropped words.
- `--review-think` is untested (much slower).
- Flag thresholds (`LOW_WORD_PROBABILITY` etc. in `longform.py`) are untuned.
- Changing `--speaker-names` after the LLM stages ran re-runs them (the names are in their checkpoint key). Set names before step 3.
