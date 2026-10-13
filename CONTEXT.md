# Whisper MD Transcriber

Local transcription tool that converts speech audio into Markdown transcript packages, with Hebrew-first defaults and layered local post-processing for readability and plausibility.

## Language

**Audio source**:
A local media file that contains the speech to transcribe.
_Avoid_: input, blob, asset

**Transcription job**:
A single requested conversion of one audio source into one Markdown transcript package.
_Avoid_: task, batch item, run

**Raw transcript**:
The direct speech-to-text output from Whisper before any cleanup or reconciliation.
_Avoid_: final transcript, reviewed transcript

**First-pass transcript**:
The transcript after the initial readability cleanup pass, currently produced by a Hebrew-specialized model.
_Avoid_: raw transcript, final transcript

**Reviewed transcript**:
The transcript after a second-pass reconciliation stage that compares the raw and first-pass transcripts and tries to rescue context mistakes.
_Avoid_: draft, raw transcript

**Transcript package**:
A Markdown file emitted for one transcription job, including metadata plus one or more transcript stages.
_Avoid_: export, note, artifact bundle

**Cleanup model**:
A local language model used to improve readability of the raw transcript without changing meaning.
_Avoid_: translator, summarizer

**Review model**:
A second local language model that works from the first-pass transcript and the raw transcript together to repair mistakes the cleanup model missed.
_Avoid_: responder, chat bot

**Speaker turn**:
A contiguous stretch of words attributed to one speaker, built from diarization plus word timestamps; the unit every LLM stage edits.
_Avoid_: segment, utterance, line

**Glossary**:
Names, places, and terms supplied by someone who knows the recording, collected through the Hebrew questionnaire and fed to Whisper hotwords and both LLM stages.
_Avoid_: dictionary, vocabulary list

**Alternative reading**:
The transcript from a second ASR model, aligned to the primary speaker turns by time and shown to the review model as extra evidence.
_Avoid_: backup transcript, second transcript

**Flagged turn**:
A speaker turn marked ⚠️ in the transcript and listed under "קטעים לבדיקה" because its evidence is weak (low confidence, model disagreement, a heavy review edit, or a failed LLM stage), so a person should listen to it.
_Avoid_: error, warning, low-quality segment
