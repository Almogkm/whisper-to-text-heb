# App Sharing

This project is already packaged as a CLI via `whisper_md_transcriber.cli`, but it is not yet an app people can use from anywhere.

## Current constraint

The runtime assumes:

- local `ffmpeg`
- local Whisper weights
- local Ollama on `http://127.0.0.1:11434`
- local DICTA and review models

That makes the current setup good for one machine, but not good for easy sharing with friends and family.

## Recommended path

1. Extract the orchestration in `cli.py` into a reusable pipeline such as `run_transcription_job(config, audio_job) -> transcript_package`.
2. Return a structured transcript package before rendering Markdown.
3. Add a thin web UI on top of that pipeline, preferably Gradio for the first version.
4. Dockerize the app so it can be deployed with `ffmpeg` and pinned Python dependencies.
5. Host it on a public service.

## Hosting choices

- Fastest prototype: Gradio with `share=True`, or a local web app exposed with Tailscale Funnel.
- Simplest permanent web app: Render or Railway with Docker.
- Best fit if the full heavy pipeline stays server-side or GPU use becomes important: Modal or a dedicated server.

## Practical recommendation

Ship a Whisper-only web app first, then make the DICTA cleanup and review stages optional follow-on layers. Keeping Ollama in the first public version will make deployment much heavier.

## Minimum safety bar

- add authentication
- set upload-size limits
- rate-limit requests
- delete uploaded audio after processing

Family voice notes should be treated as private data.
