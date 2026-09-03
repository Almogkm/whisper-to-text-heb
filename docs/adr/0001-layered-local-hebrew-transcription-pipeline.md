# Use a layered local pipeline for Hebrew transcript quality

This project produces transcripts locally and is optimized for Hebrew speech. We use Whisper for raw speech-to-text, a Hebrew-specialized DICTA pass for first cleanup, and a second local review pass over both the raw and cleaned text to rescue context errors the first pass misses. We chose this over Whisper-only or a single generic cleanup model because those paths left too many obvious Hebrew mistakes, while the layered local pipeline preserves privacy, avoids API cost, and keeps intermediate stages auditable in one Markdown output.

## Considered Options

- Whisper only
- Whisper plus one generic cleanup model
- Whisper plus one Hebrew-specialized cleanup model
- Whisper plus layered local cleanup and review
