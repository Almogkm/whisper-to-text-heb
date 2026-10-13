# WhatsApp Bot Plan

Research date: 2026-09-07

## Goal

Turn this local Hebrew voice-note transcriber into a WhatsApp bot:

1. A user forwards or sends a WhatsApp voice message to a dedicated bot number.
2. The bot downloads the audio.
3. The existing Hebrew transcription pipeline produces text.
4. The bot replies in WhatsApp with the transcript.

The target is family-and-friends usage, so privacy, reliability, and a low-maintenance deployment matter more than marketing-scale features.

## Recommendation

Use the official WhatsApp Business Cloud API for the real bot.

Avoid WhatsApp Web automation libraries for any phone number you care about. They are useful for learning and quick experiments, but most open-source voice-note bots that use them warn about breakage and account restriction risk. The official Cloud API needs a Meta developer app, a WhatsApp Business Account, and a dedicated business phone number, but it is the stable path.

## Current Project Fit

The current repo already has the hard part: Hebrew-first transcription from WhatsApp-style `.ogg` files.

Useful existing pieces:

- `SUPPORTED_EXTENSIONS` already includes `.ogg`, `.mp3`, `.m4a`, `.mp4`, `.wav`, `.webm`, and other common audio formats.
- The default language is `he`.
- The default quality path is `Whisper medium -> DICTA cleanup -> Qwen review`.
- The Markdown renderer preserves final, intermediate, and raw transcript stages for auditability.

Missing before a bot:

- A reusable pipeline function that returns structured data instead of only writing Markdown.
- A webhook server for WhatsApp events.
- Media download code for the Cloud API.
- A queue or worker so webhook responses return quickly.
- Runtime limits, auth, idempotency, and deletion policy for private audio.

## End-To-End Flow

```text
WhatsApp user
  -> forwards/sends voice note to bot number
  -> Meta WhatsApp Cloud API sends webhook
  -> webhook verifies request and extracts audio media ID
  -> webhook stores a queued job and returns 200 quickly
  -> worker fetches media URL from Graph API
  -> worker downloads audio with Bearer token
  -> worker runs local transcription pipeline
  -> bot sends text reply to original sender
  -> temporary audio is deleted
```

## Cloud API Mechanics

What the bot needs from Meta:

- `WHATSAPP_ACCESS_TOKEN`
- `WHATSAPP_PHONE_NUMBER_ID`
- `WHATSAPP_VERIFY_TOKEN`
- `WHATSAPP_APP_SECRET`
- `WHATSAPP_API_VERSION`

Required runtime behavior:

- `GET /webhooks/whatsapp` handles Meta's verify-token challenge.
- `POST /webhooks/whatsapp` validates `X-Hub-Signature-256` using the app secret.
- The webhook extracts `entry[].changes[].value.messages[]`.
- Only `type == "audio"` should enter the transcription queue in v1.
- Use the WhatsApp message ID as the idempotency key.
- For audio, read the media ID from the message's `audio.id`.
- Retrieve a temporary media URL with `GET /{media-id}`.
- Download the media URL with the same Bearer token.
- Reply with `POST /{phone-number-id}/messages`, preferably using the incoming `message_id` as reply context.

Meta's media endpoint supports audio up to 16 MB and accepts `audio/ogg` when the OGG uses Opus, which matches normal WhatsApp voice notes.

## Pricing Notes

As of 2026-09-07, WhatsApp's public pricing page says businesses are charged per delivered message, but service messages inside the 24-hour customer service window are not charged. For this use case, each user initiates the interaction by sending a voice note, so replies should be service messages inside that window.

Verify pricing again before public launch. WhatsApp Business Platform pricing changes often, and external provider pages already discuss future 2026 pricing changes.

## Why Not A Personal WhatsApp Bot

Unofficial WhatsApp Web bots are attractive because they can log in with a QR code and use a normal WhatsApp account. The tradeoff is operational risk:

- WhatsApp Web DOM/protocol changes can break the bot.
- Session storage becomes a production concern.
- Automation on a personal account can trigger restrictions.
- It is harder to explain privacy and support guarantees to family users.

Use unofficial libraries only with a throwaway test number if we want to study UX ideas.

## Open-Source Projects To Learn From

| Project | Shape | Useful lesson |
| --- | --- | --- |
| `mehnoorsiddiqui/whatsapp-voice-transcriber` | WhatsApp Cloud API plus OpenAI Whisper API | Closest public example to the official webhook/media path. Learn from the separation between WhatsApp audio handling and transcription. |
| `nerveband/whatsapp_voice_transcription` | Node bot with Baileys, OpenAI or Deepgram, summaries | Good config and Docker reference, but it uses an unofficial WhatsApp library. Useful for reply formatting, summaries, and deployment ergonomics. |
| `cr2007/whatsapp-voice-ai-assistant` | Go WhatsApp bot plus Python FastAPI transcription service using faster-whisper | Good architecture pattern: split WhatsApp transport from the transcription service, add trigger commands, add tests, run both with Docker Compose. It uses unofficial WhatsApp protocol libraries, so treat it as a design reference only. |
| `ayazalam/whatsapp-voice-note-transcriber` | Chrome extension for WhatsApp Web | Not a bot, but strong UX and privacy lessons: no backend, no telemetry, explicit click-to-transcribe, and direct insertion near the message. |
| `hoodini/whatsapp-public-logan` | Larger AI WhatsApp bot with voice features and Hebrew support | Useful for production controls: authorization lists, rate limits, queues, retries, and feature flags. It is much broader than this project needs. |
| Meta `fbsamples/whatsapp-api-examples` | Official examples | Use for webhook shape, media messages, and signature validation patterns. |

I did not find an open-source project that exactly matches "official WhatsApp Cloud API + local Hebrew Whisper/DICTA/Qwen pipeline". That is the useful niche for this repo.

## Lessons From `Tichtov Achi`

The older Israeli app `תכתוב אחי` is relevant for product behavior more than architecture.

Public reporting from 2017-2018 describes it as an Android share-sheet app: long-press a WhatsApp voice note, share it into the app, and receive a text rendering. Reports said it was Hebrew-first at launch and could reach roughly 90-95 percent accuracy on clear audio, with accuracy depending heavily on recording quality.

What to copy:

- Keep the input action simple: forward voice note to one contact.
- Return only the useful text by default.
- Add a confidence or quality warning when the transcript is likely weak.
- Optimize the short voice-note path aggressively.

What to improve:

- Add punctuation and paragraph cleanup.
- Add opt-in summaries only after the basic transcript is reliable.
- Preserve raw/intermediate text internally for debugging, but do not send all stages to WhatsApp by default.
- Let the user send a context/glossary message for names, family terms, and place names.

## Native WhatsApp Transcripts

WhatsApp now has native voice-message transcripts. The official help page says transcripts are generated on-device and language support depends on OS. Public Hebrew reporting from December 2024 found Hebrew available on iOS 17+, while Android support was still limited at that time.

This means the bot should not just duplicate WhatsApp's UI feature. It should compete on:

- Android Hebrew availability.
- Better punctuation and readability.
- Optional summaries.
- Cross-device consistency.
- Handling forwarded notes without asking each family member to enable the right local setting.

## Repo Implementation Plan

### Slice 1: Extract The Transcription Pipeline

Create `src/whisper_md_transcriber/pipeline.py`:

- `TranscriptionConfig`
- `TranscriptPackage`
- `run_transcription_job(config, audio_path) -> TranscriptPackage`

Then make `cli.py` use that function and keep `render_markdown()` as the CLI-only output layer.

The bot should read `TranscriptPackage.final_text`; it should not parse Markdown.

### Slice 2: Add A Minimal Webhook App

Create a small FastAPI app:

- `GET /webhooks/whatsapp` for verification.
- `POST /webhooks/whatsapp` for incoming messages.
- `GET /healthz` for uptime checks.

Dependencies to add:

- `fastapi`
- `uvicorn`
- `requests` or `httpx`
- a small queue backend for deployment, probably `rq` plus Redis or a SQLite-backed job table.

For the first local prototype, an in-process background task is enough as long as the webhook returns quickly. For anything persistent, use a real queue.

### Slice 3: Add WhatsApp Adapters

Create `src/whisper_md_transcriber/whatsapp/`:

- `webhook.py` parses webhook payloads.
- `client.py` wraps Graph API calls.
- `media.py` retrieves and downloads audio.
- `messages.py` formats transcript replies.
- `settings.py` loads environment variables.

Keep WhatsApp-specific code out of the transcription pipeline.

### Slice 4: Add Guardrails

Minimum guardrails before sharing the number:

- Whitelist allowed sender numbers during testing.
- Validate webhook signatures.
- Deduplicate by WhatsApp message ID.
- Reject non-audio messages with a short Hebrew usage reply.
- Enforce audio size and duration limits.
- Rate-limit by sender.
- Delete downloaded audio after processing.
- Store only minimal job metadata by default.
- Send a failure reply that says the bot could not transcribe the audio, without exposing stack traces.

### Slice 5: Improve Latency

The current quality pipeline may be too slow for a WhatsApp interaction if it runs entirely on CPU.

Latency options:

- Preload the Whisper model once at worker startup.
- Start with `medium -> DICTA` as the default bot path, then make Qwen review optional for longer or user-requested quality mode.
- Benchmark `faster-whisper` with CTranslate2 and CPU int8; its project reports up to 4x speedup over `openai/whisper` for the same accuracy and lower memory use.
- Consider a GPU-backed deployment if "near instant" is the goal.
- Return a short "processing" message only when expected latency is high; otherwise avoid extra chat noise.
- Cache results by media `sha256` when available, so forwarded duplicates are not transcribed twice.

## Local Prototype Setup

Manual setup outside the repo:

1. Create a Meta developer app and add WhatsApp.
2. Use Meta's test number first.
3. Add your own WhatsApp number as a test recipient.
4. Start the local webhook server:

```bash
conda activate whisper
uvicorn whisper_md_transcriber.whatsapp.webhook:app --host 0.0.0.0 --port 8000
```

5. Expose it with an HTTPS tunnel such as ngrok or Cloudflare Tunnel.
6. Configure the Meta webhook callback URL to `/webhooks/whatsapp`.
7. Send or forward a voice note to the test number.

## Deployment Options

Best first production-ish setup:

- A small VPS or home Ubuntu server for the webhook.
- Cloudflare Tunnel for public HTTPS if running at home.
- Redis or SQLite-backed queue.
- One long-running transcription worker.
- Audio retention disabled or limited to a few minutes.

If latency is disappointing:

- Move only the transcription worker to a GPU host.
- Keep the WhatsApp webhook on a cheap stable server.
- Keep the same pipeline API between the webhook and worker.

## Source Links

- WhatsApp Cloud API documentation: https://www.postman.com/meta/whatsapp-business-platform/documentation/wlk6lh4/whatsapp-cloud-api
- WhatsApp Cloud API media endpoints: https://www.postman.com/meta/whatsapp-business-platform/folder/13382743-ecb27be5-4d27-4763-bbee-6a8002c04bf3
- WhatsApp Business Platform pricing: https://whatsappbusiness.com/products/platform-pricing/
- WhatsApp Help Center voice-message transcripts: https://faq.whatsapp.com/241617298315321/
- Meta WhatsApp API examples: https://github.com/fbsamples/whatsapp-api-examples
- WhatsApp Cloud API voice transcriber example: https://github.com/mehnoorsiddiqui/whatsapp-voice-transcriber
- WhatsApp Web voice transcription example: https://github.com/nerveband/whatsapp_voice_transcription
- Go/Python voice assistant example: https://github.com/cr2007/whatsapp-voice-ai-assistant
- WhatsApp Web Chrome extension: https://github.com/ayazalam/whatsapp-voice-note-transcriber
- Logan WhatsApp AI bot: https://github.com/hoodini/whatsapp-public-logan
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Globes on `תכתוב אחי`: https://www.globes.co.il/news/article.aspx?did=1001218520
- Geektime on native Hebrew WhatsApp transcripts: https://www.geektime.co.il/whatsapp-voice-message-transcripts-now-live-in-israel/
