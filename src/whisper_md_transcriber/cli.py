from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SUPPORTED_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_CLEANUP_MODEL = "dicta-hebrew-cleaner"
DEFAULT_CLEANUP_MODEL_SOURCE = "hf.co/dicta-il/DictaLM-3.0-Nemotron-12B-Instruct-GGUF:Q4_K_M"
DEFAULT_CLEANUP_MODELS = [
    DEFAULT_CLEANUP_MODEL,
    DEFAULT_CLEANUP_MODEL_SOURCE,
]
PREFERRED_CLEANUP_MODELS = [
    *DEFAULT_CLEANUP_MODELS,
]
DEFAULT_REVIEW_MODEL = "qwen36-reader-16k"
PREFERRED_REVIEW_MODELS = [
    DEFAULT_REVIEW_MODEL,
    "qwen36-thinker-16k",
    "gpt-oss:20b",
]


@dataclass(frozen=True)
class AudioJob:
    source: Path
    relative_output: Path


@dataclass(frozen=True)
class StageResult:
    text: str
    model_name: str
    chunk_count: int
    status: str
    error: str | None = None
    chunks: tuple[str, ...] = ()


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "whisper-md-transcriber"


def default_output_dir() -> Path:
    return PROJECT_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files to Markdown using a local OpenAI Whisper model."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Audio file(s) or directory/directories to transcribe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory to write Markdown files into. Default: the project directory.",
    )
    parser.add_argument(
        "--language",
        default="he",
        help="ISO-639-1 input language code. Default: he",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name. Default: small",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Default: auto",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional initial prompt, useful for Hebrew names or domain terms.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories for directory inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Markdown files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="Cache directory for downloaded Whisper model files.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip the Ollama cleanup pass and write only the raw Whisper transcript.",
    )
    parser.add_argument(
        "--cleanup-model",
        default=None,
        help=(
            "Ollama model to use for transcript cleanup. "
            f"Default: auto-select `{DEFAULT_CLEANUP_MODEL}` or its official Hugging Face source model."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama API base URL. Default: {DEFAULT_OLLAMA_URL}",
    )
    parser.add_argument(
        "--cleanup-max-chars",
        type=int,
        default=3500,
        help="Maximum transcript characters per cleanup chunk. Default: 3500",
    )
    parser.add_argument(
        "--cleanup-timeout-seconds",
        type=int,
        default=300,
        help="Timeout for each Ollama cleanup request. Default: 300",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip the second reconciliation pass and keep the DICTA output as the final transcript.",
    )
    parser.add_argument(
        "--review-model",
        default=None,
        help=(
            "Ollama model to use for the second transcript review pass. "
            f"Default: auto-select `{DEFAULT_REVIEW_MODEL}`."
        ),
    )
    parser.add_argument(
        "--review-max-chars",
        type=int,
        default=None,
        help="Maximum transcript characters per review chunk. Default: reuse --cleanup-max-chars",
    )
    parser.add_argument(
        "--review-timeout-seconds",
        type=int,
        default=None,
        help="Timeout for each review request. Default: reuse --cleanup-timeout-seconds",
    )
    return parser.parse_args(argv)


def normalize_inputs(raw_inputs: Iterable[str]) -> list[Path]:
    return [Path(item).expanduser().resolve() for item in raw_inputs]


def is_supported_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_jobs(inputs: list[Path], recursive: bool) -> list[AudioJob]:
    jobs: list[AudioJob] = []

    for item in inputs:
        if item.is_file():
            if not is_supported_audio_file(item):
                raise ValueError(f"Unsupported audio file: {item}")
            jobs.append(AudioJob(source=item, relative_output=Path(item.stem + ".md")))
            continue

        if item.is_dir():
            iterator = item.rglob("*") if recursive else item.glob("*")
            found_any = False
            for candidate in sorted(iterator):
                if is_supported_audio_file(candidate):
                    found_any = True
                    relative = candidate.relative_to(item).with_suffix(".md")
                    jobs.append(AudioJob(source=candidate, relative_output=relative))
            if not found_any:
                raise ValueError(f"No supported audio files found in directory: {item}")
            continue

        raise ValueError(f"Input path does not exist: {item}")

    return jobs


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return

    env_ffmpeg = Path(sys.executable).resolve().with_name("ffmpeg")
    if env_ffmpeg.exists():
        env_bin = str(env_ffmpeg.parent)
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = env_bin if not current_path else env_bin + os.pathsep + current_path
        return

    raise RuntimeError("ffmpeg is not available on PATH. Install it into the active conda env or system.")


def load_model(model_name: str, device: str, cache_dir: Path):
    import whisper

    model_dir = cache_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return whisper.load_model(model_name, device=device, download_root=str(model_dir))


def load_ollama_models(ollama_url: str, timeout_seconds: int) -> set[str]:
    request = Request(
        f"{ollama_url.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.code} while listing models") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {ollama_url}: {exc.reason}") from exc

    models = payload.get("models", [])
    names: set[str] = set()
    for item in models:
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and value:
                names.add(value)
                if value.endswith(":latest"):
                    names.add(value[:-7])
    return names


def resolve_cleanup_model(requested_model: str | None, ollama_url: str, timeout_seconds: int) -> str:
    available_models = load_ollama_models(ollama_url, timeout_seconds)

    if requested_model:
        if requested_model in available_models or f"{requested_model}:latest" in available_models:
            return requested_model
        raise RuntimeError(f"Requested cleanup model is not installed in Ollama: {requested_model}")

    for candidate in PREFERRED_CLEANUP_MODELS:
        if candidate in available_models or f"{candidate}:latest" in available_models:
            return candidate

    raise RuntimeError(
        "No supported local DICTA cleanup model found. "
        f"Install the default model with:\n"
        f"  ollama create {DEFAULT_CLEANUP_MODEL} -f {PROJECT_DIR / 'ollama' / 'DictaHebrewCleaner.Modelfile'}\n"
        "Or install the upstream model directly with:\n"
        f"  ollama run {DEFAULT_CLEANUP_MODEL_SOURCE}"
    )


def resolve_review_model(requested_model: str | None, ollama_url: str, timeout_seconds: int) -> str:
    available_models = load_ollama_models(ollama_url, timeout_seconds)

    if requested_model:
        if requested_model in available_models or f"{requested_model}:latest" in available_models:
            return requested_model
        raise RuntimeError(f"Requested review model is not installed in Ollama: {requested_model}")

    for candidate in PREFERRED_REVIEW_MODELS:
        if candidate in available_models or f"{candidate}:latest" in available_models:
            return candidate

    raise RuntimeError(
        "No supported local review model found. Expected one of: "
        + ", ".join(PREFERRED_REVIEW_MODELS)
    )


def split_cleanup_text(text: str, max_chars: int) -> list[str]:
    compact_text = " ".join(text.split())
    if not compact_text:
        return [""]
    if len(compact_text) <= max_chars:
        return [compact_text]

    words = compact_text.split(" ")
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        additional = len(word) if not current_words else len(word) + 1
        if current_words and current_length + additional > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += additional

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def sanitize_cleanup_response(text: str) -> str:
    cleaned = (
        text.replace("<|im_start|>", "\n")
        .replace("<|im_end|>", "\n")
        .replace("<|endoftext|>", "\n")
    )
    filtered_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"assistant", "user", "system"}:
            continue
        filtered_lines.append(stripped)
    return "\n".join(filtered_lines).strip()


def build_cleanup_messages(text: str, language: str) -> list[dict[str, str]]:
    if language.lower() == "he":
        return [
            {
                "role": "system",
                "content": (
                    "אתה עורך תמלולים של הודעות קוליות בעברית.\n"
                    "המטרה שלך היא להפוך תמלול גולמי לטקסט עברי ברור, טבעי ונאמן למקור.\n"
                    "כללים:\n"
                    "- תקן שגיאות כתיב, הטיות, רווחים, פיסוק וטעויות זיהוי ברורות.\n"
                    "- שמור את כל המידע המקורי.\n"
                    "- אל תסכם, אל תתרגם, אל תוסיף עובדות חדשות ואל תשמיט מידע.\n"
                    "- אל תהפוך את הטקסט לרשמי מדי; זה עדיין תמלול דיבור.\n"
                    "- אם קטע אינו ברור, בחר ניסוח שמרני שקרוב ככל האפשר למקור.\n"
                    "- החזר רק את התמליל המתוקן, בלי כותרות ובלי הסברים."
                ),
            },
            {
                "role": "user",
                "content": f"תמלול גולמי:\n{text}",
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are editing a speech transcript.\n"
                "Fix spelling, obvious recognition mistakes, phrasing, spacing, and punctuation.\n"
                "Keep all original information.\n"
                "Do not summarize, translate, add facts, or omit content.\n"
                "If a fragment is unclear, keep it conservative and close to the source.\n"
                "Return only the corrected transcript text, with no heading and no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Raw transcript:\n{text}",
        },
    ]


def build_review_messages(raw_text: str, cleaned_text: str, language: str) -> list[dict[str, str]]:
    if language.lower() == "he":
        return [
            {
                "role": "system",
                "content": (
                    "אתה מבצע מעבר שני על תמלול בעברית אחרי ניקוי ראשוני.\n"
                    "יש לפניך שני מקורות: תמלול גולמי של Whisper, ותמלול שעבר ניקוי ראשון.\n"
                    "המטרה שלך היא להחזיר גרסה סופית טובה יותר מהניקוי הראשון.\n"
                    "כללים:\n"
                    "- עבוד בעיקר מתוך התמליל שעבר ניקוי ראשון.\n"
                    "- כשיש צירוף מילים לא טבעי או לא הגיוני, בדוק מול התמליל הגולמי ונסה לשחזר את המילה הסבירה ביותר.\n"
                    "- תקן שגיאות זיהוי קוליות של מילים דומות בצליל, במיוחד במסמכים, שמות, מדינות, קשרי משפחה ומילים יומיומיות.\n"
                    "- שמור את כל המידע המקורי.\n"
                    "- אל תענה לדובר, אל תמשיך את השיחה, אל תסכם, אל תתרגם, אל תוסיף עובדות חדשות ואל תשמיט מידע.\n"
                    "- אם אינך בטוח, העדף תיקון שמרני שנשאר קרוב לגרסת הניקוי הראשון.\n"
                    "- החזר רק את התמליל הסופי, בלי כותרות ובלי הסברים."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"תמלול גולמי:\n{raw_text}\n\n"
                    f"תמלול אחרי ניקוי ראשון:\n{cleaned_text}"
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are performing a second pass over a speech transcript after an initial cleanup.\n"
                "You are given a raw Whisper transcript and a first-pass cleaned transcript.\n"
                "Return a final version that is better than the first-pass result.\n"
                "Work mainly from the cleaned transcript, but use the raw transcript to fix phrases that still sound implausible.\n"
                "Correct likely speech-recognition mistakes, keep all original information, and do not answer the speaker.\n"
                "Do not summarize, translate, add facts, or omit content.\n"
                "If uncertain, prefer a conservative edit close to the first-pass transcript.\n"
                "Return only the final corrected transcript, with no heading and no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Raw transcript:\n{raw_text}\n\n"
                f"First-pass cleaned transcript:\n{cleaned_text}"
            ),
        },
    ]


def ollama_chat(
    *,
    ollama_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    timeout_seconds: int,
    num_predict: int,
    temperature: float,
    think: bool | None = None,
) -> dict:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": temperature,
        },
    }
    if think is not None:
        payload["think"] = think
    request = Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {ollama_url}: {exc.reason}") from exc


def cleanup_transcript(
    transcript_text: str,
    *,
    language: str,
    requested_model: str | None,
    ollama_url: str,
    timeout_seconds: int,
    max_chars: int,
) -> StageResult:
    if not transcript_text.strip():
        return StageResult(
            text=transcript_text,
            model_name="",
            chunk_count=0,
            status="skipped-empty",
        )

    model_name = resolve_cleanup_model(requested_model, ollama_url, timeout_seconds)
    chunks = split_cleanup_text(transcript_text, max_chars=max_chars)
    cleaned_chunks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Cleanup [{index}/{len(chunks)}] with {model_name} ...", file=sys.stderr)
        messages = build_cleanup_messages(chunk, language)
        response = ollama_chat(
            ollama_url=ollama_url,
            model_name=model_name,
            messages=messages,
            timeout_seconds=timeout_seconds,
            num_predict=2048,
            temperature=0.1,
            think=False,
        )
        message = response.get("message")
        cleaned_text = ""
        if isinstance(message, dict):
            cleaned_text = sanitize_cleanup_response(str(message.get("content", "")))
        if not cleaned_text:
            raise RuntimeError(
                "Ollama cleanup returned an empty response"
                + (f" (done_reason={response.get('done_reason')})" if response.get("done_reason") else "")
            )
        cleaned_chunks.append(cleaned_text)

    return StageResult(
        text="\n\n".join(cleaned_chunks).strip(),
        model_name=model_name,
        chunk_count=len(chunks),
        status="applied",
        chunks=tuple(cleaned_chunks),
    )


def estimate_review_num_predict(text: str) -> int:
    compact_text = " ".join(text.split())
    estimated = len(compact_text) // 4 + 128
    return max(256, min(1024, estimated))


def review_transcript(
    raw_transcript: str,
    first_pass_result: StageResult,
    *,
    language: str,
    requested_model: str | None,
    ollama_url: str,
    timeout_seconds: int,
    max_chars: int,
) -> StageResult:
    if not first_pass_result.text.strip():
        return StageResult(
            text=first_pass_result.text,
            model_name="",
            chunk_count=0,
            status="skipped-empty",
        )

    model_name = resolve_review_model(requested_model, ollama_url, timeout_seconds)
    raw_chunks = split_cleanup_text(raw_transcript, max_chars=max_chars)
    cleaned_chunks = list(first_pass_result.chunks) or split_cleanup_text(first_pass_result.text, max_chars=max_chars)

    if len(raw_chunks) != len(cleaned_chunks):
        raw_chunks = [raw_transcript]
        cleaned_chunks = [first_pass_result.text]

    reviewed_chunks: list[str] = []

    for index, (raw_chunk, cleaned_chunk) in enumerate(zip(raw_chunks, cleaned_chunks), start=1):
        print(f"Review [{index}/{len(cleaned_chunks)}] with {model_name} ...", file=sys.stderr)
        messages = build_review_messages(raw_chunk, cleaned_chunk, language)
        response = ollama_chat(
            ollama_url=ollama_url,
            model_name=model_name,
            messages=messages,
            timeout_seconds=timeout_seconds,
            num_predict=estimate_review_num_predict(cleaned_chunk),
            temperature=0.0,
            think=False,
        )
        message = response.get("message")
        reviewed_text = ""
        if isinstance(message, dict):
            reviewed_text = sanitize_cleanup_response(str(message.get("content", "")))
        if not reviewed_text:
            raise RuntimeError(
                "Ollama review returned an empty response"
                + (f" (done_reason={response.get('done_reason')})" if response.get("done_reason") else "")
            )
        reviewed_chunks.append(reviewed_text)

    return StageResult(
        text="\n\n".join(reviewed_chunks).strip(),
        model_name=model_name,
        chunk_count=len(reviewed_chunks),
        status="applied",
        chunks=tuple(reviewed_chunks),
    )


def rtl_markdown_block(text: str) -> str:
    return f"<div dir=\"rtl\">\n\n{text.strip()}\n\n</div>\n"


def render_markdown(
    source: Path,
    raw_transcript: str,
    final_transcript: str,
    *,
    model_name: str,
    language: str,
    generated_at: str,
    cleanup_result: StageResult,
    review_result: StageResult,
) -> str:
    lines = [
        f"# תמלול: {source.name}",
        "",
        f"- מקור: `{source}`",
        f"- שפה: `{language}`",
        f"- מודל Whisper: `{model_name}`",
        f"- נוצר: `{generated_at}`",
    ]

    if cleanup_result.status == "applied":
        lines.extend(
            [
                f"- מודל ניקוי ראשון: `{cleanup_result.model_name}`",
                f"- ניקוי ראשון: `{cleanup_result.status}`",
                f"- מספר מקטעי ניקוי ראשון: `{cleanup_result.chunk_count}`",
            ]
        )

        if review_result.status == "applied":
            lines.extend(
                [
                    f"- מודל שיפור שני: `{review_result.model_name}`",
                    f"- שיפור שני: `{review_result.status}`",
                    f"- מספר מקטעי שיפור שני: `{review_result.chunk_count}`",
                    "",
                    "## תמליל סופי",
                    "",
                    rtl_markdown_block(final_transcript),
                    "",
                    "## תמליל אחרי שכבת DICTA",
                    "",
                    rtl_markdown_block(cleanup_result.text),
                    "",
                    "## תמליל גולמי (Whisper)",
                    "",
                    rtl_markdown_block(raw_transcript),
                ]
            )
        else:
            lines.extend(
                [
                    f"- שיפור שני: `{review_result.status}`",
                    "",
                    "## תמליל משופר",
                    "",
                    rtl_markdown_block(final_transcript),
                    "",
                    "## תמליל גולמי (Whisper)",
                    "",
                    rtl_markdown_block(raw_transcript),
                ]
            )
            if review_result.error:
                lines.extend(
                    [
                        "",
                        "## הערת שיפור שני",
                        "",
                        review_result.error,
                    ]
                )
    else:
        lines.extend(
            [
                f"- ניקוי טקסט: `{cleanup_result.status}`",
                "",
                "## תמליל",
                "",
                rtl_markdown_block(raw_transcript),
            ]
        )
        if cleanup_result.error:
            lines.extend(
                [
                    "",
                    "## הערת ניקוי",
                    "",
                    cleanup_result.error,
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def destination_for(job: AudioJob, output_dir: Path | None) -> Path:
    if output_dir is None:
        return default_output_dir() / job.relative_output
    return output_dir / job.relative_output


def transcribe_job(
    job: AudioJob,
    *,
    model,
    device: str,
    output_dir: Path | None,
    overwrite: bool,
    language: str,
    model_name: str,
    prompt: str | None,
    cleanup_enabled: bool,
    cleanup_model: str | None,
    review_enabled: bool,
    review_model: str | None,
    ollama_url: str,
    cleanup_timeout_seconds: int,
    cleanup_max_chars: int,
    review_timeout_seconds: int,
    review_max_chars: int,
) -> Path:
    destination = destination_for(job, output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")

    use_fp16 = device == "cuda"
    result = model.transcribe(
        str(job.source),
        task="transcribe",
        language=language,
        initial_prompt=prompt,
        fp16=use_fp16,
        verbose=False,
    )
    transcript_text = result.get("text", "").strip()
    cleanup_result = StageResult(
        text=transcript_text,
        model_name="",
        chunk_count=0,
        status="skipped",
    )
    review_result = StageResult(
        text="",
        model_name="",
        chunk_count=0,
        status="skipped",
    )
    final_transcript = transcript_text

    if cleanup_enabled and transcript_text:
        try:
            cleanup_result = cleanup_transcript(
                transcript_text,
                language=language,
                requested_model=cleanup_model,
                ollama_url=ollama_url,
                timeout_seconds=cleanup_timeout_seconds,
                max_chars=cleanup_max_chars,
            )
            final_transcript = cleanup_result.text
        except Exception as exc:
            cleanup_result = StageResult(
                text=transcript_text,
                model_name=cleanup_model or "",
                chunk_count=0,
                status="failed",
                error=str(exc),
            )
            print(f"Cleanup failed for {job.source}: {exc}", file=sys.stderr)

    if cleanup_result.status == "applied" and review_enabled and cleanup_result.text.strip():
        try:
            review_result = review_transcript(
                transcript_text,
                cleanup_result,
                language=language,
                requested_model=review_model,
                ollama_url=ollama_url,
                timeout_seconds=review_timeout_seconds,
                max_chars=review_max_chars,
            )
            final_transcript = review_result.text
        except Exception as exc:
            review_result = StageResult(
                text=cleanup_result.text,
                model_name=review_model or "",
                chunk_count=0,
                status="failed",
                error=str(exc),
            )
            final_transcript = cleanup_result.text
            print(f"Second-pass review failed for {job.source}: {exc}", file=sys.stderr)

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    markdown = render_markdown(
        job.source,
        transcript_text,
        final_transcript,
        model_name=model_name,
        language=language,
        generated_at=generated_at,
        cleanup_result=cleanup_result,
        review_result=review_result,
    )
    destination.write_text(markdown, encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir.expanduser().resolve()
    inputs = normalize_inputs(args.inputs)

    try:
        jobs = discover_jobs(inputs, recursive=args.recursive)
        ensure_ffmpeg_available()
        device = resolve_device(args.device)
        model = load_model(args.model, device=device, cache_dir=cache_dir)
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    written_paths: list[Path] = []

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] Transcribing {job.source} ...", file=sys.stderr)
        try:
            written_paths.append(
                transcribe_job(
                    job,
                    model=model,
                    device=device,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                    language=args.language,
                    model_name=args.model,
                    prompt=args.prompt,
                    cleanup_enabled=not args.skip_cleanup,
                    cleanup_model=args.cleanup_model,
                    review_enabled=not args.skip_review,
                    review_model=args.review_model,
                    ollama_url=args.ollama_url,
                    cleanup_timeout_seconds=args.cleanup_timeout_seconds,
                    cleanup_max_chars=args.cleanup_max_chars,
                    review_timeout_seconds=args.review_timeout_seconds or args.cleanup_timeout_seconds,
                    review_max_chars=args.review_max_chars or args.cleanup_max_chars,
                )
            )
        except Exception as exc:
            print(f"Failed for {job.source}: {exc}", file=sys.stderr)
            return 1

    print("Wrote transcript files:", file=sys.stderr)
    for path in written_paths:
        print(path, file=sys.stderr)
    return 0
