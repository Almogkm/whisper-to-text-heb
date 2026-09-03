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
PREFERRED_CLEANUP_MODELS = [
    "qwen36-reader-16k",
    "qwen36-reader-32k",
    "qwen36-reader-64k",
    "qwen36-16k",
    "qwen-coder-16k",
]


@dataclass(frozen=True)
class AudioJob:
    source: Path
    relative_output: Path


@dataclass(frozen=True)
class CleanupResult:
    text: str
    model_name: str
    chunk_count: int
    status: str
    error: str | None = None


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
        help="Ollama model to use for transcript cleanup. Default: auto-select a local reader profile.",
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
        "No supported local cleanup model found. Expected one of: "
        + ", ".join(PREFERRED_CLEANUP_MODELS)
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


def build_cleanup_prompt(text: str, language: str) -> str:
    if language.lower() == "he":
        return (
            "אתה עורך תמלולים בעברית.\n"
            "תקן שגיאות כתיב, טעויות זיהוי ברורות, ניסוח וסימני פיסוק כדי שהטקסט יהיה קריא והגיוני יותר.\n"
            "אל תסכם, אל תתרגם, אל תוסיף עובדות חדשות, ואל תשמיט מידע.\n"
            "אם משהו אינו ברור, השאר אותו קרוב ככל האפשר למקור.\n"
            "החזר רק את הטקסט המתוקן, בלי כותרות ובלי הסברים.\n\n"
            "תמלול גולמי:\n"
            f"{text}"
        )

    return (
        "You are editing a speech transcript.\n"
        "Fix spelling, obvious recognition mistakes, phrasing, and punctuation so the text reads naturally.\n"
        "Do not summarize, translate, add facts, or omit content.\n"
        "If a fragment is unclear, keep it close to the source.\n"
        "Return only the corrected transcript text, with no heading and no explanation.\n\n"
        "Raw transcript:\n"
        f"{text}"
    )


def ollama_generate(
    *,
    ollama_url: str,
    model_name: str,
    prompt: str,
    timeout_seconds: int,
    num_predict: int,
) -> dict:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.1,
        },
    }
    request = Request(
        f"{ollama_url.rstrip('/')}/api/generate",
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
) -> CleanupResult:
    if not transcript_text.strip():
        return CleanupResult(
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
        prompt = build_cleanup_prompt(chunk, language)
        response = ollama_generate(
            ollama_url=ollama_url,
            model_name=model_name,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            num_predict=2048,
        )
        cleaned_text = str(response.get("response", "")).strip()
        if not cleaned_text:
            raise RuntimeError(
                "Ollama cleanup returned an empty response"
                + (f" (done_reason={response.get('done_reason')})" if response.get("done_reason") else "")
            )
        cleaned_chunks.append(cleaned_text)

    return CleanupResult(
        text="\n\n".join(cleaned_chunks).strip(),
        model_name=model_name,
        chunk_count=len(chunks),
        status="applied",
    )


def rtl_markdown_block(text: str) -> str:
    return f"<div dir=\"rtl\">\n\n{text.strip()}\n\n</div>\n"


def render_markdown(
    source: Path,
    raw_transcript: str,
    cleaned_transcript: str,
    *,
    model_name: str,
    language: str,
    generated_at: str,
    cleanup_result: CleanupResult,
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
                f"- מודל ניקוי: `{cleanup_result.model_name}`",
                f"- ניקוי טקסט: `applied`",
                f"- מספר מקטעי ניקוי: `{cleanup_result.chunk_count}`",
                "",
                "## תמליל משופר",
                "",
                rtl_markdown_block(cleaned_transcript),
                "",
                "## תמליל גולמי (Whisper)",
                "",
                rtl_markdown_block(raw_transcript),
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
    ollama_url: str,
    cleanup_timeout_seconds: int,
    cleanup_max_chars: int,
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
    cleanup_result = CleanupResult(
        text=transcript_text,
        model_name="",
        chunk_count=0,
        status="skipped",
    )
    cleaned_transcript = transcript_text

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
            cleaned_transcript = cleanup_result.text
        except Exception as exc:
            cleanup_result = CleanupResult(
                text=transcript_text,
                model_name=cleanup_model or "",
                chunk_count=0,
                status="failed",
                error=str(exc),
            )
            print(f"Cleanup failed for {job.source}: {exc}", file=sys.stderr)

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    markdown = render_markdown(
        job.source,
        transcript_text,
        cleaned_transcript,
        model_name=model_name,
        language=language,
        generated_at=generated_at,
        cleanup_result=cleanup_result,
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
                    ollama_url=args.ollama_url,
                    cleanup_timeout_seconds=args.cleanup_timeout_seconds,
                    cleanup_max_chars=args.cleanup_max_chars,
                )
            )
        except Exception as exc:
            print(f"Failed for {job.source}: {exc}", file=sys.stderr)
            return 1

    print("Wrote transcript files:", file=sys.stderr)
    for path in written_paths:
        print(path, file=sys.stderr)
    return 0
