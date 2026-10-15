"""Long-form, speaker-attributed transcription pipeline.

Stages, each checkpointed as JSON under the work directory so a multi-hour run
can resume after an interruption:

    prepare -> asr (one pass per Whisper model) -> diarize -> align
            -> cleanup (DICTA) -> review (Qwen) -> render (Markdown, SRT, JSON)

The first ASR model is the primary transcript. Extra ASR models are aligned to
the primary speaker turns and handed to the review stage as alternative
readings. Speaker labels never pass through an LLM: every LLM request carries
turn ids, and the output is rejected unless the same ids come back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from .cli import (
    PROJECT_DIR,
    ensure_ffmpeg_available,
    ollama_chat,
    resolve_cleanup_model,
    resolve_review_model,
    rtl_markdown_block,
    sanitize_cleanup_response,
)

DEFAULT_ASR_MODELS = (
    "ivrit-ai/whisper-large-v3-ct2",
    "Systran/faster-whisper-large-v3",
)
GATED_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
OPEN_DIARIZATION_MODEL = "ivrit-ai/pyannote-speaker-diarization-3.1"
SAMPLE_RATE = 16000

# Turn building.
MAX_TURN_CHARS = 600
MAX_TURN_GAP_SECONDS = 1.5
MAX_FLICKER_SECONDS = 0.4
MIN_DIARIZATION_TURN_SECONDS = 0.15

# LLM batching. Review blocks are smaller because each turn carries three versions.
CLEANUP_BLOCK_CHARS = 2500
REVIEW_BLOCK_CHARS = 1500
CLEANUP_NUM_CTX = 8192

# Reject an LLM rewrite of a turn when its length drifts this far from the source.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 1.8
MAX_WORD_CHANGE = 0.6

# Review flags.
LOW_WORD_PROBABILITY = 0.3
LOW_MEAN_PROBABILITY = 0.6
HIGH_ASR_DISAGREEMENT = 0.35
HIGH_REVIEW_CHANGE = 0.3

GLOSSARY_CATEGORY_LABELS = {
    "people": "אדם",
    "places": "מקום",
    "organizations": "ארגון",
    "terms": "מונח",
    "foreign": "מילה לועזית",
    "other": "אחר",
}
GENDER_LABELS = {"female": "אישה", "male": "גבר"}

TURN_MARKER = re.compile(r"^\s*<<\s*(\d+)\s*(?:\|[^>]*)?>>\s*(.*)$")
BIDI_CONTROLS = re.compile("[‎‏‪-‮⁦-⁩]")
PUNCTUATION = re.compile(r"[^\w\s]")


# ---------------------------------------------------------------------------
# Glossary


@dataclass(frozen=True)
class GlossaryEntry:
    category: str
    term: str
    note: str


@dataclass(frozen=True)
class GlossarySpeaker:
    name: str
    gender: str
    description: str


@dataclass(frozen=True)
class Glossary:
    context: str
    speakers: tuple[GlossarySpeaker, ...]
    entries: tuple[GlossaryEntry, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.context or self.speakers or self.entries)


EMPTY_GLOSSARY = Glossary(context="", speakers=(), entries=())


def load_glossary(path: Path | None) -> Glossary:
    """Read the JSON file exported by docs/glossary-questionnaire.html."""
    if path is None:
        return EMPTY_GLOSSARY

    payload = json.loads(path.read_text(encoding="utf-8"))
    recording = payload.get("recording", {})
    context_parts = [
        str(recording.get(key, "")).strip()
        for key in ("topic", "setting", "background")
        if str(recording.get(key, "")).strip()
    ]

    speakers = tuple(
        GlossarySpeaker(
            name=str(item.get("name", "")).strip(),
            gender=str(item.get("gender", "")).strip(),
            description=str(item.get("description", "")).strip(),
        )
        for item in payload.get("speakers", [])
        if str(item.get("name", "")).strip()
    )

    entries: list[GlossaryEntry] = []
    for speaker in speakers:
        entries.append(GlossaryEntry(category="people", term=speaker.name, note="דובר בהקלטה"))
    for category in GLOSSARY_CATEGORY_LABELS:
        for item in payload.get(category, []):
            term = str(item.get("term", "")).strip()
            if term:
                entries.append(
                    GlossaryEntry(category=category, term=term, note=str(item.get("note", "")).strip())
                )

    return Glossary(context=" ".join(context_parts), speakers=speakers, entries=tuple(entries))


def whisper_prompts(glossary: Glossary) -> tuple[str | None, str | None]:
    """Return (initial_prompt, hotwords) for faster-whisper.

    The initial prompt only conditions the first window; hotwords are injected
    into every window, so the terms go there and the free-text context goes
    into the initial prompt. faster-whisper truncates both to the decoder limit.
    """
    initial_prompt = glossary.context[:300] or None
    seen: set[str] = set()
    terms: list[str] = []
    for entry in glossary.entries:
        if entry.term not in seen:
            seen.add(entry.term)
            terms.append(entry.term)
    hotwords = ", ".join(terms)[:400] or None
    return initial_prompt, hotwords


def glossary_for_llm(glossary: Glossary) -> str:
    if glossary.is_empty:
        return ""

    lines = ["מידע רקע על ההקלטה (השתמש בו כדי לזהות מילים ושמות, אל תוסיף אותו לתמליל):"]
    if glossary.context:
        lines.append(f"נושא והקשר: {glossary.context}")
    if glossary.speakers:
        lines.append("הדוברים בהקלטה:")
        for speaker in glossary.speakers:
            details = ", ".join(
                part for part in (GENDER_LABELS.get(speaker.gender, ""), speaker.description) if part
            )
            lines.append(f"- {speaker.name}" + (f" ({details})" if details else ""))
    entries = [entry for entry in glossary.entries if entry.note != "דובר בהקלטה"]
    if entries:
        lines.append("שמות ומונחים שעשויים להופיע. כשהשמע קרוב לאחד מהם, העדף את הכתיב הזה:")
        for entry in entries:
            label = GLOSSARY_CATEGORY_LABELS.get(entry.category, "")
            note = f": {entry.note}" if entry.note else ""
            lines.append(f"- {entry.term} ({label}){note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoints


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def load_checkpoint(path: Path, params: dict) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("params") != params:
        return None
    return payload


def save_checkpoint(path: Path, params: dict, data: object) -> dict:
    payload = {"params": params, "data": data}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp_path.replace(path)
    return payload


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Audio


def prepare_audio(source: Path, destination: Path, loudnorm: bool) -> float:
    ensure_ffmpeg_available()
    filters = ["loudnorm=I=-16:TP=-1.5:LRA=11"] if loudnorm else []
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ac", "1", "-ar", str(SAMPLE_RATE)]
    if filters:
        command += ["-af", ",".join(filters)]
    command += ["-c:a", "pcm_s16le", str(destination)]
    subprocess.run(command, check=True)
    return wav_duration(destination)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def read_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_rate = handle.getframerate()
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, sample_rate


# ---------------------------------------------------------------------------
# ASR


def resolve_asr_device() -> str:
    import ctranslate2

    return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"


def run_asr(
    wav_path: Path,
    model_id: str,
    *,
    duration: float,
    compute_type: str,
    beam_size: int,
    initial_prompt: str | None,
    hotwords: str | None,
) -> list[dict]:
    from faster_whisper import WhisperModel

    device = resolve_asr_device()
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "float32"
    log(f"ASR {model_id} on {device} ({compute_type}, beam {beam_size}) ...")
    model = WhisperModel(model_id, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 0)
    segments, _info = model.transcribe(
        str(wav_path),
        language="he",
        task="transcribe",
        beam_size=beam_size,
        best_of=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        condition_on_previous_text=True,
        hallucination_silence_threshold=2.0,
    )

    results: list[dict] = []
    started = time.monotonic()
    last_report = 0.0
    for segment in segments:
        results.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "words": [
                    {"start": word.start, "end": word.end, "word": word.word, "probability": word.probability}
                    for word in (segment.words or [])
                ],
            }
        )
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 60:
            last_report = elapsed
            log(f"  {model_id}: {segment.end / duration:.0%} of audio after {elapsed / 60:.1f} min")
    del model
    return results


def asr_words(segments: list[dict]) -> list[dict]:
    return [word for segment in segments for word in segment["words"] if word["word"].strip()]


# ---------------------------------------------------------------------------
# Diarization


def resolve_diarization_model(requested: str | None, token: str | None) -> str:
    if requested:
        return requested
    return GATED_DIARIZATION_MODEL if token else OPEN_DIARIZATION_MODEL


def resolve_hf_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    from huggingface_hub import get_token

    return get_token()


def run_diarization(
    wav_path: Path,
    model_id: str,
    *,
    token: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[dict]:
    import torch
    from huggingface_hub.errors import GatedRepoError
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines import speaker_diarization
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    # pyannote 4 always loads PLDA weights from the gated community-1 repo, but
    # only VBx clustering uses them. 3.x pipelines (agglomerative clustering)
    # run fine without, so tolerate the gate instead of demanding a token.
    strict_get_plda = speaker_diarization.get_plda

    def lenient_get_plda(plda, token=None, cache_dir=None):
        try:
            return strict_get_plda(plda, token=token, cache_dir=cache_dir)
        except GatedRepoError:
            log("PLDA weights are gated; not needed by this pipeline, ignore the message above.")
            return None

    speaker_diarization.get_plda = lenient_get_plda

    log(f"Diarization with {model_id} ...")
    pipeline = Pipeline.from_pretrained(model_id, token=token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {model_id}. For gated pyannote models, accept the terms on Hugging Face"
            " and log in with `hf auth login`."
        )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    samples, sample_rate = read_wav(wav_path)
    audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": sample_rate}
    kwargs: dict[str, int] = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    with ProgressHook() as hook:
        output = pipeline(audio, hook=hook, **kwargs)

    # pyannote 4 returns an object with an overlap-free "exclusive" diarization,
    # which is what word-to-speaker assignment wants. Older pipelines return the
    # annotation directly.
    annotation = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", None)
        or output
    )
    return [
        {"start": float(segment.start), "end": float(segment.end), "speaker": str(label)}
        for segment, _track, label in annotation.itertracks(yield_label=True)
    ]


# ---------------------------------------------------------------------------
# Alignment


def assign_speakers(words: list[dict], diarization: list[dict]) -> list[str]:
    """Give each word the speaker with the most time overlap, else the nearest diarization turn.

    pyannote emits slivers (a few hundredths of a second) at speaker changes;
    they are ignored so a word in a short gap does not snap to one.
    """
    turns = sorted(
        (turn for turn in diarization if turn["end"] - turn["start"] >= MIN_DIARIZATION_TURN_SECONDS),
        key=lambda item: item["start"],
    ) or sorted(diarization, key=lambda item: item["start"])
    if not turns:
        return ["SPEAKER_00"] * len(words)

    starts = [turn["start"] for turn in turns]
    longest = max(turn["end"] - turn["start"] for turn in turns)
    speakers: list[str] = []
    for word in words:
        overlap: dict[str, float] = {}
        index = bisect_right(starts, word["end"])
        nearby = range(max(0, bisect_right(starts, word["start"] - longest) - 1), min(len(turns), index + 1))
        for i in nearby:
            turn = turns[i]
            shared = min(word["end"], turn["end"]) - max(word["start"], turn["start"])
            if shared > 0:
                overlap[turn["speaker"]] = overlap.get(turn["speaker"], 0.0) + shared
        if overlap:
            speakers.append(max(overlap, key=overlap.get))
            continue
        midpoint = (word["start"] + word["end"]) / 2
        nearest = min(
            (turns[i] for i in nearby),
            key=lambda turn: min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"])),
        )
        speakers.append(nearest["speaker"])
    return speakers


def smooth_speakers(words: list[dict], speakers: list[str]) -> list[str]:
    """Remove one-word speaker flips sandwiched inside another speaker's turn (timing jitter)."""
    smoothed = list(speakers)
    for index in range(1, len(words) - 1):
        previous, current, following = smoothed[index - 1], smoothed[index], smoothed[index + 1]
        word = words[index]
        if current != previous and previous == following and word["end"] - word["start"] <= MAX_FLICKER_SECONDS:
            smoothed[index] = previous
    return smoothed


def speaker_order(speakers: list[str]) -> dict[str, int]:
    order: dict[str, int] = {}
    for speaker in speakers:
        order.setdefault(speaker, len(order) + 1)
    return order


def build_turns(words: list[dict], speakers: list[str]) -> list[dict]:
    order = speaker_order(speakers)
    turns: list[dict] = []
    for word, speaker in zip(words, speakers):
        current = turns[-1] if turns else None
        if (
            current is not None
            and current["speaker"] == speaker
            and word["start"] - current["end"] <= MAX_TURN_GAP_SECONDS
            and len(current["raw"]) + len(word["word"]) <= MAX_TURN_CHARS
        ):
            current["raw"] += word["word"]
            current["end"] = word["end"]
            current["probabilities"].append(word["probability"])
            current["words"].append(word["word"].strip())
            continue
        turns.append(
            {
                "id": len(turns) + 1,
                "speaker": speaker,
                "speaker_number": order[speaker],
                "start": word["start"],
                "end": word["end"],
                "raw": word["word"],
                "probabilities": [word["probability"]],
                "words": [word["word"].strip()],
            }
        )
    for turn in turns:
        turn["raw"] = " ".join(turn["raw"].split())
    return turns


def attach_alternative(turns: list[dict], alt_words: list[dict], key: str) -> None:
    """Attach the words of another ASR pass whose midpoints fall inside each turn's time span."""
    index = 0
    for position, turn in enumerate(turns):
        span_end = turns[position + 1]["start"] if position + 1 < len(turns) else float("inf")
        collected: list[str] = []
        while index < len(alt_words):
            word = alt_words[index]
            midpoint = (word["start"] + word["end"]) / 2
            if midpoint >= span_end:
                break
            collected.append(word["word"])
            index += 1
        turn[key] = " ".join("".join(collected).split())


def strip_bidi(text: str) -> str:
    """Drop invisible direction controls (LRM/RLM, embeddings) that ivrit sometimes emits."""
    return BIDI_CONTROLS.sub("", text)


def comparable_words(text: str) -> list[str]:
    return PUNCTUATION.sub("", strip_bidi(text)).split()


def word_change_ratio(before: str, after: str) -> float:
    """Share of words that differ, ignoring punctuation and direction controls."""
    before_words, after_words = comparable_words(before), comparable_words(after)
    if not before_words and not after_words:
        return 0.0
    return 1.0 - SequenceMatcher(None, before_words, after_words).ratio()


# ---------------------------------------------------------------------------
# LLM stages


def speaker_label(turn: dict, names: list[str]) -> str:
    number = turn["speaker_number"]
    if number <= len(names) and names[number - 1]:
        return names[number - 1]
    return f"דובר {number}"


def blocks_of(turns: list[dict], key: str, max_chars: int) -> list[list[dict]]:
    blocks: list[list[dict]] = []
    size = 0
    for turn in turns:
        length = len(turn[key])
        if blocks and size + length <= max_chars:
            blocks[-1].append(turn)
            size += length
        else:
            blocks.append([turn])
            size = length
    return blocks


def parse_marked_lines(text: str) -> dict[int, str]:
    parsed: dict[int, str] = {}
    current: int | None = None
    for line in text.splitlines():
        match = TURN_MARKER.match(line)
        if match:
            current = int(match.group(1))
            parsed[current] = match.group(2).strip()
        elif current is not None and line.strip():
            parsed[current] = f"{parsed[current]} {line.strip()}".strip()
    return parsed


def plausible_rewrite(source: str, rewrite: str) -> bool:
    """Reject rewrites that lost or gained content, or that belong to another turn."""
    if not rewrite.strip():
        return False
    if len(source) < 20:
        return len(rewrite) <= len(source) + 20
    ratio = len(rewrite) / len(source)
    if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
        return False
    return word_change_ratio(source, rewrite) <= MAX_WORD_CHANGE


FORMAT_RULES_HE = (
    "פורמט: הטקסט מחולק לתורות דיבור. כל תור מתחיל בסימון בצורה <<מספר|דובר>>.\n"
    "החזר כל תור בשורה אחת שמתחילה באותו סימון בדיוק, באותו סדר, ואחריו הטקסט המתוקן.\n"
    "אל תמזג תורות, אל תפצל תורות, אל תשמיט תורות ואל תוסיף תורות. אל תעביר מילים מתור לתור.\n"
    "החזר רק את השורות המסומנות, בלי כותרות ובלי הסברים."
)


def build_cleanup_messages(block: list[dict], names: list[str], glossary_text: str) -> list[dict[str, str]]:
    system = (
        "אתה עורך תמלולים של שיחות מוקלטות בעברית.\n"
        "המטרה שלך היא להפוך תמלול גולמי לטקסט עברי ברור, טבעי ונאמן למקור.\n"
        "כללים:\n"
        "- תקן שגיאות כתיב, הטיות, רווחים, פיסוק וטעויות זיהוי ברורות.\n"
        "- התאם זכר ונקבה לדובר ולמי שמדברים אליו כשזה ברור מההקשר.\n"
        "- שמור את כל המידע המקורי.\n"
        "- אל תסכם, אל תתרגם, אל תוסיף עובדות חדשות ואל תשמיט מידע.\n"
        "- אל תהפוך את הטקסט לרשמי מדי; זה עדיין תמלול דיבור.\n"
        "- אם קטע אינו ברור, בחר ניסוח שמרני שקרוב ככל האפשר למקור.\n\n"
        f"{FORMAT_RULES_HE}"
    )
    if glossary_text:
        system += f"\n\n{glossary_text}"
    lines = [f"<<{turn['id']}|{speaker_label(turn, names)}>> {turn['raw']}" for turn in block]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "תמלול גולמי:\n" + "\n".join(lines)},
    ]


def build_review_messages(block: list[dict], names: list[str], glossary_text: str) -> list[dict[str, str]]:
    has_alternative = any(turn.get("alt") for turn in block)
    sources = (
        "לכל תור יש שלוש גרסאות: 'גולמי א' (מודל זיהוי דיבור שאומן על עברית), "
        "'גולמי ב' (מודל זיהוי דיבור כללי), ו'ניקוי' (אחרי ניקוי ראשון)."
        if has_alternative
        else "לכל תור יש שתי גרסאות: 'גולמי' (זיהוי דיבור) ו'ניקוי' (אחרי ניקוי ראשון)."
    )
    instructions = (
        "בצע ביקורת סופית לתמלול של שיחה מוקלטת בעברית.\n"
        f"{sources}\n"
        "עבוד מהניקוי כבסיס. כשיש בו מילה או צירוף לא טבעי, לא הגיוני או לא דקדוקי,"
        " בדוק מול הגרסאות הגולמיות ובחר את המילה או הניסוח הסבירים ביותר.\n"
        "כששני המודלים הגולמיים מסכימים על מילה והניקוי שינה אותה, שקול ברצינות להחזיר אותה.\n"
        "בדוק במיוחד שמות, מילות יחס, קרובי משפחה, שמות מקומות והתאמת זכר ונקבה לדובר.\n"
        "אל תענה לדוברים, אל תסכם ואל תוסיף מידע.\n\n"
        f"{FORMAT_RULES_HE}"
    )
    if glossary_text:
        instructions += f"\n\n{glossary_text}"

    parts: list[str] = []
    for turn in block:
        header = f"<<{turn['id']}|{speaker_label(turn, names)}>>"
        if has_alternative:
            parts.append(
                f"{header}\nגולמי א: {turn['raw']}\nגולמי ב: {turn.get('alt', '')}\nניקוי: {turn['cleaned']}"
            )
        else:
            parts.append(f"{header}\nגולמי: {turn['raw']}\nניקוי: {turn['cleaned']}")
    return [{"role": "user", "content": instructions + "\n\nהתורות:\n\n" + "\n\n".join(parts)}]


def estimate_num_predict(block: list[dict], key: str) -> int:
    characters = sum(len(turn[key]) + 16 for turn in block)
    return max(512, min(6144, characters + 256))


def run_llm_stage(
    stage: str,
    turns: list[dict],
    *,
    source_key: str,
    block_chars: int,
    build_messages,
    model_name: str,
    ollama_url: str,
    timeout_seconds: int,
    num_ctx: int | None,
    think: bool,
    progress_path: Path,
    progress_params: dict,
) -> dict[int, dict]:
    """Rewrite every turn's `source_key` text. Returns {turn id: {"text", "status"}}.

    Progress is appended to a JSONL file after every block, so an interrupted
    run continues from the last finished block.
    """
    done: dict[int, dict] = {}
    if progress_path.exists():
        lines = progress_path.read_text(encoding="utf-8").splitlines()
        if lines and json.loads(lines[0]).get("params") == progress_params:
            for line in lines[1:]:
                record = json.loads(line)
                done[int(record["id"])] = {"text": record["text"], "status": record["status"]}
        else:
            done = {}
            progress_path.write_text(json.dumps({"params": progress_params}) + "\n", encoding="utf-8")
    else:
        progress_path.write_text(json.dumps({"params": progress_params}) + "\n", encoding="utf-8")

    pending = [turn for turn in turns if turn["id"] not in done]
    blocks = blocks_of(pending, source_key, block_chars)
    if done:
        log(f"{stage}: resuming, {len(done)} turns already done, {len(blocks)} blocks left")

    def request(block: list[dict]) -> dict[int, str]:
        response = ollama_chat(
            ollama_url=ollama_url,
            model_name=model_name,
            messages=build_messages(block),
            timeout_seconds=timeout_seconds,
            num_predict=estimate_num_predict(block, source_key) * (4 if think else 1),
            temperature=0.0,
            think=think,
            num_ctx=num_ctx,
        )
        message = response.get("message")
        content = sanitize_cleanup_response(str(message.get("content", ""))) if isinstance(message, dict) else ""
        parsed = parse_marked_lines(content)
        if not parsed and len(block) == 1 and content:
            parsed = {block[0]["id"]: content}
        return parsed

    started = time.monotonic()
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, block in enumerate(blocks, start=1):
            log(f"{stage} [{index}/{len(blocks)}] turns {block[0]['id']}-{block[-1]['id']} with {model_name}")
            results: dict[int, dict] = {}
            try:
                parsed = request(block)
            except Exception as exc:
                log(f"  request failed: {exc}")
                parsed = {}

            rejected = [
                turn for turn in block if not plausible_rewrite(turn[source_key], parsed.get(turn["id"], ""))
            ]
            if rejected and len(block) > 1:
                log(f"  {len(rejected)}/{len(block)} turns did not round-trip; retrying them one by one")
                for turn in rejected:
                    try:
                        single = request([turn])
                        if len(single) == 1:
                            parsed[turn["id"]] = next(iter(single.values()))
                    except Exception as exc:
                        log(f"  turn {turn['id']} failed: {exc}")

            for turn in block:
                rewrite = parsed.get(turn["id"], "")
                if plausible_rewrite(turn[source_key], rewrite):
                    results[turn["id"]] = {"text": rewrite, "status": "applied"}
                else:
                    results[turn["id"]] = {"text": turn[source_key], "status": "kept-source"}

            for turn_id, result in results.items():
                progress.write(json.dumps({"id": turn_id, **result}, ensure_ascii=False) + "\n")
            progress.flush()
            done.update(results)

            elapsed = time.monotonic() - started
            remaining = elapsed / index * (len(blocks) - index)
            log(f"  {elapsed / 60:.1f} min elapsed, ~{remaining / 60:.0f} min left in {stage}")

    return done


# ---------------------------------------------------------------------------
# Flags and rendering


def compute_flags(turn: dict) -> list[str]:
    flags: list[str] = []
    probabilities = turn["probabilities"]
    low_words = [word for word, p in zip(turn["words"], probabilities) if p < LOW_WORD_PROBABILITY]
    mean_probability = sum(probabilities) / len(probabilities) if probabilities else 1.0
    if low_words:
        flags.append("מילים בביטחון נמוך: " + ", ".join(low_words[:6]))
    elif mean_probability < LOW_MEAN_PROBABILITY:
        flags.append(f"ביטחון ממוצע נמוך ({mean_probability:.0%})")
    if turn.get("alt") is not None:
        disagreement = word_change_ratio(turn["raw"], turn["alt"])
        if disagreement > HIGH_ASR_DISAGREEMENT:
            flags.append(f"מודלי הזיהוי חלוקים ({disagreement:.0%})")
    if turn.get("cleaned") and turn.get("final"):
        change = word_change_ratio(turn["cleaned"], turn["final"])
        if change > HIGH_REVIEW_CHANGE:
            flags.append(f"הביקורת שינתה הרבה ({change:.0%})")
    for stage in ("cleanup", "review"):
        if turn.get(f"{stage}_status") == "kept-source":
            flags.append(f"שלב {stage} לא הצליח, נשאר הטקסט הקודם")
    return flags


def format_timestamp(seconds: float, separator: str = ".") -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if separator == ",":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_markdown(source: Path, turns: list[dict], names: list[str], metadata: dict) -> str:
    lines = [f"# תמלול: {source.name}", ""]
    for label, value in metadata.items():
        lines.append(f"- {label}: `{value}`")

    transcript_lines: list[str] = []
    for turn in turns:
        mark = " ⚠️" if turn["flags"] else ""
        transcript_lines.append(
            f"**[{format_timestamp(turn['start'])}] {speaker_label(turn, names)}:** {turn['final']}{mark}"
        )
    lines += ["", "## תמליל", "", rtl_markdown_block("\n\n".join(transcript_lines))]

    flagged = [turn for turn in turns if turn["flags"]]
    lines += ["", f"## קטעים לבדיקה ({len(flagged)})", ""]
    if flagged:
        review_lines = [
            f"- **[{format_timestamp(turn['start'])}] {speaker_label(turn, names)}:** " + "; ".join(turn["flags"])
            for turn in flagged
        ]
        lines.append(rtl_markdown_block("\n".join(review_lines)))
    else:
        lines.append(rtl_markdown_block("אין קטעים מסומנים."))
    return "\n".join(lines).rstrip() + "\n"


def render_srt(turns: list[dict], names: list[str]) -> str:
    cues = []
    for index, turn in enumerate(turns, start=1):
        cues.append(
            f"{index}\n{format_timestamp(turn['start'], ',')} --> {format_timestamp(turn['end'], ',')}\n"
            f"{speaker_label(turn, names)}: {turn['final']}\n"
        )
    return "\n".join(cues)


# ---------------------------------------------------------------------------
# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one long Hebrew recording with speaker attribution: "
            "faster-whisper (several models) + pyannote + DICTA cleanup + Qwen review."
        )
    )
    parser.add_argument("audio", type=Path, help="Audio file to transcribe.")
    parser.add_argument("--glossary", type=Path, default=None, help="glossary.json from the Hebrew questionnaire.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR, help="Where to write .md/.srt/.json.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Checkpoint directory. Default: <project>/work/<audio stem>",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and recompute everything.")
    parser.add_argument(
        "--stop-after",
        choices=["prepare", "asr", "diarize", "align", "cleanup", "review"],
        default=None,
        help="Stop after this stage (checkpoints are kept, so a later run continues).",
    )
    parser.add_argument("--no-loudnorm", action="store_true", help="Skip loudness normalization.")

    parser.add_argument(
        "--asr-models",
        default=",".join(DEFAULT_ASR_MODELS),
        help="Comma-separated faster-whisper models. The first is primary. Default: %(default)s",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type (auto = float32 on CPU, float16 on GPU; int8 is ~3x faster on CPU).",
    )
    parser.add_argument("--beam-size", type=int, default=5)

    parser.add_argument("--num-speakers", type=int, default=2, help="Exact speaker count. 0 = estimate. Default: 2")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument(
        "--diarization-model",
        default=None,
        help=(
            f"pyannote pipeline. Default: {GATED_DIARIZATION_MODEL} when a Hugging Face token is available,"
            f" else {OPEN_DIARIZATION_MODEL}."
        ),
    )
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Default: HF_TOKEN or cached login.")
    parser.add_argument(
        "--speaker-names",
        default="",
        help="Comma-separated names in order of first appearance, e.g. 'רחל,דוד'. Re-render is instant.",
    )

    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--cleanup-model", default=None)
    parser.add_argument("--review-model", default=None)
    parser.add_argument("--review-think", action="store_true", help="Let the review model think (much slower).")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-timeout-seconds", type=int, default=1800)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.audio.expanduser().resolve()
    if not source.is_file():
        print(f"Audio file not found: {source}", file=sys.stderr)
        return 1

    work_dir = (args.work_dir or PROJECT_DIR / "work" / source.stem).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for path in work_dir.iterdir():
            if path.suffix in {".json", ".jsonl", ".wav"}:
                path.unlink()

    glossary = load_glossary(args.glossary.expanduser().resolve() if args.glossary else None)
    names = [name.strip() for name in args.speaker_names.split(",")] if args.speaker_names else []
    asr_models = [model.strip() for model in args.asr_models.split(",") if model.strip()]
    run_started = time.monotonic()

    # prepare
    wav_path = work_dir / "audio.wav"
    prepare_params = {"source": str(source), "mtime": source.stat().st_mtime, "loudnorm": not args.no_loudnorm}
    prepared = load_checkpoint(work_dir / "prepare.json", prepare_params)
    if prepared is None or not wav_path.exists():
        log(f"Preparing audio from {source} ...")
        duration = prepare_audio(source, wav_path, loudnorm=not args.no_loudnorm)
        prepared = save_checkpoint(work_dir / "prepare.json", prepare_params, {"duration": duration})
    duration = prepared["data"]["duration"]
    log(f"Audio duration: {duration / 60:.1f} min")
    if args.stop_after == "prepare":
        return 0

    # asr
    initial_prompt, hotwords = whisper_prompts(glossary)
    asr_results: list[list[dict]] = []
    for model_id in asr_models:
        params = {
            "audio": fingerprint(prepare_params),
            "model": model_id,
            "compute_type": args.compute_type,
            "beam_size": args.beam_size,
            "initial_prompt": initial_prompt,
            "hotwords": hotwords,
        }
        path = work_dir / f"asr-{model_id.replace('/', '__')}.json"
        checkpoint = load_checkpoint(path, params)
        if checkpoint is None:
            stage_started = time.monotonic()
            segments = run_asr(
                wav_path,
                model_id,
                duration=duration,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            )
            checkpoint = save_checkpoint(path, params, segments)
            log(f"ASR {model_id} done in {(time.monotonic() - stage_started) / 60:.1f} min")
        asr_results.append(checkpoint["data"])
    if args.stop_after == "asr":
        return 0

    # diarize
    token = resolve_hf_token(args.hf_token)
    diarization_model = resolve_diarization_model(args.diarization_model, token)
    num_speakers = args.num_speakers or None
    diarize_params = {
        "audio": fingerprint(prepare_params),
        "model": diarization_model,
        "num_speakers": num_speakers,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers,
    }
    diarization_checkpoint = load_checkpoint(work_dir / "diarization.json", diarize_params)
    if diarization_checkpoint is None:
        stage_started = time.monotonic()
        diarization = run_diarization(
            wav_path,
            diarization_model,
            token=token,
            num_speakers=num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        diarization_checkpoint = save_checkpoint(work_dir / "diarization.json", diarize_params, diarization)
        log(f"Diarization done in {(time.monotonic() - stage_started) / 60:.1f} min")
    diarization = diarization_checkpoint["data"]
    if args.stop_after == "diarize":
        return 0

    # align
    primary_words = asr_words(asr_results[0])
    speakers = smooth_speakers(primary_words, assign_speakers(primary_words, diarization))
    turns = build_turns(primary_words, speakers)
    for extra in asr_results[1:2]:
        attach_alternative(turns, asr_words(extra), "alt")
    # Downstream checkpoints key on the turns themselves, so any change in
    # alignment logic or inputs invalidates the LLM results.
    align_fingerprint = fingerprint(
        [[turn["id"], turn["speaker"], turn["raw"], turn.get("alt")] for turn in turns]
    )
    save_checkpoint(work_dir / "turns.json", {"upstream": align_fingerprint}, turns)
    log(f"Aligned {len(primary_words)} words into {len(turns)} speaker turns")
    if args.stop_after == "align":
        return 0

    # cleanup
    glossary_text = glossary_for_llm(glossary)
    for turn in turns:
        turn["cleaned"] = turn["raw"]
        turn["cleanup_status"] = "skipped"
        turn["final"] = turn["raw"]
        turn["review_status"] = "skipped"
    cleanup_model_name = ""
    review_model_name = ""

    if not args.skip_cleanup:
        cleanup_model_name = resolve_cleanup_model(args.cleanup_model, args.ollama_url, 30)
        cleanup_params = {
            "upstream": align_fingerprint,
            "model": cleanup_model_name,
            "glossary": fingerprint(glossary_text),
            "names": names,
        }
        cleanup = run_llm_stage(
            "cleanup",
            turns,
            source_key="raw",
            block_chars=CLEANUP_BLOCK_CHARS,
            build_messages=lambda block: build_cleanup_messages(block, names, glossary_text),
            model_name=cleanup_model_name,
            ollama_url=args.ollama_url,
            timeout_seconds=args.llm_timeout_seconds,
            num_ctx=CLEANUP_NUM_CTX,
            think=False,
            progress_path=work_dir / "cleanup.jsonl",
            progress_params=cleanup_params,
        )
        for turn in turns:
            turn["cleaned"] = cleanup[turn["id"]]["text"]
            turn["cleanup_status"] = cleanup[turn["id"]]["status"]
            turn["final"] = turn["cleaned"]
        if args.stop_after == "cleanup":
            return 0

        if not args.skip_review:
            review_model_name = resolve_review_model(args.review_model, args.ollama_url, 30)
            review_params = {**cleanup_params, "review_model": review_model_name, "think": args.review_think}
            review = run_llm_stage(
                "review",
                turns,
                source_key="cleaned",
                block_chars=REVIEW_BLOCK_CHARS,
                build_messages=lambda block: build_review_messages(block, names, glossary_text),
                model_name=review_model_name,
                ollama_url=args.ollama_url,
                timeout_seconds=args.llm_timeout_seconds,
                num_ctx=None,
                think=args.review_think,
                progress_path=work_dir / "review.jsonl",
                progress_params=review_params,
            )
            for turn in turns:
                turn["final"] = review[turn["id"]]["text"]
                turn["review_status"] = review[turn["id"]]["status"]

    # render
    for turn in turns:
        for key in ("raw", "alt", "cleaned", "final"):
            if isinstance(turn.get(key), str):
                turn[key] = strip_bidi(turn[key])
        turn["words"] = [strip_bidi(word) for word in turn["words"]]
        turn["flags"] = compute_flags(turn)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem + ".speakers"
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    metadata = {
        "מקור": str(source),
        "משך": format_timestamp(duration),
        "מודלי זיהוי דיבור": ", ".join(asr_models),
        "זיהוי דוברים": diarization_model,
        "מספר דוברים": str(len({turn["speaker"] for turn in turns})),
        "ניקוי ראשון": cleanup_model_name or "דולג",
        "ביקורת": review_model_name or "דולג",
        "מילון מונחים": str(args.glossary) if args.glossary else "ללא",
        "נוצר": generated_at,
    }
    (output_dir / f"{stem}.md").write_text(render_markdown(source, turns, names, metadata), encoding="utf-8")
    (output_dir / f"{stem}.srt").write_text(render_srt(turns, names), encoding="utf-8")
    (output_dir / f"{stem}.json").write_text(
        json.dumps({"metadata": metadata, "diarization": diarization, "turns": turns}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    flagged = sum(1 for turn in turns if turn["flags"])
    log(
        f"Done in {(time.monotonic() - run_started) / 60:.1f} min. {len(turns)} turns, {flagged} flagged. "
        f"Wrote {output_dir / stem}.{{md,srt,json}}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
