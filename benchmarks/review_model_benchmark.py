from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_DIR / "src"
sys.path.insert(0, str(PROJECT_SRC))

from whisper_md_transcriber.cli import StageResult, review_transcript


@dataclass(frozen=True)
class ReviewCase:
    id: str
    language: str
    raw: str
    cleaned: str
    gold: str
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark second-pass review models against judged gold transcripts."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_DIR / "benchmarks" / "review_cases.json",
        help="Path to benchmark cases JSON. Default: benchmarks/review_cases.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write benchmark outputs and summary into.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama API base URL.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Timeout per review request.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3500,
        help="Maximum characters per review chunk.",
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more Ollama review model names to compare.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[ReviewCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ReviewCase(**item) for item in payload]


def normalized_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("–", " ").replace("-", " ").replace("־", " ")
    return re.findall(r"\w+", normalized.casefold(), flags=re.UNICODE)


def levenshtein_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            substitution_cost = 0 if left_token == right_token else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def correction_accuracy(required_edits: int, remaining_edits: int) -> float:
    if required_edits <= 0:
        return 1.0 if remaining_edits == 0 else 0.0
    score = 1.0 - (remaining_edits / required_edits)
    return max(0.0, score)


def run_model(
    model_name: str,
    cases: list[ReviewCase],
    *,
    ollama_url: str,
    timeout_seconds: int,
    max_chars: int,
    output_dir: Path,
) -> dict:
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    total_required_edits = 0
    total_remaining_edits = 0
    total_gold_words = 0
    total_word_errors = 0
    case_results: list[dict] = []

    for case in cases:
        first_pass = StageResult(
            text=case.cleaned,
            model_name="benchmark-first-pass",
            chunk_count=1,
            status="applied",
            chunks=(case.cleaned,),
        )
        result = review_transcript(
            raw_transcript=case.raw,
            first_pass_result=first_pass,
            language=case.language,
            requested_model=model_name,
            ollama_url=ollama_url,
            timeout_seconds=timeout_seconds,
            max_chars=max_chars,
        )

        predicted_words = normalized_words(result.text)
        cleaned_words = normalized_words(case.cleaned)
        gold_words = normalized_words(case.gold)

        required_edits = levenshtein_distance(cleaned_words, gold_words)
        remaining_edits = levenshtein_distance(predicted_words, gold_words)
        word_errors = remaining_edits

        total_required_edits += required_edits
        total_remaining_edits += remaining_edits
        total_gold_words += len(gold_words)
        total_word_errors += word_errors

        (model_dir / f"{case.id}.txt").write_text(result.text + "\n", encoding="utf-8")
        case_results.append(
            {
                "id": case.id,
                "required_edits": required_edits,
                "remaining_edits": remaining_edits,
                "correction_accuracy": correction_accuracy(required_edits, remaining_edits),
                "word_accuracy_to_gold": 1.0 - (word_errors / max(len(gold_words), 1)),
                "notes": case.notes,
                "prediction_path": str(model_dir / f"{case.id}.txt"),
            }
        )

    summary = {
        "model": model_name,
        "cases": case_results,
        "aggregate": {
            "required_edits": total_required_edits,
            "remaining_edits": total_remaining_edits,
            "correction_accuracy": correction_accuracy(total_required_edits, total_remaining_edits),
            "word_accuracy_to_gold": 1.0 - (total_word_errors / max(total_gold_words, 1)),
            "gold_word_count": total_gold_words,
        },
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases.expanduser().resolve())

    summaries = [
        run_model(
            model_name=model_name,
            cases=cases,
            ollama_url=args.ollama_url,
            timeout_seconds=args.timeout_seconds,
            max_chars=args.max_chars,
            output_dir=output_dir,
        )
        for model_name in args.models
    ]

    if len(summaries) >= 2:
        baseline_score = summaries[0]["aggregate"]["correction_accuracy"]
        for summary in summaries[1:]:
            candidate_score = summary["aggregate"]["correction_accuracy"]
            if baseline_score > 0:
                relative_gain = (candidate_score - baseline_score) / baseline_score
            else:
                relative_gain = float("inf") if candidate_score > 0 else 0.0
            summary["aggregate"]["relative_gain_vs_first_model"] = relative_gain

    combined = {"cases": [asdict(case) for case in cases], "summaries": summaries}
    (output_dir / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    print("model\tcorrection_accuracy\tword_accuracy_to_gold\trequired_edits\tremaining_edits")
    for summary in summaries:
        aggregate = summary["aggregate"]
        print(
            f"{summary['model']}\t"
            f"{aggregate['correction_accuracy']:.4f}\t"
            f"{aggregate['word_accuracy_to_gold']:.4f}\t"
            f"{aggregate['required_edits']}\t"
            f"{aggregate['remaining_edits']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
