# Quality and Benchmarks

## Quality strategy

The project reached acceptable quality only after moving from a single-model transcription flow to a layered correction flow.

Current sequence:

1. Whisper creates the raw transcript
2. DICTA improves Hebrew readability and obvious recognition errors
3. Qwen reconciles DICTA against the raw Whisper text to recover context DICTA still missed

## Why the second layer matters

On the reference OGG used during development, the review layer fixed several meaningful DICTA failures, including:

- `את אימך` -> `ממך`
- `מסבכים` -> `מסמכים`
- `תעודת ידע` -> `תעודת לידה`
- `אפשר לאטטר` -> `אפשר לאתר`

The largest remaining error class is semantically plausible but contextually wrong words, especially names and kinship terms.

## Benchmark: medium vs large

Benchmarked on 2026-09-03 against the reference OGG at:

`/home/almogked/Downloads/WhatsApp Ptt 2026-08-28 at 10.35.38.ogg`

Audio duration:

- 48.4065 seconds

Results:

| Run | Time |
| --- | ---: |
| `medium` Whisper only | 22.668s |
| `medium` full pipeline | 95.421s |
| `large` Whisper only, first uncached run | 292.287s |
| `large` Whisper only, warm cache | 46.514s |
| `large` full pipeline, warm cache | 119.828s |

Interpretation:

- warm-cache `large` is about 2.05x slower than `medium` for the raw Whisper step
- end-to-end `large` is only about 1.26x slower because the Ollama stages already dominate a large part of runtime
- the first ever `large` run pays a heavy one-time download cost

## Quality outcome: medium vs large on this file

`large` was not a clear win on the reference file.

Observed `large` strengths:

- improved some wording around `אשתך`
- improved some punctuation and article usage
- kept document-related phrases closer to the intended text in some spots

Observed `large` weaknesses:

- introduced awkward phrasing such as `אני אצליח ממך` in raw output
- still required the same post-processing layers
- did not clearly outperform the final `medium` pipeline output

Current recommendation:

- keep `medium` as the practical default for most files
- use `large` only for difficult recordings where `medium` still misses too much

## Benchmark artifacts

The benchmark outputs are checked into:

`benchmarks/2026-09-03-medium-vs-large/`

That directory contains:

- one transcript package per benchmark run
- stdout/stderr captures
- `results.json` timing summary

## How to evaluate future changes

Use the same file set and compare:

- raw Whisper output quality
- DICTA-only output quality
- final reviewed output quality
- wall-clock runtime

For future benchmarking, use more than one Hebrew file. One WhatsApp note is enough to expose regressions, but not enough to prove general improvement.
