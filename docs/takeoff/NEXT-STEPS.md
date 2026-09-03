# Next Steps

## Highest-value follow-on work

### 1. Multi-file A/B harness

The current benchmark uses one reference OGG. The next meaningful quality step is to compare candidate pipelines across several Hebrew files with different speakers and noise conditions.

Useful comparisons:

- `medium` only
- `medium -> DICTA`
- `medium -> DICTA -> Qwen`
- `large -> DICTA -> Qwen`

### 2. User-provided glossary support

Hebrew names, surnames, document names, and family terms are still the hardest errors. A small project glossary passed into Whisper and the review stage would likely buy more accuracy than simply scaling model size.

### 3. Structured transcript package

Refactor the runtime to produce a structured in-memory transcript package before rendering Markdown. That would make it much easier to:

- emit JSON
- compare stage outputs programmatically
- build future UI layers

### 4. Stage abstraction

Cleanup and review are close enough in shape that they should likely become explicit sibling stages. That would simplify:

- retries
- benchmarking
- fallback models
- experimentation with additional layers

### 5. Better review fallback policy

The second pass currently prefers `qwen36-reader-16k`. A future improvement is to make the review stage smarter about fallback choice based on:

- transcript length
- available RAM
- installed models

## Known open quality issues

- kinship terms can still flip incorrectly
- rare proper nouns still need help
- some document-related words remain vulnerable to phonetic confusion
- the system has no timestamp output yet

## When to revisit architecture

Refactor `cli.py` only when one of these becomes true:

- stage experimentation slows down because too many concerns are coupled
- new output formats are needed
- more than one benchmark harness or automation entrypoint appears
