# Domain Docs

How engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repo root
- `docs/adr/` for architectural decisions that affect the current area

If a file is missing in a future clone or branch, proceed silently rather than stopping work.

## File structure

This repo is currently single-context:

```text
/
├── CONTEXT.md
├── docs/
│   ├── adr/
│   ├── agents/
│   └── takeoff/
└── src/
```

## Use the glossary's vocabulary

When naming concepts in issues, plans, bugs, refactors, or new docs, use the terms defined in `CONTEXT.md`.

## Flag ADR conflicts

If a proposed change conflicts with an ADR, surface that conflict explicitly instead of silently overriding it.
