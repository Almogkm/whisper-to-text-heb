# Local tracker workspace

This directory is reserved for local Markdown issues, PRDs, and agent-generated work items.

## Layout

```text
.scratch/
└── <feature-slug>/
    ├── PRD.md
    └── issues/
        ├── 01-some-slice.md
        └── 02-next-slice.md
```

## Issue header convention

Start each issue with a small header block such as:

```md
# Improve review-stage fallback

Status: needs-triage
Owner: unassigned
```

The `Status:` value should match one of the mappings in `docs/agents/triage-labels.md`.
