# Issue tracker: Local Markdown

As of 2026-09-03, this repo has no configured remote and no external issue tracker. Issues, PRDs, and future agent work items should live as Markdown files under `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The PRD is `.scratch/<feature-slug>/PRD.md`
- Implementation issues are `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Triage state is recorded as a `Status:` line near the top of each issue file
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`, creating directories as needed.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. For this repo, the path is the tracker identifier.
