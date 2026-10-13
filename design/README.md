# Design handoffs

Approved designs for the Open Text Detector site, committed so every session works from the same version.

Designs are explored on the design canvas (claude.ai artifact `MVuxhqZ2gDVpihzaEMAg19`, owned by Alice). The canvas is a sketchbook: once a board is approved, it is exported here and stops being the source of truth. The code on `main` is the source of truth for what is live.

## How a handoff works

1. Alice approves a board on the canvas.
2. The board is exported to `design/<page>/<date>-<page>.dc.html` (open it in a browser to see it) with a spec next to it, `<date>-<page>.md`, saying exactly what changes in the code and what does not.
3. A coding session implements the spec on its own branch, from `main`, and shows Alice screenshots.
4. The branch is merged to `main` and deployed per the deploy rule in `CLAUDE.md`.
5. The spec's status line is updated (approved → built → live, with commits).

## Rules for implementing a handoff

- The spec is the instruction; the `.dc.html` board shows the intended result. If they disagree, the spec wins; ask Alice.
- Copy text verbatim from the spec. Do not reword, tidy or "improve" copy. Alice edits copy herself, and the project avoids wording that steers how readers interpret scores.
- Change only what the spec lists. The board's header, footer and fonts are approximations of the real site chrome; do not restyle the site to match them.
- Small copy tweaks after a page ships are made in the code directly. New layouts go back through the canvas.

## Handoffs

| Date | Page | Spec | Status |
|---|---|---|---|
| 2026-10-13 | Landing (home) | [landing/2026-10-13-landing.md](landing/2026-10-13-landing.md) | Live (main cee0927) |
