# CLAUDE.md — Data Basis Chatbot

## Git Flow (branch & PR workflow)

**Read this before creating branches, committing, or opening PRs. This repo follows Git Flow.**
Canonical reference: https://github.com/basedosdados/backend/wiki/Boas-Pr%C3%A1ticas#segue-o-fluxo

### The environment branches — never commit or push directly
- `main` — production. Source of truth: resets flow *from* `main`.
- `staging` — pre-production / QA.

This repo has no `dev` branch; `staging` is the pre-production environment. If the team
later adds a `dev` branch to match `backend`, insert it here as the integration target.

### How work flows
Cut every feature off `main`, then merge the **same commits** into `staging` and `main` —
one PR per target. The two branches are meant to hold identical history for each feature.
They drift apart as changes land at different times, so the team periodically **resets
`staging` back to `main`** to re-align them. Because resets flow from `main`, a change must
reach `main` to survive — anything living only on `staging` is discarded at the next reset.

### Feature workflow — same commits into every branch
1. Cut your feature branch off `main` (ideally just after a reset, when the branches are aligned).
   Name it by intent: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `refactor/…`. One logical change per branch.
2. Open a PR from that branch into each target branch: `staging` and `main`.
3. Merge with a **merge commit or fast-forward — never squash, never cherry-pick** — so the
   identical commit SHA lands on both branches.
4. If `staging` has drifted far from `main`, realign it via the periodic reset rather than
   forcing a noisy cross-branch merge.

### Rules for agents working in this repo
- Never commit or push to `main` or `staging` directly — always a feature branch + PR.
- Cut features off `main` so the same commit can enter both environments.
- Merge as merge-commit/FF, **never squash** (squash mints a new SHA per branch) and
  **never cherry-pick** (also a new SHA) — both break the "same commits everywhere" rule.
- Never merge `staging` into `main` (or vice versa) to promote a feature.
- Before committing, verify you are on a feature branch: `git branch --show-current`.
- Do not push without explicit permission.
