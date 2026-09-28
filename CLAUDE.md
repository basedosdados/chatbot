# CLAUDE.md — Data Basis Chatbot

## Git Flow (branch & PR workflow)

**Read this before creating branches, committing, or opening PRs. This repo follows Git Flow.**
Canonical reference: https://github.com/basedosdados/backend/wiki/Boas-Pr%C3%A1ticas#segue-o-fluxo

### The environment branches — never commit or push directly
- `main` — production. Source of truth: resets flow *from* `main`.
- `staging` — pre-production / QA.

This repo has no `dev` branch. If the team later adds one to match `backend`, insert it here
as a third target and open three PRs instead of two.

### How work flows
Every feature starts from `main` and is promoted back into `staging` using **one branch and
two PRs** — the same head branch is PR'd into `staging` and into `main`. Merging the same
branch into each target puts the *same commit objects* into both, so the feature's own
commits share SHAs. Only the merge commits differ, which is expected and fine.

The environments still drift apart as those merge commits accumulate at different times, so
the team **resets `staging` back to `main` roughly every two weeks**. Because resets flow
*from* `main`, a change must reach `main` to survive — anything living only on `staging` is
discarded at the next reset.

### Feature workflow — one branch, two PRs
1. Cut your feature branch off `main` — never off `staging`.
   Name it by intent: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `refactor/…`.
   One logical change per branch.
2. From that **same branch**, open two PRs: one into `staging`, one into `main`.
   Do not cut a separate branch per target, and do not cherry-pick.
3. Merge with a **merge commit or fast-forward — never squash**. A squash mints a new,
   unrelated commit on each branch and breaks the shared history the resets rely on.
4. Timing: a `main`-based branch merges cleanly into `staging` when `staging` is aligned
   with `main` — in practice, shortly after a reset. The longer since the last reset, the
   more of `main`'s accumulated commits the PR will drag along. If `staging` has drifted
   far, wait for the reset rather than forcing a noisy merge.

### Rules for agents working in this repo
- Never commit or push to `main` or `staging` directly — always a feature branch + PR.
- Always cut features off `main`, never off `staging`.
- Use **one branch for all PRs**. Never a branch-per-target, never cherry-pick.
- **Never squash-merge.** Merge commit or fast-forward only.
- Never merge `staging` into `main` (or vice versa) to promote a feature.
- Before committing, verify you are on a feature branch: `git branch --show-current`.
- Do not push without explicit permission.
