# CLAUDE.md — Data Basis Chatbot

## Git Flow (branch & PR workflow)

**Read this before creating branches, committing, or opening PRs. This repo follows Git Flow.**
Canonical reference: https://github.com/basedosdados/backend/wiki/Boas-Pr%C3%A1ticas#segue-o-fluxo

### The environment branches — never commit or push directly
- `main` — production.
- `staging` — pre-production / QA, and the first integration/test target.

This repo has no `dev` branch; `staging` is where new work is integrated first. The two
branches are **parallel, independently maintained** lines, not a chain — you never merge
`staging` into `main` to move a feature, since that would drag the whole environment
forward. Each feature is promoted **selectively**, via its own PR carrying only that feature.
If the team later adds a `dev` branch to match `backend`, update this section to insert
`dev` as the first target ahead of `staging`.

### Feature workflow — promote the feature, not the environment
1. Cut your feature branch off `staging` (the branch you integrate and test in first).
   Name it by intent: `feat/…`, `fix/…`, `chore/…`, `docs/…`, `refactor/…`.
   Keep one logical change per branch, with tidy commits — you will cherry-pick them.
2. Open a PR from that branch into `staging`.
3. To promote the same feature to `main`, cut a new branch off `main` and cherry-pick
   only this feature's commit(s) onto it, then open a PR into `main`.
4. Result: one clean PR per environment, each carrying only this feature.

### Rules for agents working in this repo
- Never commit or push to `main` or `staging` directly.
- Move a feature to production by cherry-picking it onto a branch cut off `main` —
  never by merging `staging → main`.
- Each promotion branch is cut off its own target, so the PR diff is only this feature.
- One logical change per branch; one PR at a time per target; keep commits clean for cherry-picking.
- Before committing, verify you are on a feature branch: `git branch --show-current`.
- Do not push without explicit permission.
