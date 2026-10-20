# Contributing to late_db

Changes should make the Rust retrieval engine and its demo easier to run,
verify, or maintain. Start with a concrete problem and an observable outcome.

## Issues and Branches

Use the bug report, feature/demo enhancement, or benchmark/experiment issue
template to capture the problem, acceptance criteria, and available evidence.
Record the issue number before starting an implementation.

Treat `main` as protected: work on a feature branch and submit a pull request
against `main`. A merge request (MR) means the same thing as a pull request (PR)
in project discussions. Use a descriptive branch name such as
`feat/12-interactive-demo`, `fix/13-query-validation`,
`feat/14-cached-replay`, or `docs/15-setup-guide`.

Keep each PR focused on one reviewable change. Link the issue with
`Closes #12` when merging will complete it, or `Refs #12` for partial work.
Publish branches and open PRs only within the authorization given for the task;
merging and changing repository protection settings require authorization.

## Implementation Loop

1. Define the expected behavior and the evidence needed to accept it.
2. Reproduce the current behavior or establish a measured baseline.
3. Implement the smallest complete change. Use subagents for independent work
   when useful, with explicit file ownership and shared acceptance criteria.
4. Run targeted checks, review the result, and repeat when evidence exposes a
   failure. Finish the relevant project checks before requesting review.

Keep Rust storage, scoring, and orchestration boundaries explicit. Update
`docs/architecture.md` for changed boundaries and `docs/results_analysis.md` for
new benchmark conclusions. Put learning-only work in ignored `learning/`,
`docs/learning/`, or `rustlings/` directories.

Use repository data and context. Web browsing is limited to development best
practices and execution-platform integration. Never inspect or display
environment-variable values, `.env` contents, tokens, or secret-bearing files.
Authorized tools may consume credentials internally without exposing them.
Include only sanitized command output in issues and PRs.

## Validation Evidence

For engine or Python integration changes, run the applicable checks from the
repository root after activating the project environment:

```sh
source ./env.sh
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
maturin develop --release
python -m unittest discover -s tests -p 'test_*.py' -v
python -m late_db.demo
```

For documentation-only changes, verify commands, links, and examples relevant
to the edit. Describe what was checked; say which checks were not run and why.

Benchmark reports should include the corpus and query selection, model/cache
identity, device and runtime versions, exact command, quality metrics, timing,
and limitations. Keep generated artifacts under
`data/universal_retrieval/runs`, which is ignored by Git. Report synthetic and
real-model evidence separately. Do not claim model quality from the synthetic
demo or compare latency across mismatched workloads.

## Pull Request Templates

The default PR template covers the problem, resulting behavior, evidence, and
remaining limitations. Choose a more specific template by adding the matching
query parameter to the GitHub compare/new-PR URL:

| Change | Template parameter |
| --- | --- |
| Feature or demo enhancement | `?template=feature.md` |
| Bug fix | `?template=fix.md` |
| Documentation or maintenance | `?template=docs.md` |

If the URL already has a query string, append `&template=feature.md` instead.
For example, start from
`https://github.com/ldr7/late_db/compare/main...YOUR_BRANCH?expand=1&template=feature.md`.
Replace `YOUR_BRANCH` with the actual branch name. The Markdown templates live
in `.github/PULL_REQUEST_TEMPLATE/` and can also be used as the body of a CLI
created PR.

Write the description for a reviewer who has not seen the working conversation.
Lead with what changes for a caller or demo user, then provide validation and
known limits. Review feedback should lead to another implementation and
validation pass on the same branch.
