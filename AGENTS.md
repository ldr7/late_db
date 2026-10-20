# AGENTS.md

## Project

`late_db` is a Rust-first in-memory late-interaction retrieval engine with a
thin Python benchmark/package layer.

## Working Rules

- Keep Rust storage, scoring, and orchestration boundaries explicit.
- Prioritize a working, reproducible demo. Implement the required core logic;
  learning exercises are deferred and must not block delivery.
- Keep learning-only material in ignored `learning/`, `docs/learning/`, or
  `rustlings/` directories. Keep product documentation and tests publishable.
- Prefer small, typed interfaces and zero-copy slices over ad hoc data movement.
- Keep benchmark scripts reproducible and write artifacts under
  `data/universal_retrieval/runs`.
- Update `docs/architecture.md` and result docs when module boundaries or
  benchmark conclusions change.

## Development Loop

- Define observable acceptance criteria before implementation.
- Use subagents for independent implementation or review tasks, with explicit
  file ownership to avoid overlapping edits.
- Reproduce failures, implement the smallest complete fix, run targeted checks,
  and review the resulting changes. Repeat when evidence reveals a failure.
- Finish with formatting, Rust tests and Clippy, Python integration tests, and
  an actual demo run. Report measured results and remaining limitations.
- Use only repository data and context. Web browsing is limited to development
  best practices and integration with execution platforms such as Colab/Kaggle.
- Do not commit, push, publish, or provision paid compute without authorization.

## GitHub Workflow

- Treat `main` as protected. Make changes on descriptive `feat/`, `fix/`,
  `docs/`, or `chore/` branches and propose them through pull requests.
- The user uses PR and MR interchangeably; both mean a GitHub pull request.
- Track substantive work in issues with observable acceptance criteria. Link
  each pull request to its issue and use the matching issue and PR templates.
- Use the six canonical raw issue bodies under `docs/templates/issues/` and
  PR bodies under `.github/PULL_REQUEST_TEMPLATE/`; see
  `docs/templates/README.md`. Draft complete first passes from repository
  evidence without asking the user to fill the template.
- PR descriptions must explain the problem, resulting behavior, validation,
  and material limitations. Review changes and check CI before merging.
- Publishing is authorized for the initial version in `ldr7/late_db`. Open an
  implementation PR and leave merging to the user unless explicitly authorized.

## Standing Credential Rule

- Never read, display, quote, or otherwise expose environment-variable values,
  including values stored in `.env` files. Do not inspect secret-bearing files
  or run commands that dump environment values or authentication tokens.
- Authorized scripts and tools may consume credentials internally without
  returning their values to the agent. Disable shell tracing and redact errors
  and logs so credentials cannot appear in tool output.
- Apply this rule to every subagent and future task in this project.

## Useful Commands

```sh
cd late_db
source ./env.sh
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
maturin develop --release
python -m unittest discover -s tests -p 'test_*.py' -v
python -m late_db.demo
python scripts/benchmark/check_universal_retrieval_ready.py --manifest data/universal_retrieval/public_smoke_manifest.jsonl
```
