# Reusable Issue and PR Bodies

These raw Markdown issue bodies and the existing PR Markdown templates are the
canonical starting points for API-created and CLI-created work items. The YAML
issue forms serve the GitHub UI; their headings match the raw issue bodies.

## Six Template Pairs

| Type | Raw issue body | GitHub UI form | Raw PR body |
| --- | --- | --- | --- |
| Feature or demo enhancement | [feature.md](issues/feature.md) | [feature.yml](../../.github/ISSUE_TEMPLATE/feature.yml) | [feature.md](../../.github/PULL_REQUEST_TEMPLATE/feature.md) |
| Bug fix | [bug.md](issues/bug.md) | [bug.yml](../../.github/ISSUE_TEMPLATE/bug.yml) | [fix.md](../../.github/PULL_REQUEST_TEMPLATE/fix.md) |
| Benchmark or experiment | [benchmark.md](issues/benchmark.md) | [experiment.yml](../../.github/ISSUE_TEMPLATE/experiment.yml) | [benchmark.md](../../.github/PULL_REQUEST_TEMPLATE/benchmark.md) |
| Refactor | [refactor.md](issues/refactor.md) | [refactor.yml](../../.github/ISSUE_TEMPLATE/refactor.yml) | [refactor.md](../../.github/PULL_REQUEST_TEMPLATE/refactor.md) |
| Documentation | [docs.md](issues/docs.md) | [docs.yml](../../.github/ISSUE_TEMPLATE/docs.yml) | [docs.md](../../.github/PULL_REQUEST_TEMPLATE/docs.md) |
| Maintenance | [maintenance.md](issues/maintenance.md) | [maintenance.yml](../../.github/ISSUE_TEMPLATE/maintenance.yml) | [maintenance.md](../../.github/PULL_REQUEST_TEMPLATE/maintenance.md) |

Each of the six types has a distinct issue body and PR body, for twelve reusable
Markdown bodies. The [default PR template](../../.github/pull_request_template.md)
remains a fallback for changes that do not fit a specific type.

The GitHub PR UI can select a specific body with `?template=feature.md`; replace
the filename with the PR filename in the table for another type. Use
`&template=...` when the URL already contains a query string.

## Agent First Pass

1. Select the matching issue or PR template and copy it to a temporary body file.
2. Write a complete first draft using repository evidence. Replace every prompt
   with concrete behavior, measurable acceptance criteria, and available results.
   Record unknowns and limits explicitly; never invent validation evidence.
3. For a PR, link its issue using `Closes #NUMBER` or `Refs #NUMBER`, and explain
   the final change for a reviewer who has not seen the working conversation.
4. Review the exact body, then supply it to the authorized API or CLI operation.
   With `gh`, use `--body-file` to preserve Markdown and actual newlines.

Agents decide the first draft rather than asking the user to fill a blank
outline. Remove unused prompts before submission. Keep generated benchmark
artifacts in ignored run directories and summarize their measured evidence in
the body. Never include credentials, environment-variable values, or secret
file contents.

Follow [CONTRIBUTING.md](../../CONTRIBUTING.md) for branches, validation, and the
protected-main workflow. Update matching raw issue bodies and YAML forms
together when their structure changes.
