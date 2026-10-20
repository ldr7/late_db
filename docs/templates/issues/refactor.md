## Structural problem

Describe the current coupling, duplication, or unclear boundary and the concrete
development or maintenance work it makes harder.

## Proposed structure

Describe the responsibility or interface that should move, and the intended
module boundaries. Reference the affected repository paths.

## Behavior to preserve

Identify observable contracts such as scores, ranking ties, validation errors,
Python interfaces, and storage invariants that must remain consistent.

## Acceptance criteria

Replace these prompts with the structural result and preservation evidence:

- [ ] The intended responsibility has a clear owner and interface.
- [ ] Relevant checks demonstrate preserved caller behavior and invariants.
- [ ] Architecture documentation reflects changed boundaries where applicable.

## Evidence, scope, and limitations

Reference code or existing checks that establish the current behavior. Identify
dependencies, migration needs, and any performance effects that require measurement.
