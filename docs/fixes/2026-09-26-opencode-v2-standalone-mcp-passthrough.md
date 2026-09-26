# Fix: opencode v2 silently dropped MCP pass-through; `atmos ai` now detects the major version

**Date:** 2026-09-26

## Summary

opencode v2 changed the execution model behind `opencode run`: requests route through a
persistent background service that resolves config in its own process. That service ignores
`OPENCODE_CONFIG` from the invoking environment, so Atmos's MCP pass-through — a temp config
referenced via `OPENCODE_CONFIG` — was silently ignored on v2 installs: configured servers,
their env values, and `atmos auth exec` wrappers never reached the model, violating the
provider's own "never silently run without the servers" contract. Atmos now probes
`opencode --version` once per client construction and, on v2+, invokes `run` with
`--standalone` (in-process config resolution, verified to honor `OPENCODE_CONFIG`) and
`--format json` (structured answer extraction). v1 binaries keep the legacy invocation.

## Context

The opencode CLI provider (pkg/ai/agent/opencode, added in #3189) was written against v1,
where `opencode run` was a standalone process that read `OPENCODE_CONFIG` from its own
environment. Empirical checks against opencode v2.0.18:

- `opencode debug config` with `OPENCODE_CONFIG` set does not include the referenced file
  in the resolved config sources (service mode).
- Control experiment: a temp config disabling the `read` tool — in service mode the tool
  still ran (override ignored); with `--standalone` the tool errored (override honored).
- `--format json` now reliably emits the final text event (the v1-era dropped-final-event
  problem is gone), enabling JSONL answer extraction.
- v2 also natively discovers agent skills in `.opencode/skills/` (project) and
  `~/.config/opencode/skills/` (user), which `atmos ai skill install` did not target.

## Changes

- `pkg/ai/agent/opencode/client.go`: `NewClient` now takes a context, probes
  `<binary> --version` (5s timeout), and stores the major version, assuming v2 when the
  probe fails (a broken probe must never silently drop MCP servers). `buildArgs` adds
  `--standalone` and `--format json` for v2+. `ExtractResultJSON` parses the JSONL event
  stream — answer is the text of the final step (after the last `step_start`), `tool_use`
  noise and interim narration excluded, top-level `{"error":...}` objects surfaced as
  errors, non-JSON output falling back to plain-text extraction.
- `pkg/ai/agent/opencode/register.go`: factory threads the context through.
- `pkg/ai/skills/marketplace/clients.go`: opencode added as a skill-distribution client
  (`opencode` in `SupportedClients`; project leaf `.opencode/skills`, user leaf
  `~/.config/opencode/skills`, signal dirs `.opencode` / `~/.config/opencode`), which
  propagates to `atmos ai skill install/uninstall/update` flags, detection, and pickers.
- Website docs updated: providers page (v2 behavior, `#variant` model slugs), skill command
  page (opencode client lists and paths), MCP config page (standalone note).

## Validation

- `go build ./...`, `go vet ./pkg/ai/... ./cmd/ai/...` — clean.
- `go test ./pkg/ai/agent/opencode/ -count=1`,
  `go test ./pkg/ai/skills/... -count=1`, and
  `go test ./cmd/ai/... ./pkg/ai/... ./pkg/mcp/... -count=1` — all pass (41 packages).
  New tests cover version parsing (v2/v1/garbage), assume-v2 fallback, per-version
  `buildArgs`, JSONL extraction (multi-step, tool noise, error objects, fallbacks),
  end-to-end v2 `SendMessage` through the fake-binary harness, and opencode skill
  leaf/signal/detection paths at both scopes. The assume-v2 test caught a real bug during
  development (the fallback was logged but never applied) before it shipped.
- Patch-scoped lint (`./custom-gcl run --config=.golangci.yml --new-from-rev=origin/main`)
  — 0 issues. The custom binary was built manually per `.custom-gcl.yml` (clone v2.13.2,
  blank-import the lintroller plugin, replace to `./tools/lintroller`) because
  `golangci-lint custom`'s own internal `git clone` failed locally with exit 128 even
  though the identical clone command run by hand succeeds; worth re-checking `atmos lint
  --changed` on a machine where that clone works.
- Manual v2 verification (opencode v2.0.18): service vs standalone config-resolution
  asymmetry confirmed as described above; `run --format json` event shapes
  (`step_start`/`text`/`tool_use`/`step_finish`) confirmed against live output.
- End-to-end smoke with a locally built atmos against the real v2 binary:
  `atmos ai exec -p opencode` in this repo (which configures the `atmos` MCP server)
  printed "MCP servers configured: 1 (via OPENCODE_CONFIG)" and returned the exact
  requested reply — proving the pass-through fix and JSONL extraction together. In a
  scratch project, `atmos ai skill install atmos-terraform --client opencode` wrote
  `.opencode/skills/atmos-terraform/SKILL.md` and uninstall removed it.

## Follow-ups

None.
