# hone

A CLI that points an LLM agent at a Go package and lets it work: a quality
pass by default, or your own goal — with a second, read-only agent
reviewing every "done" before hone believes it.

It's a thin preset over [ai-sdk](https://github.com/samcharles93/ai-sdk)'s
`agentloop`: the mission, context strategy, and quality gate are configured
here; the loop mechanics (budgets, gated tools, notes, structured results)
live in the shared library.

## Quick start

```sh
make install                # builds and copies to ~/.local/bin/hone

hone ./mypackage
```

With no flags, that runs the built-in quality pass once: read the package,
fix what's wrong, validate against `go vet` + `go fix -diff` (+ lint/tests
if enabled), call `finish`. Exit code tells you how it went.

## Two ways to run it

### 1. Quality pass (default)

One agent, one mission, one gate. Good for "clean this package up."

```sh
hone -gate-test ./mypackage
```

### 2. Custom goal, with a review gate

```sh
hone -goal "add table-driven tests for the parser" -max-runs 10 ./mypackage
```

`-goal` replaces the built-in mission. hone re-invokes the agent (each
call still bounded by `-max-steps`) until it calls `finish` or `-max-runs`
runs out, carrying context forward between invocations via
`AGENT_NOTES.md` in the package root.

Every time a run reports "passed," hone **doesn't just trust it**. It
spawns a second hone process — read-only, no write/edit/shell access
— to independently judge the diff and the worker's own summary against the
original goal:

- **Approved** → done, exit 0.
- **Rejected** → the reviewer's reasoning is written to `AGENT_NOTES.md`
  as a note, and the loop tries again.
- Runs out of `-max-runs` without an approval → exit 2 (parked).

Turn it off with `-verify=false` if you want the worker's self-report
trusted outright. Point the reviewer at a different (e.g. stronger, or
cheaper) model with `-review-model` / `-review-host`.

This is one child process, spawned and awaited synchronously — not
background parallelism, and it never touches shared library code.

## Models

`-model` is `provider/model`, e.g.:

```sh
hone -model openai/gpt-5.4 -gate-test ./mypackage
hone -model ollama/qwen3-coder -host http://remote-box:11434 ./mypackage
hone -model openai-compatible/my-local-model -host http://remote-box:8081 ./mypackage
```

The provider must name an ai-sdk built-in class: `openai`, `anthropic`,
`deepseek`, `groq`, `mistral`, `cohere`, `gemini`, `perplexity`, `xai`,
`azure`, `ollama`, `openai-compatible`, ...

API keys come from `<PROVIDER>_API_KEY` (e.g. `OPENAI_API_KEY`). `ollama`
and `openai-compatible` need none — the latter is any self-hosted server
speaking the OpenAI chat protocol (llama.cpp's `llama-server`, vLLM, ...),
reached entirely via `-host`.

## Output

`-format` controls how a run's result prints:

| Value | What you get |
|---|---|
| `auto` (default) | Markdown rundown rendered through `glow` if it's installed and stdout is a terminal; ANSI-styled plain text if it's a terminal without `glow`; raw markdown if redirected to a file/pipe. |
| `markdown` | The raw markdown, unrendered — for pasting into a PR or piping to your own renderer. |
| `json` | The raw `Result` struct — for scripting. |

## Flags

Run `hone -h` for the full, current list with defaults. The ones
worth knowing about up front:

| Flag | Purpose |
|---|---|
| `-model` | `provider/model` ref (default `deepseek/deepseek-chat`) |
| `-host` | Base URL override for `-model`'s provider |
| `-goal` | Custom mission; enables the multi-run loop + review gate |
| `-max-runs` | Cap on consecutive runs when `-goal` is set (default 20) |
| `-max-steps` | Cap on agent steps per run (default 200) |
| `-max-tokens` | Token budget per run (0 = unlimited) |
| `-verify` | Enable/disable the review gate (default on, `-goal` only) |
| `-review-model` / `-review-host` | Override the reviewer's model/endpoint |
| `-gate-test` | Include `go test ./... -count=1` in the quality gate |
| `-gate-lint` | Include `golangci-lint run` (on by default if installed) |
| `-context` | `preload` (default: all sources up front) or `ondemand` |
| `-format` | `auto` (default), `markdown`, or `json` |
| `-readonly` | Register only read/grep/find — no mutation (used internally by the reviewer; also useful for planner/analysis-only runs) |

## eval_config.json

Optional, package-root file for project-specific context:

```json
{
  "package_name": "mypackage",
  "description": "what this package is for",
  "source_files": ["explicit.go", "list.go"],
  "skip_dirs": ["testdata"],
  "package_structure": "freeform description for the prompt",
  "build_commands": "Build:  go build ./...\nTest:   go test ./...",
  "extra_rules": ["an additional rule the agent should follow"]
}
```

All fields optional. `source_files`, if set, replaces auto-discovery of
`.go` files for the preload list.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Passed (and, if `-goal` was set, review-approved), or idle (nothing to do) |
| 2 | Parked — gate failures hit the cap, budget exhausted, or `-max-runs` ran out without approval |
| 1 | Infrastructure error (bad flags, provider/model resolution failure, etc.) |

## Notes across runs

`AGENT_NOTES.md` in the package root is the agent's persistent memory
across invocations of the same package — quality-pass runs, `-goal` loop
iterations, and reviewer rejections all read and write it. Every note
should cite how it was verified.
