// eval_loop — generic Go package evaluator powered by an LLM agent.
//
// A thin preset over ai-sdk/agentloop: the quality-pass mission, the
// preload-all context strategy, and the vet + go-fix gate are configured
// here; the loop mechanics (gated tools, budgets, loop breaker, notes,
// structured result) live in agentloop.
//
// Usage:
//
//	go run . ./mypackage
//	go run . -model openai/gpt-5.4 -gate-test ./mypackage
//	go run . -model ollama/qwen3-coder -host http://remote-box:11434 ./mypackage
//	go run . -model openai-compatible/my-local-model -host http://remote-box:8081 ./mypackage
//	go run . -goal "add table-driven tests for the parser" -max-runs 10 ./mypackage
//
// -goal replaces the built-in quality-pass mission with a custom one and
// keeps re-invoking the agent (each call still bounded by -max-steps)
// until it calls finish or -max-runs is exhausted, carrying context
// forward across invocations via AGENT_NOTES.md. Without -goal, eval_loop
// runs the quality pass exactly once, as before.
//
// The model is a runtime ref ("provider/model"); the provider must name
// one of the ai-sdk built-in classes (openai, anthropic, deepseek, groq,
// mistral, cohere, gemini, perplexity, xai, azure, ollama,
// openai-compatible, ...). Keys are read from the conventional
// "<PROVIDER>_API_KEY" env var; ollama and openai-compatible need none.
// -host overrides that provider's base URL for the run, for a remote
// ollama server, a self-hosted OpenAI-protocol server (llama.cpp's
// llama-server, vLLM, ...) via the openai-compatible class, or a
// cloud-compatible endpoint (e.g. an Azure OpenAI gateway or self-hosted
// proxy).
//
// Optional: eval_config.json in the package root provides
// package-specific context (description, extra rules, source list).
//
// Exit codes: 0 = passed or idle (nothing to do), 2 = parked
// (gate/budget/blocked), 1 = infrastructure error.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/samcharles93/ai-sdk/agentloop"
	"github.com/samcharles93/ai-sdk/runtime"
)

var (
	version = "dev"
	commit  = "unknown"
)

// EvalConfig holds optional per-project configuration loaded from
// eval_config.json in the package root. All fields are optional.
type EvalConfig struct {
	PackageName      string   `json:"package_name"`
	Description      string   `json:"description"`
	SourceFiles      []string `json:"source_files"`      // explicit preload list; if empty, auto-discovered
	SkipDirs         []string `json:"skip_dirs"`         // dirs to exclude from discovery
	PackageStructure string   `json:"package_structure"` // freeform description for the prompt
	BuildCommands    string   `json:"build_commands"`    // freeform build/test commands shown to the model
	ExtraRules       []string `json:"extra_rules"`       // additional <rules> bullet points
}

// PromptData is passed to system_prompt.tmpl.
type PromptData struct {
	PackageName      string
	Description      string
	PackageStructure string
	BuildCommands    string
	ExtraRules       []string
}

//go:embed system_prompt.tmpl
var systemPromptTmpl string

func main() {
	os.Exit(run())
}

func usage() {
	fmt.Fprintf(os.Stderr, `eval_loop — generic Go package evaluator powered by an LLM agent.

Usage:
  eval_loop [flags] [package-path]

  package-path defaults to $PKG_PATH, or "." if unset.

Examples:
  eval_loop ./mypackage
  eval_loop -model openai/gpt-5.4 -gate-test ./mypackage
  eval_loop -model ollama/qwen3-coder -host http://remote-box:11434 ./mypackage
  eval_loop -model openai-compatible/my-local-model -host http://remote-box:8081 ./mypackage

Flags:
`)
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, `
The provider in -model must name an ai-sdk built-in class (openai,
anthropic, deepseek, groq, mistral, cohere, gemini, perplexity, xai,
azure, ollama, openai-compatible, ...). API keys are read from
"<PROVIDER>_API_KEY" (e.g. OPENAI_API_KEY, GROQ_API_KEY). Ollama and
openai-compatible (any self-hosted server speaking the OpenAI chat
protocol, e.g. llama.cpp's llama-server or vLLM) need no key — point
them at your server with -host.

eval_config.json in the package root optionally provides package-specific
context (description, extra rules, source list) — see EvalConfig.

Exit codes: 0 = passed or idle (nothing to do), 2 = parked (gate/budget/
blocked), 1 = infrastructure error.
`)
}

func run() int {
	flag.Usage = usage
	showVersion := flag.Bool("version", false, "print version and exit")
	model := flag.String("model", "deepseek/deepseek-chat", `model ref "provider/model", e.g. openai/gpt-5.4, anthropic/claude-sonnet-5, ollama/qwen3-coder`)
	host := flag.String("host", "", "override the base URL for -model's provider (e.g. a remote ollama server or a custom endpoint for a cloud-compatible model)")
	contextMode := flag.String("context", "preload", "context strategy: preload (all sources up front) or ondemand")
	gateTest := flag.Bool("gate-test", false, "include 'go test ./... -count=1' in the quality gate")
	gateLint := flag.Bool("gate-lint", lintAvailable(), "include 'golangci-lint run' in the quality gate (default: on when golangci-lint is installed)")
	maxSteps := flag.Int("max-steps", 200, "maximum agent steps")
	maxTokens := flag.Int("max-tokens", 0, "token budget (0 = unlimited)")
	goal := flag.String("goal", "", "custom mission, replacing the built-in quality-pass mission. "+
		"When set, eval_loop re-invokes the agent (each call still bounded by -max-steps) up to "+
		"-max-runs times in a row until it calls finish, carrying context forward via AGENT_NOTES.md")
	maxRuns := flag.Int("max-runs", 20, "max consecutive agent invocations when -goal is set (ignored for the default quality-pass mission)")
	format := flag.String("format", "auto", "result output format: auto (rendered via glow on a terminal, "+
		"ANSI-styled text as a fallback, plain markdown when not a terminal), markdown, or json")
	flag.Parse()

	if *showVersion {
		fmt.Printf("eval_loop version: %s  commit: %s\n", version, commit)
		return 0
	}

	pkgRoot := flag.Arg(0)
	if pkgRoot == "" {
		pkgRoot = os.Getenv("PKG_PATH")
	}
	if pkgRoot == "" {
		pkgRoot = "."
	}
	abs, err := filepath.Abs(pkgRoot)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot resolve package root %q: %v\n", pkgRoot, err)
		return 1
	}
	pkgRoot = abs
	if info, err := os.Stat(pkgRoot); err != nil || !info.IsDir() {
		fmt.Fprintf(os.Stderr, "package root %q does not exist or is not a directory\n", pkgRoot)
		return 1
	}

	cfg := loadConfig(pkgRoot)
	skip := normalizeSkipDirs(cfg.SkipDirs)
	if len(skip) == 0 {
		skip = []string{".git", "bin", "vendor"}
	}

	gate := []agentloop.GateCommand{
		{Name: "vet", Argv: []string{"go", "vet", "./..."}},
		// go fix -diff exits 0 even when it finds issues; an empty diff
		// is the pass condition, so wrap it in a test for empty output.
		{Name: "go-fix-clean", Argv: []string{"sh", "-c", `test -z "$(go fix -diff ./... 2>&1)"`}},
	}
	if *gateLint {
		gate = append(gate, agentloop.GateCommand{Name: "lint", Argv: []string{"golangci-lint", "run"}})
	}
	if *gateTest {
		gate = append(gate, agentloop.GateCommand{Name: "test", Argv: []string{"go", "test", "./...", "-count=1"}})
	}

	mission := "Perform the iterative code-quality pass described in your instructions. " +
		"Work through the priorities in order, validate every change against the gate, " +
		"and call finish with a summary of what you improved and what you verified."
	if *goal != "" {
		mission = *goal
	}

	providerID, _, hasSlash := strings.Cut(*model, "/")
	providerID = strings.TrimSpace(providerID)
	if !hasSlash || providerID == "" {
		fmt.Fprintf(os.Stderr, "eval_loop: -model %q must be of the form \"provider/model\"\n", *model)
		return 1
	}
	pcfg := providerConfigFor(providerID)
	if *host != "" {
		pcfg.BaseURL = *host
	}

	runtime.RegisterBuiltinClasses()
	rt := runtime.NewRuntime(runtime.Config{
		Providers: map[string]runtime.ProviderConfig{providerID: pcfg},
	})

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
	logger.Info("starting eval run", "model", *model, "host", *host, "pkg_root", pkgRoot,
		"max_steps", *maxSteps, "gate_test", *gateTest, "gate_lint", *gateLint, "goal", *goal != "")

	runs := 1
	if *goal != "" {
		runs = *maxRuns
	}

	var res agentloop.Result
	for i := 1; i <= runs; i++ {
		if *goal != "" {
			logger.Info("goal run starting", "run", i, "of", runs)
		}

		// Re-rendered every run: a prior run in the same goal loop may
		// have added, renamed, or removed files.
		system, err := renderSystemPrompt(pkgRoot, cfg, skip)
		if err != nil {
			fmt.Fprintf(os.Stderr, "render system prompt: %v\n", err)
			return 1
		}
		var preload []string
		if *contextMode == "preload" {
			preload = sourceFilesFor(pkgRoot, cfg, skip)
		}

		stopHeartbeat := startHeartbeat(logger)
		res, err = agentloop.Run(context.Background(), agentloop.Config{
			Runtime:  rt,
			ModelRef: *model,
			WorkDir:  pkgRoot,
			// The template is fully rendered here; agentloop sees no actions.
			SystemTmpl:   system,
			Mission:      mission,
			PreloadFiles: preload,
			Notes:        agentloop.FileNotesStore{Path: filepath.Join(pkgRoot, "AGENT_NOTES.md")},
			Gate:         agentloop.GateConfig{Commands: gate, MaxConsecutiveFailures: 5},
			Preflight: []agentloop.GateCommand{
				{Name: "go-version", Argv: []string{"go", "version"}},
				{Name: "go-fix-diff", Argv: []string{"go", "fix", "-diff", "./..."}},
				{Name: "go-doc", Argv: []string{"go", "doc", "-all", "."}},
			},
			Budget: agentloop.Budget{MaxSteps: *maxSteps, MaxTokens: *maxTokens},
			Logger: logger,
		})
		stopHeartbeat()
		if err != nil {
			fmt.Fprintf(os.Stderr, "eval_loop: %v\n", err)
			return 1
		}

		printResult(res, i, runs, *format)

		if res.Status == agentloop.StatusPassed || res.Status == agentloop.StatusIdle {
			break
		}
		if i < runs {
			logger.Warn("goal not finished, starting another run", "status", res.Status, "stop_reason", res.StopReason)
		}
	}

	switch res.Status {
	case agentloop.StatusPassed, agentloop.StatusIdle:
		return 0
	default:
		return 2
	}
}

// startHeartbeat logs progress every 15s until the returned func is
// called. agentloop.Run makes no progress a caller can observe until it
// either calls a tool (logged at debug via HeadlessBridge) or returns, so
// a run stuck deep in a single slow model generation looks identical to
// a hung one; the heartbeat is the only signal available from outside
// the loop.
func startHeartbeat(logger *slog.Logger) func() {
	ctx, cancel := context.WithCancel(context.Background())
	start := time.Now()
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				logger.Info("still running", "elapsed", time.Since(start).Round(time.Second))
			}
		}
	}()
	return cancel
}

// printResult renders one run's result to stdout per format:
//
//   - "json": raw Result, for scripting.
//   - "markdown": the raw markdown rundown, unrendered — for piping to a
//     file, a PR description, or a tool that renders markdown itself.
//   - "auto" (default): the markdown rundown piped through glow when
//     it's on PATH and stdout is a terminal; ANSI-styled plain text when
//     stdout is a terminal but glow isn't installed; otherwise the same
//     raw markdown as the "markdown" format (redirected output should
//     stay plain and portable).
func printResult(res agentloop.Result, run, of int, format string) {
	switch format {
	case "json":
		out, _ := json.MarshalIndent(res, "", "  ")
		fmt.Println(string(out))
		return
	case "markdown":
		fmt.Println(renderMarkdown(res, run, of))
		return
	}

	tty := isTerminal(os.Stdout)
	switch {
	case tty && glowAvailable() && renderWithGlow(renderMarkdown(res, run, of)) == nil:
		return
	case tty:
		fmt.Println(renderPlainTerm(res, run, of))
	default:
		fmt.Println(renderMarkdown(res, run, of))
	}
}

// renderMarkdown formats a run's result as a short markdown rundown: a
// heading carrying the status, a fact list, and the model's own summary
// or (for a parked run) the gate/budget explanation. run/of are only
// shown when part of a -goal loop of more than one possible run.
func renderMarkdown(res agentloop.Result, run, of int) string {
	var b strings.Builder

	heading := strings.ToUpper(string(res.Status)[:1]) + string(res.Status)[1:]
	fmt.Fprintf(&b, "## eval_loop — %s\n\n", heading)

	if of > 1 {
		fmt.Fprintf(&b, "- **Run:** %d of %d\n", run, of)
	}
	fmt.Fprintf(&b, "- **Stop reason:** %s\n", res.StopReason)
	fmt.Fprintf(&b, "- **Iterations:** %d\n", res.Iterations)
	fmt.Fprintf(&b, "- **Tokens used:** %s\n", commas(res.TokensUsed))
	if len(res.Changes) > 0 {
		fmt.Fprintf(&b, "- **Files changed:** %s\n", strings.Join(res.Changes, ", "))
	} else {
		b.WriteString("- **Files changed:** none\n")
	}

	if res.Summary != "" {
		fmt.Fprintf(&b, "\n### Summary\n%s\n", res.Summary)
	}
	if res.Detail != "" {
		fmt.Fprintf(&b, "\n### Detail\n```\n%s\n```\n", res.Detail)
	}
	return b.String()
}

// ANSI styling for renderPlainTerm. No dependency, so it's always
// available as the fallback when glow isn't installed.
const (
	ansiReset  = "\033[0m"
	ansiBold   = "\033[1m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
	ansiCyan   = "\033[36m"
)

func statusANSI(status agentloop.Status) string {
	switch status {
	case agentloop.StatusPassed:
		return ansiGreen
	case agentloop.StatusIdle:
		return ansiCyan
	default:
		return ansiYellow
	}
}

// renderPlainTerm is the same rundown as renderMarkdown, styled with raw
// ANSI escapes instead of markdown syntax so it reads cleanly on a
// terminal that has no markdown renderer.
func renderPlainTerm(res agentloop.Result, run, of int) string {
	var b strings.Builder

	fmt.Fprintf(&b, "%s%seval_loop — %s%s\n\n", ansiBold, statusANSI(res.Status), strings.ToUpper(string(res.Status)), ansiReset)

	if of > 1 {
		fmt.Fprintf(&b, "  %sRun:%s           %d of %d\n", ansiBold, ansiReset, run, of)
	}
	fmt.Fprintf(&b, "  %sStop reason:%s   %s\n", ansiBold, ansiReset, res.StopReason)
	fmt.Fprintf(&b, "  %sIterations:%s    %d\n", ansiBold, ansiReset, res.Iterations)
	fmt.Fprintf(&b, "  %sTokens used:%s   %s\n", ansiBold, ansiReset, commas(res.TokensUsed))
	changed := "none"
	if len(res.Changes) > 0 {
		changed = strings.Join(res.Changes, ", ")
	}
	fmt.Fprintf(&b, "  %sFiles changed:%s %s\n", ansiBold, ansiReset, changed)

	if res.Summary != "" {
		fmt.Fprintf(&b, "\n%sSummary%s\n  %s\n", ansiBold, ansiReset, res.Summary)
	}
	if res.Detail != "" {
		fmt.Fprintf(&b, "\n%sDetail%s\n  %s\n", ansiBold, ansiReset, res.Detail)
	}
	return b.String()
}

// isTerminal reports whether f is connected to a terminal (as opposed to
// a redirected file or pipe) and the user hasn't opted out via NO_COLOR.
func isTerminal(f *os.File) bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

// glowAvailable reports whether the glow markdown renderer is on PATH.
func glowAvailable() bool {
	_, err := exec.LookPath("glow")
	return err == nil
}

// renderWithGlow pipes md through `glow -` so it prints as rendered
// markdown instead of raw syntax. Any failure (glow missing, crashed,
// etc.) is left for the caller to fall back on.
func renderWithGlow(md string) error {
	cmd := exec.Command("glow", "-")
	cmd.Stdin = strings.NewReader(md)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// commas formats n with thousands separators (e.g. 189944 -> "189,944").
func commas(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	if neg {
		s = "-" + s
	}
	return s
}

// lintAvailable reports whether golangci-lint is on PATH. A gate command
// for a missing binary would park every run, so we avoid adding it
// when absent.
func lintAvailable() bool {
	_, err := exec.LookPath("golangci-lint")
	return err == nil
}

// providerConfigFor derives a runtime.ProviderConfig for id by convention:
// the ai-sdk provider class names match their conventional provider ID
// (openai, anthropic, deepseek, groq, mistral, cohere, gemini,
// perplexity, xai, azure, ollama, openai-compatible, ...), and keys are
// read from "<PROVIDER>_API_KEY". Ollama and openai-compatible (any
// self-hosted server speaking the OpenAI chat protocol) take no key —
// the latter is reached entirely via -host.
func providerConfigFor(id string) runtime.ProviderConfig {
	switch id {
	case "ollama", "openai-compatible":
		return runtime.ProviderConfig{ID: id, Class: id, Auth: runtime.AuthConfig{Type: runtime.AuthTypeNone}}
	}
	return runtime.ProviderConfig{
		ID:    id,
		Class: id,
		Auth:  runtime.AuthConfig{Type: runtime.AuthTypeAPIKey, APIKeyEnv: strings.ToUpper(id) + "_API_KEY"},
	}
}

// loadConfig reads eval_config.json from pkgRoot and returns an
// EvalConfig. If the file doesn't exist or is invalid JSON, it returns
// an empty EvalConfig with a stderr message.
func loadConfig(pkgRoot string) EvalConfig {
	data, err := os.ReadFile(filepath.Join(pkgRoot, "eval_config.json"))
	if err != nil {
		return EvalConfig{}
	}
	var cfg EvalConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Fprintf(os.Stderr, "[config] eval_config.json parse error: %v — using defaults\n", err)
		return EvalConfig{}
	}
	return cfg
}

// normalizeSkipDirs normalizes a list of skip directories: each path is
// converted to the current OS's representation, trailing separators are
// removed, and empty results are dropped. Used for matching walk paths.
func normalizeSkipDirs(dirs []string) []string {
	out := make([]string, 0, len(dirs))
	for _, d := range dirs {
		d = strings.TrimSuffix(filepath.FromSlash(d), string(filepath.Separator))
		if d != "" {
			out = append(out, d)
		}
	}
	return out
}

// shouldSkip returns true if rel matches any of the given skip patterns.
// A match is exact equality or rel starting with pattern+separator.
func shouldSkip(rel string, skip []string) bool {
	for _, s := range skip {
		if rel == s || strings.HasPrefix(rel, s+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// discoverGoFiles walks pkgRoot for .go files, honouring skip dirs.
// Returns a sorted list of relative paths.
func discoverGoFiles(pkgRoot string, skip []string) []string {
	var files []string
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel, skip) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !info.IsDir() && strings.HasSuffix(rel, ".go") {
			files = append(files, rel)
		}
		return nil
	})
	sort.Strings(files)
	return files
}

// sourceFilesFor returns the preload list: cfg.SourceFiles if provided,
// otherwise go.mod first (to establish the language version) then all
// discovered .go files. This list is passed to agentloop as PreloadFiles.
func sourceFilesFor(pkgRoot string, cfg EvalConfig, skip []string) []string {
	if len(cfg.SourceFiles) > 0 {
		return cfg.SourceFiles
	}
	return append([]string{"go.mod"}, discoverGoFiles(pkgRoot, skip)...)
}

// renderSystemPrompt builds the system prompt template by:
// 1. Deriving packageName from cfg.PackageName or falling back to
//    filepath.Base(pkgRoot).
// 2. Building pkgStructure: a sorted list of - go.mod and all discovered
//    .go files (excluding skip dirs).
// 3. Using cfg.BuildCommands if provided, otherwise the default.
// 4. Parsing system_prompt.tmpl with PromptData and returning the rendered
//    string, or an error on template failure.
func renderSystemPrompt(pkgRoot string, cfg EvalConfig, skip []string) (string, error) {
	packageName := cfg.PackageName
	if packageName == "" {
		packageName = filepath.Base(pkgRoot)
	}
	pkgStructure := cfg.PackageStructure
	if pkgStructure == "" {
		var lines []string
		for _, f := range discoverGoFiles(pkgRoot, skip) {
			lines = append(lines, "- "+f)
		}
		lines = append(lines, "- go.mod")
		sort.Strings(lines)
		pkgStructure = strings.Join(lines, "\n")
	}
	buildCmds := cfg.BuildCommands
	if buildCmds == "" {
		buildCmds = "Build:  go build ./...\nVet:    go vet ./...\nTest:   go test ./... -count=1"
	}

	t, err := template.New("system_prompt").Parse(systemPromptTmpl)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, PromptData{
		PackageName:      packageName,
		Description:      cfg.Description,
		PackageStructure: pkgStructure,
		BuildCommands:    buildCmds,
		ExtraRules:       cfg.ExtraRules,
	}); err != nil {
		return "", err
	}
	return buf.String(), nil
}
