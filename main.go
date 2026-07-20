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
//
// The model is a runtime ref ("provider/model"); the provider must name
// one of the ai-sdk built-in classes (openai, anthropic, deepseek, groq,
// mistral, cohere, gemini, perplexity, xai, azure, ollama, ...). Keys are
// read from the conventional "<PROVIDER>_API_KEY" env var; ollama needs
// none. -host overrides that provider's base URL for the run, for a
// remote ollama server or a cloud-compatible endpoint (e.g. an Azure
// OpenAI gateway or self-hosted proxy).
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
	"strings"
	"text/template"

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

Flags:
`)
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, `
The provider in -model must name an ai-sdk built-in class (openai,
anthropic, deepseek, groq, mistral, cohere, gemini, perplexity, xai,
azure, ollama, ...). API keys are read from "<PROVIDER>_API_KEY"
(e.g. OPENAI_API_KEY, GROQ_API_KEY). Ollama needs no key.

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

	system, err := renderSystemPrompt(pkgRoot, cfg, skip)
	if err != nil {
		fmt.Fprintf(os.Stderr, "render system prompt: %v\n", err)
		return 1
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

	var preload []string
	if *contextMode == "preload" {
		preload = sourceFilesFor(pkgRoot, cfg, skip)
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

	res, err := agentloop.Run(context.Background(), agentloop.Config{
		Runtime:  rt,
		ModelRef: *model,
		WorkDir:  pkgRoot,
		// The template is fully rendered here; agentloop sees no actions.
		SystemTmpl: system,
		Mission: "Perform the iterative code-quality pass described in your instructions. " +
			"Work through the priorities in order, validate every change against the gate, " +
			"and call finish with a summary of what you improved and what you verified.",
		PreloadFiles: preload,
		Notes:        agentloop.FileNotesStore{Path: filepath.Join(pkgRoot, "AGENT_NOTES.md")},
		Gate:         agentloop.GateConfig{Commands: gate, MaxConsecutiveFailures: 5},
		Preflight: []agentloop.GateCommand{
			{Name: "go-version", Argv: []string{"go", "version"}},
			{Name: "go-fix-diff", Argv: []string{"go", "fix", "-diff", "./..."}},
			{Name: "go-doc", Argv: []string{"go", "doc", "-all", "."}},
		},
		Budget: agentloop.Budget{MaxSteps: *maxSteps, MaxTokens: *maxTokens},
		Logger: slog.New(slog.NewTextHandler(os.Stderr, nil)),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "eval_loop: %v\n", err)
		return 1
	}

	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))

	switch res.Status {
	case agentloop.StatusPassed, agentloop.StatusIdle:
		return 0
	default:
		return 2
	}
}

// lintAvailable reports whether golangci-lint is on PATH; a gate command
// for a missing binary would park every run.
func lintAvailable() bool {
	_, err := exec.LookPath("golangci-lint")
	return err == nil
}

// providerConfigFor derives a runtime.ProviderConfig for id by convention
// rather than a hardcoded registry: the ai-sdk provider class names match
// their conventional provider ID (openai, anthropic, deepseek, groq,
// mistral, cohere, gemini, perplexity, xai, azure, ollama, ...), and keys
// are read from "<PROVIDER>_API_KEY". ollama is the one built-in class
// that takes no key.
func providerConfigFor(id string) runtime.ProviderConfig {
	if id == "ollama" {
		return runtime.ProviderConfig{ID: id, Class: id, Auth: runtime.AuthConfig{Type: runtime.AuthTypeNone}}
	}
	return runtime.ProviderConfig{
		ID:    id,
		Class: id,
		Auth:  runtime.AuthConfig{Type: runtime.AuthTypeAPIKey, APIKeyEnv: strings.ToUpper(id) + "_API_KEY"},
	}
}

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

func shouldSkip(rel string, skip []string) bool {
	for _, s := range skip {
		if rel == s || strings.HasPrefix(rel, s+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// discoverGoFiles walks pkgRoot for .go files, honouring skip dirs.
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
// otherwise go.mod first (establishes the language version) then all
// discovered .go files.
func sourceFilesFor(pkgRoot string, cfg EvalConfig, skip []string) []string {
	if len(cfg.SourceFiles) > 0 {
		return cfg.SourceFiles
	}
	return append([]string{"go.mod"}, discoverGoFiles(pkgRoot, skip)...)
}

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
