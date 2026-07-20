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
//
// The model is a runtime ref ("provider/model"). Keys come from the
// conventional env vars (DEEPSEEK_API_KEY, OPENAI_API_KEY,
// ANTHROPIC_API_KEY); ollama needs none.
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

func run() int {
	showVersion := flag.Bool("version", false, "print version and exit")
	model := flag.String("model", "deepseek/deepseek-chat", "runtime model ref (provider/model)")
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

	runtime.RegisterBuiltinClasses()
	rt := runtime.NewRuntime(runtime.Config{
		Providers: map[string]runtime.ProviderConfig{
			"deepseek":  {ID: "deepseek", Class: "deepseek", Auth: runtime.AuthConfig{APIKeyEnv: "DEEPSEEK_API_KEY"}},
			"openai":    {ID: "openai", Class: "openai", Auth: runtime.AuthConfig{APIKeyEnv: "OPENAI_API_KEY"}},
			"anthropic": {ID: "anthropic", Class: "anthropic", Auth: runtime.AuthConfig{APIKeyEnv: "ANTHROPIC_API_KEY"}},
			"ollama":    {ID: "ollama", Class: "ollama", Auth: runtime.AuthConfig{Type: runtime.AuthTypeNone}},
		},
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
