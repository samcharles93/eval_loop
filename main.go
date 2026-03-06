// eval_loop — generic Go package evaluator powered by an LLM agent.
//
// The agent reads source files, evaluates code quality across five priority
// levels, and applies targeted improvements. A file-hash watcher enforces
// validation (go vet + go fix) after every edit before other tools unblock.
// Persistent AGENT_NOTES.md accumulates verified knowledge across runs.
//
// Usage:
//
//	go run . ./mypackage
//
// Alternatively, set PKG_PATH=./mypackage. ANTHROPIC_BASE_URL and
// ANTHROPIC_API_KEY must be set in the environment.
//
// Optional: place eval_config.json in the package root to provide
// package-specific context (description, build commands, extra rules).
// Without it the harness auto-discovers .go files and uses generic commands.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"text/template"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/toolrunner"
)

var (
    version = "dev"
    commit  = "unknown"
)

// ── colour helpers ────────────────────────────────────────────────────────────

func cc(code, s string) string { return fmt.Sprintf("\033[1;%sm%s\033[0m", code, s) }
func cUser(s string) string    { return cc("36", s) }
func cAI(s string) string      { return cc("32", s) }
func cTool(s string) string    { return cc("33", s) }
func cThink(s string) string   { return cc("90", s) }
func cErr(s string) string     { return cc("31", s) }
func cInfo(s string) string    { return cc("34", s) }

// logWalkError prints a filepath.Walk error to stderr with a watcher tag.
func logWalkError(path string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s walk error on %q: %v\n", cErr("[watcher]"), path, err)
	}
}

// ── change log ────────────────────────────────────────────────────────────────

type Change struct {
	Time    string `json:"time"`
	File    string `json:"file"`
	Summary string `json:"summary"`
}

var changeLog []Change

func recordChange(file, summary string) {
	changeLog = append(changeLog, Change{
		Time:    time.Now().Format("15:04:05"),
		File:    file,
		Summary: summary,
	})
}

func printChangeLog() {
	if len(changeLog) == 0 {
		fmt.Println(cInfo("No changes were made."))
		return
	}
	fmt.Println(cInfo(fmt.Sprintf("\n══ Change log (%d edits) ══", len(changeLog))))
	for i, c := range changeLog {
		fmt.Printf("  %s  [%s] %s — %s\n",
			cc("90", c.Time),
			cc("33", fmt.Sprintf("%02d", i+1)),
			cc("36", c.File),
			c.Summary,
		)
	}
}

// ── eval config ───────────────────────────────────────────────────────────────

// EvalConfig holds optional per-project configuration loaded from
// eval_config.json in the package root. All fields are optional — the harness
// works with sensible defaults if the file is absent.
type EvalConfig struct {
	PackageName      string   `json:"package_name"`
	Description      string   `json:"description"`
	SourceFiles      []string `json:"source_files"`      // explicit list; if empty, auto-discovered
	SkipDirs         []string `json:"skip_dirs"`         // dirs to exclude from walk
	PackageStructure string   `json:"package_structure"` // freeform description for the prompt
	BuildCommands    string   `json:"build_commands"`    // freeform build/test commands
	ExtraRules       []string `json:"extra_rules"`       // additional <rules> bullet points
}

func loadConfig(pkgRoot string) EvalConfig {
	data, err := os.ReadFile(filepath.Join(pkgRoot, "eval_config.json"))
	if err != nil {
		return EvalConfig{} // no config — use all defaults
	}
	var cfg EvalConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Printf(cErr("[config] eval_config.json parse error: %v — using defaults\n"), err)
		return EvalConfig{}
	}
	fmt.Printf(cInfo("[config] loaded eval_config.json for %q\n"), cfg.PackageName)
	return cfg
}

// ── package root & agent notes ────────────────────────────────────────────────

var pkgRoot string

const agentNotesFile = "AGENT_NOTES.md"

func notesPath() string { return filepath.Join(pkgRoot, agentNotesFile) }

func readNotes() string {
	data, err := os.ReadFile(notesPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func appendNote(category, note string) error {
	entry := fmt.Sprintf("\n## [%s] %s\n%s\n",
		time.Now().Format("2006-01-02 15:04:05"),
		category,
		strings.TrimSpace(note),
	)
	f, err := os.OpenFile(notesPath(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(entry)
	return err
}

// ── file watcher / dirty gate ─────────────────────────────────────────────────
//
// Hashes every .go file at startup. After any write, the hash changes and that
// file enters the dirty set. Gated tools (list_files, read_file, fetch_docs)
// return BLOCKED until the agent runs "go vet ./..." and "run_go_fix dry_run=true"
// cleanly. A clean go fix dry-run calls clearDirty() which re-snapshots.

var knownHashes = map[string]string{}

const buildFailedSentinel = "BUILD_FAILED"

func fileHash(absPath string) string {
	f, err := os.Open(absPath)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	_, _ = io.Copy(h, f)
	return hex.EncodeToString(h.Sum(nil))
}

var skipDirs = []string{".git", "bin", "vendor"} // overridden from config in main

// normalizeSkipDirs converts forward slashes to OS-specific separators
// and removes any trailing separators.
func normalizeSkipDirs(dirs []string) []string {
	out := make([]string, 0, len(dirs))
	for _, d := range dirs {
		d = filepath.FromSlash(d)
		d = strings.TrimSuffix(d, string(filepath.Separator))
		if d != "" {
			out = append(out, d)
		}
	}
	return out
}

func shouldSkip(rel string) bool {
	for _, s := range skipDirs {
		if strings.HasPrefix(rel, s+string(filepath.Separator)) || rel == s {
			return true
		}
	}
	return false
}

func snapshotHashes() {
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel) {
			return nil
		}
		knownHashes[rel] = fileHash(path)
		return nil
	})
	fmt.Printf(cInfo("[watcher] snapshotted %d .go files\n"), len(knownHashes))
}

func dirtySet() map[string]struct{} {
	dirty := map[string]struct{}{}
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel) {
			return nil
		}
		old, known := knownHashes[rel]
		if !known || fileHash(path) != old {
			dirty[rel] = struct{}{}
		}
		return nil
	})
	return dirty
}

func clearDirty() {
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil || info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if !shouldSkip(rel) {
			knownHashes[rel] = fileHash(path)
		}
		return nil
	})
	fmt.Printf("%s", cInfo("[watcher] dirty gate cleared\n"))
}

func blocked() *anthropic.BetaToolResultBlockParamContentUnion {
	dirty := dirtySet()
	if len(dirty) == 0 {
		return nil
	}
	files := make([]string, 0, len(dirty))
	for f := range dirty {
		files = append(files, "  • "+f)
	}
	sort.Strings(files)
	msg := fmt.Sprintf(
		"BLOCKED — %d .go file(s) modified since last validation:\n%s\n\n"+
			"Run these two tools to unblock everything:\n"+
			"  1. run_command  →  \"go vet ./...\"\n"+
			"  2. run_go_fix   →  dry_run=true\n\n"+
			"A clean pass on both clears the gate.",
		len(dirty), strings.Join(files, "\n"),
	)
	fmt.Printf("  %s\n", cErr(fmt.Sprintf("[watcher] BLOCKED (%d dirty)", len(dirty))))
	r := textResult(msg)
	return &r
}

// ── tool input types ──────────────────────────────────────────────────────────

type ListFilesInput struct {
	SubDir string `json:"sub_dir,omitempty" jsonschema:"description=Optional sub-directory (empty = package root)"`
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema:"required,description=Relative path inside the package"`
}

type WriteFileInput struct {
	Path    string `json:"path"    jsonschema:"required,description=Relative path inside the package"`
	Content string `json:"content" jsonschema:"required,description=Complete new file content"`
	Summary string `json:"summary" jsonschema:"required,description=One-line description of what changed and why"`
}

type ApplyPatchInput struct {
	Path    string `json:"path"     jsonschema:"required,description=Relative path inside the package"`
	OldText string `json:"old_text" jsonschema:"required,description=Exact text to replace — must appear exactly once"`
	NewText string `json:"new_text" jsonschema:"required,description=Replacement text"`
	Summary string `json:"summary"  jsonschema:"required,description=One-line description of what changed and why"`
}

type RunCommandInput struct {
	Command string `json:"command" jsonschema:"required,description=Shell command to run in the package directory"`
}

type FetchDocsInput struct {
	Query string `json:"query" jsonschema:"required,description=Package path or symbol for go doc (e.g. 'sync', 'math/bits', 'os.ReadFile')"`
}

type RunGoFixInput struct {
	DryRun bool `json:"dry_run" jsonschema:"required,description=true=show diff without applying; a clean result clears the dirty gate. false=apply fixes."`
}

type WriteNoteInput struct {
	Category   string `json:"category"    jsonschema:"required,description=go-idiom | correctness | performance | vet | other"`
	Note       string `json:"note"        jsonschema:"required,description=Actionable plain-language rule for future agents"`
	VerifiedBy string `json:"verified_by" jsonschema:"required,description=The exact command you ran that proves this note is correct (e.g. 'go doc sync.WaitGroup', 'go fix -diff ./...')"`
}

type DeleteNoteInput struct {
	Reason string `json:"reason" jsonschema:"required,description=Why this note is wrong or outdated"`
}

// ── tool implementations ──────────────────────────────────────────────────────
//
// Gated (blocked while dirty):    list_files, read_file, fetch_docs
// Always available (escape hatch): run_command, run_go_fix, write_note, delete_notes
// Write tools (mark dirty, never blocked): write_file, apply_patch

func toolListFiles(_ context.Context, in ListFilesInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	if r := blocked(); r != nil {
		return *r, nil
	}
	root := pkgRoot
	if in.SubDir != "" {
		root = filepath.Join(pkgRoot, in.SubDir)
	}
	var files []string
	_ = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !info.IsDir() {
			files = append(files, rel)
		}
		return nil
	})
	fmt.Printf("%s listed %d files\n", cTool("[list_files]"), len(files))
	out, _ := json.Marshal(files)
	return textResult(string(out)), nil
}

func toolReadFile(_ context.Context, in ReadFileInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	if r := blocked(); r != nil {
		return *r, nil
	}
	data, err := os.ReadFile(filepath.Join(pkgRoot, filepath.Clean(in.Path)))
	if err != nil {
		return textResult("ERROR: " + err.Error()), nil
	}
	fmt.Printf("%s %s (%d bytes)\n", cTool("[read_file]"), in.Path, len(data))
	return textResult(string(data)), nil
}

func toolWriteFile(_ context.Context, in WriteFileInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	full := filepath.Join(pkgRoot, filepath.Clean(in.Path))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return textResult("ERROR mkdir: " + err.Error()), nil
	}
	if err := os.WriteFile(full, []byte(in.Content), 0o644); err != nil {
		return textResult("ERROR write: " + err.Error()), nil
	}
	recordChange(in.Path, in.Summary)
	fmt.Printf("%s %s — %s\n", cTool("[write_file]"), in.Path, in.Summary)
	return textResult(fmt.Sprintf("OK: wrote %d bytes to %s", len(in.Content), in.Path)), nil
}

func toolApplyPatch(_ context.Context, in ApplyPatchInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	full := filepath.Join(pkgRoot, filepath.Clean(in.Path))
	data, err := os.ReadFile(full)
	if err != nil {
		return textResult("ERROR reading: " + err.Error()), nil
	}
	src := string(data)
	n := strings.Count(src, in.OldText)
	switch n {
	case 0:
		return textResult("ERROR: old_text not found — check exact whitespace and line endings"), nil
	case 1:
		// good
	default:
		return textResult(fmt.Sprintf("ERROR: old_text found %d times — add more context to make it unique", n)), nil
	}
	if err := os.WriteFile(full, []byte(strings.Replace(src, in.OldText, in.NewText, 1)), 0o644); err != nil {
		return textResult("ERROR writing: " + err.Error()), nil
	}
	recordChange(in.Path, in.Summary)
	fmt.Printf("%s %s — %s\n", cTool("[apply_patch]"), in.Path, in.Summary)
	return textResult("OK: patch applied to " + in.Path), nil
}

func toolRunCommand(_ context.Context, in RunCommandInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	fmt.Printf("%s %s\n", cTool("[run_command]"), in.Command)
	cmd := exec.Command("sh", "-c", in.Command)
	cmd.Dir = pkgRoot
	out, err := cmd.CombinedOutput()
	result := strings.TrimSpace(string(out))

	isBuild := strings.Contains(in.Command, "go build")
	isVet := strings.Contains(in.Command, "go vet")

	if err != nil {
		msg := fmt.Sprintf("EXIT ERROR: %v\n%s", err, result)
		fmt.Printf("  %s\n", cErr(truncate(msg, 400)))
		if isBuild {
			dirty := dirtySet()
			if len(dirty) > 0 {
				for rel := range dirty {
					knownHashes[rel] = buildFailedSentinel
				}
				msg += "\n\nBuild failed — dirty gate re-locked. Fix the build error, " +
					"then re-run 'go vet ./...' and 'run_go_fix dry_run=true' to clear it. " +
					"Check your build command and flags before reverting any code."
			}
		}
		return textResult(msg), nil
	}
	if result == "" {
		result = "(no output, exit 0)"
	}
	fmt.Printf("  %s\n", cc("90", truncate(result, 300)))
	if isVet && len(dirtySet()) > 0 {
		result += "\n\ngo vet passed ✓  Now run run_go_fix with dry_run=true to complete validation and clear the dirty gate."
	}
	return textResult(result), nil
}

func toolFetchDocs(_ context.Context, in FetchDocsInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	if r := blocked(); r != nil {
		return *r, nil
	}
	fmt.Printf("%s %q\n", cTool("[fetch_docs]"), in.Query)

	cmd := exec.Command("go", "doc", "-all", in.Query)
	cmd.Dir = pkgRoot
	out, err := cmd.CombinedOutput()
	if err == nil && len(out) > 0 {
		fmt.Printf("  go doc: %d bytes\n", len(out))
		return textResult(truncate(string(out), 6000)), nil
	}

	// Fall back to pkg.go.dev for packages not resolvable locally.
	fields := strings.Fields(in.Query)
	if len(fields) == 0 {
		return textResult("ERROR: empty query"), nil
	}
	pkg := fields[0]
	curl := exec.Command("curl", "-s", "-L", "--max-time", "10",
		"-H", "Accept: text/plain", "https://pkg.go.dev/"+pkg)
	curlOut, curlErr := curl.CombinedOutput()
	if curlErr == nil && len(curlOut) > 200 {
		fmt.Printf("  pkg.go.dev: %d bytes\n", len(curlOut))
		return textResult(truncate(string(curlOut), 6000)), nil
	}

	return textResult(fmt.Sprintf("go doc failed for %q: %s", in.Query, string(out))), nil
}

func toolRunGoFix(_ context.Context, in RunGoFixInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	args := []string{"fix"}
	if in.DryRun {
		args = append(args, "-diff")
	}
	args = append(args, "./...")
	label := "go fix ./..."
	if in.DryRun {
		label = "go fix -diff ./... (dry run)"
	}
	fmt.Printf("%s %s\n", cTool("[run_go_fix]"), label)

	cmd := exec.Command("go", args...)
	cmd.Dir = pkgRoot
	out, err := cmd.CombinedOutput()
	result := strings.TrimSpace(string(out))
	if err != nil && result == "" {
		result = fmt.Sprintf("EXIT ERROR: %v", err)
	}

	if !in.DryRun {
		if result == "" {
			result = "(no output, exit 0)"
		}
		fmt.Printf("  %s\n", cc("90", truncate(result, 300)))
		return textResult("go fix applied: " + result), nil
	}

	if result == "" {
		clearDirty()
		return textResult("go fix: nothing to change ✓  Dirty gate cleared — all tools unblocked."), nil
	}

	fmt.Printf("  %s\n", cErr("[run_go_fix] diff found — gate stays locked"))
	fmt.Printf("  %s\n", cc("90", truncate(result, 400)))
	return textResult(fmt.Sprintf(
		"go fix found modernisation issues.\n\nDiff:\n%s\n\n"+
			"The dirty gate will NOT clear until:\n"+
			"  1. write_note  — explain in plain language which idioms are correct for this Go version\n"+
			"  2. run_go_fix  — dry_run=false  (apply the fixes)\n"+
			"  3. run_go_fix  — dry_run=true   (confirm clean — this clears the gate)\n\n"+
			"Only a clean dry_run=true result clears the gate.",
		result,
	)), nil
}

func toolWriteNote(_ context.Context, in WriteNoteInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	if strings.TrimSpace(in.VerifiedBy) == "" {
		return textResult("ERROR: verified_by is required. Cite the command you ran that proves this note is correct."), nil
	}
	noteWithEvidence := fmt.Sprintf("%s\n\nVerified by: %s", strings.TrimSpace(in.Note), strings.TrimSpace(in.VerifiedBy))
	if err := appendNote(in.Category, noteWithEvidence); err != nil {
		return textResult("ERROR writing note: " + err.Error()), nil
	}
	fmt.Printf("%s [%s] %s\n", cTool("[write_note]"), cc("35", in.Category), truncate(in.Note, 120))
	fmt.Printf("  verified by: %s\n", cc("90", in.VerifiedBy))
	return textResult("OK: note saved to " + agentNotesFile), nil
}

func toolDeleteNotes(_ context.Context, in DeleteNoteInput) (anthropic.BetaToolResultBlockParamContentUnion, error) {
	if err := os.Remove(notesPath()); err != nil && !os.IsNotExist(err) {
		return textResult("ERROR deleting notes: " + err.Error()), nil
	}
	fmt.Printf("%s %s\n", cTool("[delete_notes]"), cErr("wiped — "+in.Reason))
	return textResult(fmt.Sprintf("AGENT_NOTES.md deleted. Reason: %s", in.Reason)), nil
}

// ── helpers ───────────────────────────────────────────────────────────────────

func textResult(s string) anthropic.BetaToolResultBlockParamContentUnion {
	return anthropic.BetaToolResultBlockParamContentUnion{
		OfText: &anthropic.BetaTextBlockParam{Text: s},
	}
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + fmt.Sprintf("\n…(truncated, %d bytes total)", len(s))
}

func readSource(rel string) string {
	data, err := os.ReadFile(filepath.Join(pkgRoot, rel))
	if err != nil {
		return fmt.Sprintf("(could not read %s: %v)", rel, err)
	}
	return string(data)
}

func goDocAll() string {
	cmd := exec.Command("go", "doc", "-all", ".")
	cmd.Dir = pkgRoot
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("(go doc failed: %v\n%s)", err, string(out))
	}
	return string(out)
}

// ── preflight ─────────────────────────────────────────────────────────────────

type PreflightResult struct {
	GoVersion string
	GoFixDiff string // empty = source already idiomatic
}

func runPreflight() PreflightResult {
	var r PreflightResult

	ver := exec.Command("go", "version")
	ver.Dir = pkgRoot
	verOut, _ := ver.CombinedOutput()
	r.GoVersion = strings.TrimSpace(string(verOut))

	fix := exec.Command("go", "fix", "-diff", "./...")
	fix.Dir = pkgRoot
	fixOut, _ := fix.CombinedOutput()
	r.GoFixDiff = strings.TrimSpace(string(fixOut))

	fmt.Printf(cInfo("[preflight] %s\n"), r.GoVersion)
	if r.GoFixDiff != "" {
		fmt.Printf("%s", cInfo("[preflight] go fix -diff produced output — injecting into context\n"))
	}
	return r
}

// ── system prompt template ────────────────────────────────────────────────────

// PromptData is passed to system_prompt.tmpl.
type PromptData struct {
	PackageName      string
	Description      string
	PackageStructure string
	BuildCommands    string
	ExtraRules       []string
}

// defaultBuildCommands returns generic Go build/test commands.
func defaultBuildCommands() string {
	return "Build:  go build ./...\nVet:    go vet ./...\nTest:   go test ./... -count=1"
}

// defaultPackageStructure auto-generates a file listing from the package root.
func defaultPackageStructure(_ EvalConfig) string {
	var lines []string
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil || info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel) {
			return nil
		}
		if strings.HasSuffix(rel, ".go") || rel == "go.mod" || rel == "go.sum" {
			lines = append(lines, "- "+rel)
		}
		return nil
	})
	sort.Strings(lines)
	return strings.Join(lines, "\n")
}

//go:embed system_prompt.tmpl
var systemPromptTmpl string

func renderSystemPrompt(cfg EvalConfig) string {
	packageName := cfg.PackageName
	if packageName == "" {
		packageName = filepath.Base(pkgRoot)
	}
	pkgStructure := cfg.PackageStructure
	if pkgStructure == "" {
		pkgStructure = defaultPackageStructure(cfg)
	}
	buildCmds := cfg.BuildCommands
	if buildCmds == "" {
		buildCmds = defaultBuildCommands()
	}

	data := PromptData{
		PackageName:      packageName,
		Description:      cfg.Description,
		PackageStructure: pkgStructure,
		BuildCommands:    buildCmds,
		ExtraRules:       cfg.ExtraRules,
	}

	t, err := template.New("system_prompt").Parse(systemPromptTmpl)
	if err != nil {
		fmt.Printf(cErr("[prompt] template parse error: %v\n"), err)
		return "You are a senior Go engineer. Evaluate and improve the provided package."
	}

	var buf bytes.Buffer
	if err := t.Execute(&buf, data); err != nil {
		fmt.Printf(cErr("[prompt] template execute error: %v\n"), err)
		return "You are a senior Go engineer. Evaluate and improve the provided package."
	}
	return buf.String()
}

// ── source file discovery ─────────────────────────────────────────────────────

// sourceFilesFor returns the ordered list of source files to preload.
// Uses cfg.SourceFiles if provided; otherwise discovers all .go + go.mod files,
// with go.mod always first so the agent sees the language version immediately.
func sourceFilesFor(cfg EvalConfig) []string {
	if len(cfg.SourceFiles) > 0 {
		return cfg.SourceFiles
	}
	var goFiles []string
	_ = filepath.Walk(pkgRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			logWalkError(path, err)
			return nil
		}
		if info == nil || info.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(pkgRoot, path)
		if shouldSkip(rel) {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(rel, ".go") {
			goFiles = append(goFiles, rel)
		}
		return nil
	})
	sort.Strings(goFiles)
	// go.mod first — establishes the language version before any source file
	result := []string{"go.mod"}
	for _, f := range goFiles {
		result = append(result, f)
	}
	return result
}

// ── streaming event loop ──────────────────────────────────────────────────────

const (
	maxTotalIter = 200 // hard safety ceiling
	maxIdleIter  = 2   // consecutive tool-call-free turns before stopping
)

func streamEvents(ctx context.Context, runner *anthropic.BetaToolRunnerStreaming) error {
	idleIters := 0
	for eventsIter := range runner.AllStreaming(ctx) {
		hadToolCall := false
		for event, err := range eventsIter {
			if err != nil {
				fmt.Printf("%s %v\n", cErr("[error]:"), err)
				return err
			}
			switch v := event.AsAny().(type) {
			case anthropic.BetaRawMessageStartEvent:
				iter := runner.IterationCount()
				fmt.Printf("\n%s [iter %d] ", cAI("[assistant]"), iter)
				if iter >= maxTotalIter {
					fmt.Printf("\n%s reached %d iterations — stopping\n", cErr("[eval_loop]"), maxTotalIter)
					return nil
				}
			case anthropic.BetaRawContentBlockStartEvent:
				switch cb := v.ContentBlock.AsAny().(type) {
				case anthropic.BetaToolUseBlock:
					hadToolCall = true
					fmt.Printf("\n%s", cTool(fmt.Sprintf("[→ %s] ", cb.Name)))
				case anthropic.BetaThinkingBlock:
					fmt.Print(cThink("\n[thinking] "))
				}
			case anthropic.BetaRawContentBlockDeltaEvent:
				switch d := v.Delta.AsAny().(type) {
				case anthropic.BetaTextDelta:
					fmt.Print(cAI(d.Text))
				case anthropic.BetaInputJSONDelta:
					if d.PartialJSON != "" {
						fmt.Print(cTool(d.PartialJSON))
					}
				case anthropic.BetaThinkingDelta:
					fmt.Print(cThink(d.Thinking))
				}
			case anthropic.BetaRawContentBlockStopEvent:
				fmt.Println()
			case anthropic.BetaRawMessageStopEvent:
				fmt.Println()
			}
		}
		if !hadToolCall {
			idleIters++
			if idleIters >= maxIdleIter {
				fmt.Printf(cInfo("[eval_loop] agent idle for %d turn(s) — done\n"), idleIters)
				return nil
			}
		} else {
			idleIters = 0
		}
	}
	return nil
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	// ── resolve package root ──────────────────────────────────────────────────
	if len(os.Args) > 1 {
		pkgRoot = os.Args[1]
	} else if env := os.Getenv("PKG_PATH"); env != "" {
		pkgRoot = env
	} else {
		pkgRoot = "."
	}
	abs, err := filepath.Abs(pkgRoot)
	if err != nil {
		fmt.Printf(cErr("cannot resolve package root %q: %v\n"), pkgRoot, err)
		os.Exit(1)
	}
	pkgRoot = abs

	fmt.Printf(cInfo("[eval_loop] version: %s  commit: %s\n"), version, commit)

	if info, err := os.Stat(pkgRoot); err != nil || !info.IsDir() {
	    fmt.Printf(cErr("package root %q does not exist or is not a directory\n"), pkgRoot)
	    os.Exit(1)
	}

	fmt.Println(cInfo("[eval_loop] package root: " + pkgRoot))

	// ── load config & apply skip dirs ────────────────────────────────────────
	cfg := loadConfig(pkgRoot)
	if len(cfg.SkipDirs) > 0 {
		skipDirs = normalizeSkipDirs(cfg.SkipDirs)
	}

	// ── snapshot file hashes (defines clean baseline) ────────────────────────
	snapshotHashes()

	// ── render system prompt from template ───────────────────────────────────
	systemPrompt := renderSystemPrompt(cfg)

	// ── run preflight (go version + go fix -diff) ────────────────────────────
	pre := runPreflight()

	// ── build initial user message ───────────────────────────────────────────
	fmt.Print(cInfo("[eval_loop] building initial context..."))
	var sb strings.Builder
	sb.WriteString("Here is the current state of the package.\n\n")

	// Preflight: ground-truth toolchain facts
	sb.WriteString("<preflight>\n")
	sb.WriteString("Established by the harness before you started. Treat as ground truth.\n\n")
	sb.WriteString("## Installed Go toolchain\n")
	sb.WriteString(pre.GoVersion + "\n\n")
	sb.WriteString("## go fix -diff (against current source)\n")
	if pre.GoFixDiff == "" {
		sb.WriteString("(no output — source is already idiomatic for this toolchain)\n")
	} else {
		sb.WriteString(pre.GoFixDiff + "\n\n")
		sb.WriteString("IMPORTANT: the diff above shows what this toolchain considers correct. " +
			"These are real APIs. If applying them causes a build failure, " +
			"check your build command before reverting.\n")
	}
	sb.WriteString("</preflight>\n\n")

	// Agent notes from prior runs
	if notes := readNotes(); notes != "" {
		sb.WriteString("<agent_notes>\n")
		sb.WriteString("Written by previous agent runs. Verify claims before acting if they contradict what you observe. Use delete_notes if you find misinformation.\n\n")
		sb.WriteString(notes)
		sb.WriteString("\n</agent_notes>\n\n")
		fmt.Printf(cInfo(" (%d bytes agent notes)"), len(notes))
	}

	// go doc — full exported API surface
	sb.WriteString("<go_doc>\n")
	sb.WriteString(goDocAll())
	sb.WriteString("\n</go_doc>\n\n")

	// Source files
	srcFiles := sourceFilesFor(cfg)
	sb.WriteString("<source_files>\n")
	for _, f := range srcFiles {
		fmt.Fprintf(&sb, "<file path=%q>\n", f)
		sb.WriteString(readSource(f))
		sb.WriteString("\n</file>\n\n")
	}
	sb.WriteString("</source_files>\n\n")
	sb.WriteString("Begin your evaluation. After every edit, validate with 'go vet ./...' then " +
		"'run_go_fix dry_run=true' — other tools are blocked until you do. " +
		"Use write_note for non-obvious discoveries. Use parallel tool calls for independent operations.")

	fmt.Println(cc("90", fmt.Sprintf(" done (%d bytes)\n", sb.Len())))

	// ── register tools ────────────────────────────────────────────────────────
	if os.Getenv("ANTHROPIC_BASE_URL") == "" || os.Getenv("ANTHROPIC_API_KEY") == "" {
		fmt.Println(cErr("ERROR: ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY environment variables must be set"))
		os.Exit(1)
	}
	client := anthropic.NewClient(
		option.WithBaseURL(os.Getenv("ANTHROPIC_BASE_URL")),
	)
	ctx := context.Background()

	listFilesTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"list_files", "List source files. BLOCKED while .go files are dirty.", toolListFiles)
	readFileTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"read_file", "Read a source file. BLOCKED while .go files are dirty.", toolReadFile)
	writeFileTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"write_file", "Overwrite a file. Marks .go files dirty — validate with go vet + run_go_fix to unblock.", toolWriteFile)
	applyPatchTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"apply_patch", "Replace a unique snippet. Marks .go files dirty. old_text must appear exactly once.", toolApplyPatch)
	runCommandTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"run_command", "Run a shell command. Always available. Use 'go vet ./...' as first validation step.", toolRunCommand)
	fetchDocsTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"fetch_docs", "Fetch Go documentation via go doc or pkg.go.dev. BLOCKED while .go files are dirty.", toolFetchDocs)
	runGoFixTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"run_go_fix", "Run go fix. Always available. dry_run=true shows diff and clears the dirty gate if clean.", toolRunGoFix)
	writeNoteTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"write_note", "Append a verified note to AGENT_NOTES.md. Requires verified_by field citing the command that proves the note correct.", toolWriteNote)
	deleteNotesTool, _ := toolrunner.NewBetaToolFromJSONSchema(
		"delete_notes", "Wipe AGENT_NOTES.md. Use when it contains verified misinformation. Requires a reason.", toolDeleteNotes)

	tools := []anthropic.BetaTool{
		listFilesTool, readFileTool, writeFileTool, applyPatchTool,
		runCommandTool, fetchDocsTool, runGoFixTool, writeNoteTool, deleteNotesTool,
	}

	fmt.Println(cUser("[user]: ") + "(starting agent loop)")

	runner := client.Beta.Messages.NewToolRunnerStreaming(tools, anthropic.BetaToolRunnerParams{
		BetaMessageNewParams: anthropic.BetaMessageNewParams{
			Model:     "deepseek-chat",
			MaxTokens: 16000,
			System:    []anthropic.BetaTextBlockParam{{Text: systemPrompt}},
			Messages: []anthropic.BetaMessageParam{
				anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock(sb.String())),
			},
			Thinking: anthropic.BetaThinkingConfigParamUnion{
				OfAdaptive: &anthropic.BetaThinkingConfigAdaptiveParam{},
			},
		},
		// No MaxIterations — natural termination via idle-turn detection in streamEvents.
	})

	if err := streamEvents(ctx, runner); err != nil {
		printChangeLog()
		os.Exit(1)
	}
	printChangeLog()
	fmt.Println(cInfo("\n[eval_loop] done — review changes in " + pkgRoot))
}
