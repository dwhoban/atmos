// Package opencode provides an AI provider that invokes the opencode CLI
// (https://opencode.ai) as a subprocess, reusing the user's opencode setup and
// whichever model provider they have configured/authenticated there.
package opencode

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	errUtils "github.com/cloudposse/atmos/errors"
	"github.com/cloudposse/atmos/pkg/ai/agent/base"
	"github.com/cloudposse/atmos/pkg/ai/tools"
	"github.com/cloudposse/atmos/pkg/ai/types"
	log "github.com/cloudposse/atmos/pkg/logger"
	mcpclient "github.com/cloudposse/atmos/pkg/mcp/client"
	"github.com/cloudposse/atmos/pkg/perf"
	"github.com/cloudposse/atmos/pkg/schema"
	"github.com/cloudposse/atmos/pkg/ui"
)

const (
	// ProviderName is the name of this provider for configuration lookup.
	ProviderName = "opencode"
	// DefaultBinary is the default binary name for the opencode CLI.
	DefaultBinary = "opencode"
	// ConfigEnvVar points opencode at a specific config file. Its precedence sits between
	// the global and project configs, and opencode deep-merges it, so pointing it at a
	// temp file that only defines `mcp` servers adds those servers without disturbing the
	// user's own opencode.json. On v2 this only takes effect in --standalone mode (see
	// buildArgs): the default background service resolves config in its own process and
	// ignores this variable in the invoking environment.
	ConfigEnvVar = "OPENCODE_CONFIG"
	// Schema reference embedded in the generated opencode config file.
	configSchemaURL = "https://opencode.ai/config.json"
	// File mode for the temp MCP config (owner-only read/write).
	configFilePerms = 0o600
	// Separator used to join prompt sections (Atmos memory, system prompt, message
	// history) when a conversation is flattened into a single `run` prompt.
	promptSeparator = "\n\n"
	// Timeout for the `opencode --version` probe run once per client construction;
	// a hung binary must not stall provider creation.
	versionProbeTimeout = 5 * time.Second
)

// Client invokes the opencode CLI in non-interactive ("run") mode. Authentication and
// model-provider selection are handled by opencode itself (`opencode auth login`), the
// same way Atmos relies on the user's existing `terraform`/`tofu` installation.
type Client struct {
	binaryPath    string
	model         string
	fullAuto      bool
	mcpServers    map[string]schema.MCPServerConfig
	toolchainPATH string
	hasMCPServers bool // True if MCP servers were configured for pass-through.
	// nativeSessions enables types.NativeSessionClient behavior (provider-side session
	// continuation for `atmos ai chat`). Requires majorVersion >= 2 and no explicit
	// native_sessions: false in provider config.
	nativeSessions bool
	// majorVersion is the detected opencode major version. 0 behaves as v1 (the zero
	// value keeps struct-literal-constructed clients on the legacy invocation path);
	// NewClient always sets it via a --version probe, assuming v2 when the probe fails.
	majorVersion int
}

// NewClient creates a new opencode CLI client from Atmos configuration.
func NewClient(ctx context.Context, atmosConfig *schema.AtmosConfiguration) (*Client, error) {
	defer perf.Track(atmosConfig, "opencode.NewClient")()

	config := base.ExtractConfig(atmosConfig, ProviderName, base.ProviderDefaults{
		Model: ProviderName,
	})

	if !config.Enabled {
		return nil, errUtils.ErrAIDisabledInConfiguration
	}

	providerConfig := base.GetProviderConfig(atmosConfig, ProviderName)

	client := &Client{
		model: config.Model,
	}

	applyProviderConfig(client, providerConfig)

	// Resolve binary path.
	if client.binaryPath == "" {
		resolved, err := exec.LookPath(DefaultBinary)
		if err != nil {
			return nil, errUtils.Build(errUtils.ErrCLIProviderBinaryNotFound).
				WithContext("provider", ProviderName).
				WithContext("binary", DefaultBinary).
				WithHint("Install opencode: npm install -g opencode-ai (see https://opencode.ai/docs/#install)").
				Err()
		}
		client.binaryPath = resolved
	}

	// Detect the opencode major version so invocation flags match the binary's CLI
	// generation (see buildArgs for why v2 flags matter).
	client.majorVersion = resolveMajorVersion(ctx, client.binaryPath)
	if client.majorVersion < 2 {
		// Native sessions need v2's --session/--title flags and JSONL sessionID;
		// the v1 client must not advertise the capability.
		client.nativeSessions = false
	}

	// Capture MCP servers for pass-through (only if configured). opencode reads MCP servers
	// from its config file; we hand it a temp config via OPENCODE_CONFIG at invocation time
	// (see SendMessage), so no user file is ever modified.
	if len(atmosConfig.MCP.Servers) > 0 {
		client.mcpServers = atmosConfig.MCP.Servers
		client.toolchainPATH = base.ResolveToolchainPATH(atmosConfig)
		client.hasMCPServers = true
		ui.Info(fmt.Sprintf("MCP servers configured: %d (via %s)", len(client.mcpServers), ConfigEnvVar))
	}

	return client, nil
}

// resolveMajorVersion determines the opencode major version via a --version probe,
// assuming v2 — the current major — when the probe fails: a broken probe must never
// silently drop the user's MCP servers by falling back to v1 invocation flags.
func resolveMajorVersion(ctx context.Context, binaryPath string) int {
	major, known := detectMajorVersion(ctx, binaryPath)
	if !known {
		major = 2
		log.Debug("Could not determine opencode version; assuming v2", "binary", binaryPath)
	}
	return major
}

// applyProviderConfig applies provider-specific settings to the client.
func applyProviderConfig(client *Client, providerConfig *schema.AIProviderConfig) {
	if providerConfig == nil {
		return
	}
	if providerConfig.Binary != "" {
		client.binaryPath = providerConfig.Binary
	}
	if providerConfig.Model != "" {
		client.model = providerConfig.Model
	}
	client.fullAuto = providerConfig.FullAuto
	// Native session continuation defaults to enabled; only an explicit
	// native_sessions: false opts out (nil = unset).
	client.nativeSessions = providerConfig.NativeSessions == nil || *providerConfig.NativeSessions
}

// versionRegex extracts the major version from `opencode --version` output such as
// "opencode v2.0.18" (the leading "v" is optional, e.g. a bare "2.0.18").
var versionRegex = regexp.MustCompile(`v?(\d+)\.\d+`)

// detectMajorVersion runs `<binary> --version` and returns the parsed major version and
// whether a version was successfully determined. Best-effort: a missing, failing, or
// silent --version returns (0, false) and the caller assumes the current major (v2).
func detectMajorVersion(ctx context.Context, binaryPath string) (int, bool) {
	probeCtx, cancel := context.WithTimeout(ctx, versionProbeTimeout)
	defer cancel()

	cmd := exec.CommandContext(probeCtx, binaryPath, "--version")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil && out.Len() == 0 {
		return 0, false
	}
	m := versionRegex.FindStringSubmatch(out.String())
	if m == nil {
		return 0, false
	}
	major, err := strconv.Atoi(m[1])
	if err != nil {
		return 0, false
	}
	return major, true
}

// buildArgs constructs the CLI arguments for a non-interactive `opencode run` invocation.
func (c *Client) buildArgs(message string) []string {
	// `run` executes a single prompt non-interactively and prints the assistant's response
	// to stdout. On v1 we deliberately avoid `--format json`: its JSONL stream had known
	// issues dropping the final event, whereas the default output reliably carries the
	// answer. On v2 the JSONL stream is reliable and we use it (see ExtractResultJSON).
	args := []string{"run", message}
	if c.model != "" && c.model != ProviderName {
		args = append(args, "-m", c.model)
	}
	// opencode allows tool calls (built-in and MCP) by default, so they run non-interactively
	// without any approval flag. We pass --auto ONLY when the user explicitly opts in via
	// full_auto: --auto blanket-approves every permission that isn't explicitly denied (file,
	// shell, network) and overrides any `ask` rules in the user's opencode config, so we must
	// not enable it implicitly just because MCP servers are configured. Explicit `deny` rules
	// are always still enforced by opencode even under --auto.
	if c.fullAuto {
		args = append(args, "--auto")
	}
	if c.majorVersion >= 2 {
		// v2 routes `run` through a persistent background service by default, and that
		// service resolves config in its own process — it ignores OPENCODE_CONFIG from the
		// invoking environment, which would silently drop Atmos's MCP pass-through
		// (servers, env values, auth wrappers). --standalone keeps each invocation on a
		// private in-process server so the temp config is honored, and keeps one-shot CLI
		// usage hermetic (no daemon dependency in CI/sandboxes). Verified against v2:
		// service mode ignores an OPENCODE_CONFIG override; standalone mode honors it.
		args = append(args, "--standalone")
		// v2's JSONL stream reliably carries the final text event (unlike v1's), so use
		// it for structured answer extraction instead of scraping decorated stdout.
		args = append(args, "--format", "json")
	}
	return args
}

// SendMessage sends a prompt to opencode and returns the response.
func (c *Client) SendMessage(ctx context.Context, message string) (string, error) {
	defer perf.Track(nil, "opencode.Client.SendMessage")()

	out, err := c.runCommand(ctx, c.buildArgs(message))
	if err != nil {
		return "", err
	}

	if c.majorVersion >= 2 {
		return ExtractResultJSON(out)
	}
	return ExtractResult(out)
}

// runCommand executes the opencode CLI with the given arguments and returns its
// stdout, wrapping failures with the provider name and any stderr detail. It is
// the shared subprocess path for one-shot sends and native-session turns.
func (c *Client) runCommand(ctx context.Context, args []string) ([]byte, error) {
	defer perf.Track(nil, "opencode.Client.runCommand")()

	cmd := exec.CommandContext(ctx, c.binaryPath, args...) //nolint:gosec // Binary path is from user config or exec.LookPath.
	cmd.Env = os.Environ()

	// Point opencode at a temp config containing the pass-through MCP servers, cleaned
	// up after the subprocess exits. Writing per-call keeps multi-turn sessions correct
	// and leaves no state behind. If MCP config can't be applied we fail instead of
	// silently running without the servers, env values, and auth wrappers the user
	// configured.
	cleanup, err := c.applyMCPConfig(cmd)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		stderrStr := strings.TrimSpace(stderr.String())
		if stderrStr != "" {
			return nil, fmt.Errorf("%w: %s: %s: %w", errUtils.ErrCLIProviderExecFailed, ProviderName, stderrStr, err)
		}
		return nil, fmt.Errorf("%w: %s: %w", errUtils.ErrCLIProviderExecFailed, ProviderName, err)
	}

	return stdout.Bytes(), nil
}

// SendMessageWithTools is not supported — opencode manages its own tools.
func (c *Client) SendMessageWithTools(_ context.Context, _ string, _ []tools.Tool) (*types.Response, error) {
	return nil, errUtils.ErrCLIProviderToolsNotSupported
}

// SendMessageWithHistory concatenates history into a single prompt.
func (c *Client) SendMessageWithHistory(ctx context.Context, messages []types.Message) (string, error) {
	defer perf.Track(nil, "opencode.Client.SendMessageWithHistory")()

	return c.SendMessage(ctx, base.FormatMessagesAsPrompt(messages))
}

// SendMessageWithToolsAndHistory is not supported.
func (c *Client) SendMessageWithToolsAndHistory(_ context.Context, _ []types.Message, _ []tools.Tool) (*types.Response, error) {
	return nil, errUtils.ErrCLIProviderToolsNotSupported
}

// SendMessageWithSystemPromptAndTools sends with system prompt and memory prepended.
func (c *Client) SendMessageWithSystemPromptAndTools(
	ctx context.Context,
	systemPrompt string,
	atmosMemory string,
	messages []types.Message,
	_ []tools.Tool,
) (*types.Response, error) {
	defer perf.Track(nil, "opencode.Client.SendMessageWithSystemPromptAndTools")()

	prompt := base.FormatMessagesAsPrompt(messages)
	if systemPrompt != "" {
		prompt = systemPrompt + promptSeparator + prompt
	}
	if atmosMemory != "" {
		prompt = atmosMemory + promptSeparator + prompt
	}

	result, err := c.SendMessage(ctx, prompt)
	if err != nil {
		return nil, err
	}

	return &types.Response{
		Content:    result,
		StopReason: types.StopReasonEndTurn,
	}, nil
}

// GetModel returns the configured model name.
func (c *Client) GetModel() string { return c.model }

// GetMaxTokens returns 0 — managed by opencode internally.
func (c *Client) GetMaxTokens() int { return 0 }

// ExtractResult extracts the final text response from opencode's default `run` output
// (the v1 invocation path; v2 uses ExtractResultJSON). The default output carries the
// assistant's plain-text response on stdout.
func ExtractResult(output []byte) (string, error) {
	trimmed := strings.TrimSpace(string(output))
	if trimmed == "" {
		return "", errUtils.ErrCLIProviderParseResponse
	}
	return trimmed, nil
}

// runEvent is one line of opencode v2's `run --format json` JSONL event stream. Only the
// fields Atmos needs are modeled; unknown fields are ignored by encoding/json.
type runEvent struct {
	Type      string    `json:"type"`
	SessionID string    `json:"sessionID"`
	Part      *runPart  `json:"part"`
	Error     *runError `json:"error"`
}

// runPart carries the event payload: for "text" events, the assistant text itself.
type runPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// runError is opencode's top-level error object, e.g.
// {"error":{"type":"aborted","message":"..."}}.
type runError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// applyRunEvent folds one parsed JSONL event into the answer accumulator: a
// step_start resets it (everything before was interim narration or tool work), a
// text event appends its payload, and opencode's top-level error object aborts
// extraction.
func applyRunEvent(answer []string, ev runEvent) ([]string, error) {
	if ev.Error != nil {
		return nil, fmt.Errorf("%w: %s: %s: %s", errUtils.ErrCLIProviderExecFailed, ProviderName, ev.Error.Type, ev.Error.Message)
	}
	switch ev.Type {
	case "step_start":
		answer = answer[:0]
	case "text":
		if ev.Part != nil && ev.Part.Text != "" {
			answer = append(answer, ev.Part.Text)
		}
	}
	return answer, nil
}

// runResult is the parsed outcome of a v2 `run --format json` JSONL stream: the
// final-step answer text plus the provider session ID (carried by every event).
type runResult struct {
	Text      string
	SessionID string
}

// ExtractResultJSON extracts the final assistant answer from opencode v2's
// `run --format json` JSONL event stream (see extractRunResultJSON for the exact
// semantics). It is the one-shot companion of the native-session methods.
func ExtractResultJSON(output []byte) (string, error) {
	result, err := extractRunResultJSON(output)
	return result.Text, err
}

// extractRunResultJSON parses opencode v2's `run --format json` JSONL event stream.
// The answer is the concatenation of the "text" event payloads emitted during the
// final step (after the last "step_start"): earlier steps' text is interim narration
// ("I'll read that file for you") and "tool_use" parts are never part of the answer.
// The session ID is the last one seen on any event. A top-level {"error":...} object
// fails the call. Output that is not a JSONL stream at all falls back to plain-text
// extraction, so an unexpected format degrades to the v1 contract instead of erroring.
func extractRunResultJSON(output []byte) (runResult, error) {
	var answer []string
	result := runResult{}
	parsedAny := false
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var ev runEvent
		err := json.Unmarshal([]byte(line), &ev)
		if err != nil {
			// Not JSON. If nothing parsed yet, the output is probably plain text (format
			// flag ignored or an unexpected build) — fall back to text extraction.
			if !parsedAny {
				text, textErr := ExtractResult(output)
				return runResult{Text: text}, textErr
			}
			return runResult{}, fmt.Errorf("%w: %s: %w", errUtils.ErrCLIProviderParseResponse, ProviderName, err)
		}
		parsedAny = true
		if ev.SessionID != "" {
			result.SessionID = ev.SessionID
		}
		answer, err = applyRunEvent(answer, ev)
		if err != nil {
			return runResult{}, err
		}
	}

	result.Text = strings.TrimSpace(strings.Join(answer, "\n\n"))
	if result.Text == "" {
		return runResult{}, errUtils.ErrCLIProviderParseResponse
	}
	return result, nil
}

// NativeSessionProvider implements types.NativeSessionClient: the provider-scoped
// namespace for stored session IDs, so a provider switch mid-chat never feeds an
// opencode session ID to another provider.
func (c *Client) NativeSessionProvider() string {
	return ProviderName
}

// StartNativeSession implements types.NativeSessionClient (see the interface for the
// full contract). The opening turn mirrors SendMessageWithSystemPromptAndTools's
// prompt shaping exactly, so moving a chat onto a native session changes nothing
// about what the model sees on turn one; the title names the session in
// `opencode session list` for discoverability.
func (c *Client) StartNativeSession(ctx context.Context, systemPrompt, atmosMemory string, messages []types.Message, title string) (string, string, error) {
	defer perf.Track(nil, "opencode.Client.StartNativeSession")()

	if !c.nativeSessions {
		return "", "", errUtils.ErrCLIProviderNativeSessionsOff
	}

	prompt := base.FormatMessagesAsPrompt(messages)
	if systemPrompt != "" {
		prompt = systemPrompt + promptSeparator + prompt
	}
	if atmosMemory != "" {
		prompt = atmosMemory + promptSeparator + prompt
	}

	args := c.buildArgs(prompt)
	if title != "" {
		args = append(args, "--title", title)
	}

	out, err := c.runCommand(ctx, args)
	if err != nil {
		return "", "", err
	}
	result, err := extractRunResultJSON(out)
	if err != nil {
		return "", "", err
	}
	if result.SessionID == "" {
		return "", "", fmt.Errorf("%w: %s", errUtils.ErrCLIProviderNativeSessionIDAbsent, ProviderName)
	}
	return result.Text, result.SessionID, nil
}

// ContinueNativeSession implements types.NativeSessionClient (see the interface for
// the full contract). --session continues the existing session in place — WITHOUT
// --fork, which would copy the conversation to a new session ID and leave Atmos's
// stored ID pointing at a stale branch.
func (c *Client) ContinueNativeSession(ctx context.Context, sessionID, message string) (string, error) {
	defer perf.Track(nil, "opencode.Client.ContinueNativeSession")()

	if !c.nativeSessions {
		return "", errUtils.ErrCLIProviderNativeSessionsOff
	}

	args := append(c.buildArgs(message), "--session", sessionID)
	out, err := c.runCommand(ctx, args)
	if err != nil {
		return "", err
	}
	result, err := extractRunResultJSON(out)
	if err != nil {
		return "", err
	}
	return result.Text, nil
}

// opencodeMCPServer is a single local MCP server entry in opencode's config `mcp` map.
// The tool expects the command and its arguments as a single array, environment variables
// under `environment`, and a `type` of "local" for stdio servers.
type opencodeMCPServer struct {
	Type        string            `json:"type"`
	Command     []string          `json:"command"`
	Enabled     bool              `json:"enabled"`
	Environment map[string]string `json:"environment,omitempty"`
}

// opencodeConfig is the minimal opencode config file we generate for MCP pass-through.
type opencodeConfig struct {
	Schema string                       `json:"$schema"`
	MCP    map[string]opencodeMCPServer `json:"mcp"`
}

// writeTempMCPConfig writes a temp opencode config file describing the pass-through MCP
// servers and returns its path. The caller is responsible for removing it.
func (c *Client) writeTempMCPConfig() (string, error) {
	// Generate the shared MCP config (wraps auth-requiring servers with `atmos auth exec`
	// and injects the toolchain PATH), then translate it into opencode's schema.
	shared := mcpclient.GenerateMCPConfig(c.mcpServers, c.toolchainPATH)

	cfg := opencodeConfig{
		Schema: configSchemaURL,
		MCP:    make(map[string]opencodeMCPServer, len(shared.MCPServers)),
	}
	for name, srv := range shared.MCPServers {
		command := append([]string{srv.Command}, srv.Args...)
		cfg.MCP[name] = opencodeMCPServer{
			Type:        "local",
			Command:     command,
			Enabled:     true,
			Environment: srv.Env,
		}
	}

	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", fmt.Errorf(errUtils.ErrWrapFormat, errUtils.ErrMCPConfigMarshalFailed, err)
	}

	f, err := os.CreateTemp("", "atmos-opencode-*.json")
	if err != nil {
		return "", fmt.Errorf("%w: %w", errUtils.ErrMCPConfigWriteFailed, err)
	}
	path := f.Name()

	_, writeErr := f.Write(append(out, '\n'))
	if closeErr := f.Close(); closeErr != nil && writeErr == nil {
		writeErr = closeErr
	}
	if writeErr != nil {
		removeTempConfig(path)
		return "", fmt.Errorf("%w: %s: %w", errUtils.ErrMCPConfigWriteFailed, path, writeErr)
	}

	if err := os.Chmod(path, configFilePerms); err != nil {
		log.Debug("Failed to chmod opencode MCP config", "path", path, "error", err)
	}

	return path, nil
}

// applyMCPConfig writes a temp MCP config (when servers are configured) and points the
// subprocess at it via OPENCODE_CONFIG. On v2 this only takes effect because buildArgs
// passes --standalone: the default background service resolves config in its own process
// and ignores the caller's OPENCODE_CONFIG (verified against opencode v2.0.x). It returns
// a cleanup func the caller must defer, and an error if the config could not be written —
// callers must not run opencode in that case, since the configured MCP servers, env
// values, and auth wrappers would be silently missing.
func (c *Client) applyMCPConfig(cmd *exec.Cmd) (func(), error) {
	if !c.hasMCPServers {
		return func() {}, nil
	}
	configPath, err := c.writeTempMCPConfig()
	if err != nil {
		return func() {}, err
	}
	cmd.Env = append(cmd.Env, ConfigEnvVar+"="+configPath)
	return func() { removeTempConfig(configPath) }, nil
}

// removeTempConfig removes a generated temp config file, ignoring a missing file.
func removeTempConfig(path string) {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		log.Debug("Failed to remove opencode MCP config", "path", path, "error", err)
	}
}
