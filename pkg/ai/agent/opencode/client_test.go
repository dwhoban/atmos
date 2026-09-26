package opencode

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errUtils "github.com/cloudposse/atmos/errors"
	"github.com/cloudposse/atmos/pkg/ai/types"
	"github.com/cloudposse/atmos/pkg/schema"
)

func TestConstants(t *testing.T) {
	assert.Equal(t, "opencode", ProviderName)
	assert.Equal(t, "opencode", DefaultBinary)
	assert.Equal(t, "OPENCODE_CONFIG", ConfigEnvVar)
}

func TestNewClient_Disabled(t *testing.T) {
	_, err := NewClient(t.Context(), &schema.AtmosConfiguration{AI: schema.AISettings{Enabled: false}})
	assert.ErrorIs(t, err, errUtils.ErrAIDisabledInConfiguration)
}

func TestNewClient_BinaryNotOnPath(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	_, err := NewClient(t.Context(), &schema.AtmosConfiguration{
		AI: schema.AISettings{
			Enabled:   true,
			Providers: map[string]*schema.AIProviderConfig{ProviderName: {}},
		},
	})
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderBinaryNotFound)
}

func TestNewClient_CustomBinaryAndModel(t *testing.T) {
	client, err := NewClient(t.Context(), &schema.AtmosConfiguration{
		AI: schema.AISettings{
			Enabled: true,
			Providers: map[string]*schema.AIProviderConfig{
				ProviderName: {Binary: "/usr/local/bin/opencode", Model: "anthropic/claude-sonnet-4-5", FullAuto: true},
			},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, "/usr/local/bin/opencode", client.binaryPath)
	assert.Equal(t, "anthropic/claude-sonnet-4-5", client.model)
	assert.True(t, client.fullAuto)
	assert.False(t, client.hasMCPServers)
}

func TestNewClient_DefaultModelIsProviderName(t *testing.T) {
	client, err := NewClient(t.Context(), &schema.AtmosConfiguration{
		AI: schema.AISettings{
			Enabled:   true,
			Providers: map[string]*schema.AIProviderConfig{ProviderName: {Binary: "/usr/local/bin/opencode"}},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, ProviderName, client.GetModel())
	assert.Equal(t, 0, client.GetMaxTokens())
}

func TestNewClient_MCPServersCapturedWhenConfigured(t *testing.T) {
	client, err := NewClient(t.Context(), &schema.AtmosConfiguration{
		AI: schema.AISettings{
			Enabled:   true,
			Providers: map[string]*schema.AIProviderConfig{ProviderName: {Binary: "/usr/local/bin/opencode"}},
		},
		MCP: schema.MCPSettings{
			Servers: map[string]schema.MCPServerConfig{
				"aws-docs": {Command: "uvx", Args: []string{"docs@latest"}},
			},
		},
	})
	require.NoError(t, err)
	assert.Len(t, client.mcpServers, 1)
	assert.True(t, client.hasMCPServers)
}

// fakeProviderConfig builds a minimal enabled provider config pointing at the fake test
// binary (see testmain_test.go), which answers `--version` from fakeVersionEnv.
func fakeProviderConfig(t *testing.T) *schema.AtmosConfiguration {
	t.Helper()
	return &schema.AtmosConfiguration{
		AI: schema.AISettings{
			Enabled:   true,
			Providers: map[string]*schema.AIProviderConfig{ProviderName: {Binary: testExecutable(t)}},
		},
	}
}

// TestNewClient_DetectsMajorVersion verifies NewClient wires the --version probe
// through: the fake binary defaults to a v2 string, and a v1 string keeps the v1
// invocation path.
func TestNewClient_DetectsMajorVersion(t *testing.T) {
	client, err := NewClient(t.Context(), fakeProviderConfig(t))
	require.NoError(t, err)
	assert.Equal(t, 2, client.majorVersion)

	t.Setenv(fakeVersionEnv, "opencode v1.0.243")
	client, err = NewClient(t.Context(), fakeProviderConfig(t))
	require.NoError(t, err)
	assert.Equal(t, 1, client.majorVersion)
}

// TestNewClient_AssumesV2WhenVersionUnknown verifies the assume-v2 fallback: a binary
// whose --version produces nothing parseable must still get v2 invocation flags, or MCP
// pass-through would be silently dropped on real v2 installs (the background service
// ignores OPENCODE_CONFIG unless --standalone is passed).
func TestNewClient_AssumesV2WhenVersionUnknown(t *testing.T) {
	t.Setenv(fakeFailEnv, "1")
	client, err := NewClient(t.Context(), fakeProviderConfig(t))
	require.NoError(t, err)
	assert.Equal(t, 2, client.majorVersion)
}

// TestDetectMajorVersion exercises the --version output parser directly.
func TestDetectMajorVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
		want    int
		wantOK  bool
	}{
		{name: "v2 with v prefix", version: "opencode v2.0.18\n", want: 2, wantOK: true},
		{name: "v1 with v prefix", version: "opencode v1.0.243\n", want: 1, wantOK: true},
		{name: "bare semver", version: "2.3.1\n", want: 2, wantOK: true},
		{name: "no version number", version: "opencode\n", want: 0, wantOK: false},
		{name: "probe fails with no output", version: "", want: 0, wantOK: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(fakeVersionEnv, tt.version)
			// An empty canned version makes the fake --version exit non-zero with no
			// output, covering the "nothing to parse" probe branch.
			if tt.version == "" {
				t.Setenv(fakeFailEnv, "1")
			}
			got, ok := detectMajorVersion(t.Context(), testExecutable(t))
			assert.Equal(t, tt.wantOK, ok)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestBuildArgs(t *testing.T) {
	tests := []struct {
		name          string
		model         string
		fullAuto      bool
		hasMCPServers bool
		majorVersion  int
		want          []string
	}{
		{
			name: "default model omits -m and no --auto",
			want: []string{"run", "hello"},
		},
		{
			name:  "custom model adds -m",
			model: "openai/gpt-4o",
			want:  []string{"run", "hello", "-m", "openai/gpt-4o"},
		},
		{
			name:     "full_auto adds --auto",
			fullAuto: true,
			want:     []string{"run", "hello", "--auto"},
		},
		{
			// MCP servers alone must NOT enable --auto: opencode allows tools by default, and
			// --auto would override the user's `ask` permission rules (see buildArgs comment).
			name:          "MCP servers alone do not add --auto",
			hasMCPServers: true,
			want:          []string{"run", "hello"},
		},
		{
			// v2 adds --standalone (mandatory for OPENCODE_CONFIG/MCP pass-through to be
			// honored; the background service resolves config in its own process) and
			// --format json (structured extraction; see ExtractResultJSON).
			name:         "v2 adds --standalone and --format json",
			majorVersion: 2,
			want:         []string{"run", "hello", "--standalone", "--format", "json"},
		},
		{
			name:         "v2 keeps model (including #variant slug), --auto, and v2 flags",
			model:        "anthropic/claude-sonnet-4-5#thinking",
			fullAuto:     true,
			majorVersion: 2,
			want:         []string{"run", "hello", "-m", "anthropic/claude-sonnet-4-5#thinking", "--auto", "--standalone", "--format", "json"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Client{model: tt.model, fullAuto: tt.fullAuto, hasMCPServers: tt.hasMCPServers, majorVersion: tt.majorVersion}
			if tt.model == "" {
				c.model = ProviderName
			}
			assert.Equal(t, tt.want, c.buildArgs("hello"))
		})
	}
}

func TestExtractResult(t *testing.T) {
	got, err := ExtractResult([]byte("  the answer  \n"))
	require.NoError(t, err)
	assert.Equal(t, "the answer", got)

	_, err = ExtractResult([]byte("   \n"))
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderParseResponse)
}

func TestExtractResultJSON(t *testing.T) {
	// Multi-step stream: interim narration and tool_use must be excluded; the answer is
	// the text emitted during the final step (after the last step_start).
	multiStep := strings.Join([]string{
		`{"type":"step_start","sessionID":"s1","part":{"type":"step-start","sessionID":"s1"}}`,
		`{"type":"text","sessionID":"s1","part":{"type":"text","text":"I'll read that file for you."}}`,
		`{"type":"tool_use","sessionID":"s1","part":{"type":"tool_use","tool":"read","state":{"status":"completed"}}}`,
		`{"type":"step_finish","sessionID":"s1","part":{"type":"step-finish"}}`,
		`{"type":"step_start","sessionID":"s1","part":{"type":"step-start"}}`,
		`{"type":"text","sessionID":"s1","part":{"type":"text","text":"hello"}}`,
		`{"type":"step_finish","sessionID":"s1","part":{"type":"step-finish"}}`,
	}, "\n")
	got, err := ExtractResultJSON([]byte(multiStep))
	require.NoError(t, err)
	assert.Equal(t, "hello", got)

	// Plain text (no JSON at all) falls back to ExtractResult semantics.
	got, err = ExtractResultJSON([]byte("  plain answer \n"))
	require.NoError(t, err)
	assert.Equal(t, "plain answer", got)

	// A top-level {"error":...} object fails the call.
	_, err = ExtractResultJSON([]byte(`{"error":{"type":"aborted","message":"shut down"}}`))
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderExecFailed)

	// Blank output errors.
	_, err = ExtractResultJSON([]byte("\n \n"))
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderParseResponse)

	// Events that parse but never yield text error.
	_, err = ExtractResultJSON([]byte(`{"type":"step_start","part":{}}`))
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderParseResponse)

	// Garbage after valid JSONL errors rather than being returned as the answer.
	_, err = ExtractResultJSON([]byte("{\"type\":\"text\",\"part\":{\"type\":\"text\",\"text\":\"hi\"}}\nnot json"))
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderParseResponse)
}

// TestWriteTempMCPConfig verifies the generated file matches opencode's config schema:
// an `mcp` map whose entries carry type "local", a combined command array, and environment.
func TestWriteTempMCPConfig(t *testing.T) {
	c := &Client{
		toolchainPATH: "/opt/atmos/toolchain/bin",
		mcpServers: map[string]schema.MCPServerConfig{
			"aws-docs": {
				Command: "uvx",
				Args:    []string{"awslabs.aws-documentation-mcp-server@latest"},
				Env:     map[string]string{"AWS_REGION": "us-east-1"},
			},
		},
	}
	path, err := c.writeTempMCPConfig()
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.Remove(path) })

	assert.True(t, strings.HasSuffix(path, ".json"))

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var cfg struct {
		Schema string `json:"$schema"`
		MCP    map[string]struct {
			Type        string            `json:"type"`
			Command     []string          `json:"command"`
			Enabled     bool              `json:"enabled"`
			Environment map[string]string `json:"environment"`
		} `json:"mcp"`
	}
	require.NoError(t, json.Unmarshal(data, &cfg))
	assert.Equal(t, configSchemaURL, cfg.Schema)

	srv, ok := cfg.MCP["aws-docs"]
	require.True(t, ok, "aws-docs server must be present")
	assert.Equal(t, "local", srv.Type)
	assert.True(t, srv.Enabled)
	// Command must be a single array combining the command and its arguments.
	require.NotEmpty(t, srv.Command)
	assert.Equal(t, "uvx", srv.Command[0])
	assert.Equal(t, "awslabs.aws-documentation-mcp-server@latest", srv.Command[len(srv.Command)-1])
	// The generated environment must carry the server's own env plus the injected toolchain
	// PATH, or credential- and toolchain-dependent MCP servers would fail at runtime.
	require.NotNil(t, srv.Environment)
	assert.Equal(t, "us-east-1", srv.Environment["AWS_REGION"])
	assert.Contains(t, srv.Environment["PATH"], "/opt/atmos/toolchain/bin")
}

// TestWriteTempMCPConfig_AuthWrapped confirms servers with an identity are wrapped with
// `atmos auth exec` in the combined command array.
func TestWriteTempMCPConfig_AuthWrapped(t *testing.T) {
	c := &Client{
		mcpServers: map[string]schema.MCPServerConfig{
			"aws-billing": {Command: "uvx", Args: []string{"billing@latest"}, Identity: "readonly"},
		},
	}
	path, err := c.writeTempMCPConfig()
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.Remove(path) })

	data, err := os.ReadFile(path)
	require.NoError(t, err)

	var cfg struct {
		MCP map[string]struct {
			Command []string `json:"command"`
		} `json:"mcp"`
	}
	require.NoError(t, json.Unmarshal(data, &cfg))
	cmd := cfg.MCP["aws-billing"].Command
	require.NotEmpty(t, cmd)
	assert.Equal(t, "atmos", cmd[0])
	assert.Contains(t, cmd, "auth")
	assert.Contains(t, cmd, "exec")
	assert.Contains(t, cmd, "readonly")
}

func TestSendMessageWithTools_NotSupported(t *testing.T) {
	c := &Client{}
	_, err := c.SendMessageWithTools(t.Context(), "x", nil)
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderToolsNotSupported)

	_, err = c.SendMessageWithToolsAndHistory(t.Context(), nil, nil)
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderToolsNotSupported)
}

// testExecutable returns the path to the running test binary, which impersonates the
// opencode CLI when a fake-binary gate env var is set (see testmain_test.go).
func testExecutable(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	require.NoError(t, err)
	return exe
}

func TestSendMessage_Success(t *testing.T) {
	t.Setenv(fakeStdoutEnv, "the opencode answer\n")
	c := &Client{binaryPath: testExecutable(t), model: ProviderName}
	out, err := c.SendMessage(t.Context(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "the opencode answer", out)
}

// TestSendMessage_V2JSON exercises the full v2 path end-to-end through the fake binary:
// the --standalone/--format json args are built, the subprocess output is treated as a
// JSONL event stream, and the final-step text is extracted.
func TestSendMessage_V2JSON(t *testing.T) {
	t.Setenv(fakeStdoutEnv, strings.Join([]string{
		`{"type":"step_start","part":{"type":"step-start"}}`,
		`{"type":"text","part":{"type":"text","text":"the v2 answer"}}`,
		`{"type":"step_finish","part":{"type":"step-finish"}}`,
	}, "\n")+"\n")
	c := &Client{binaryPath: testExecutable(t), model: ProviderName, majorVersion: 2}
	out, err := c.SendMessage(t.Context(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "the v2 answer", out)
}

func TestSendMessage_ExecError(t *testing.T) {
	t.Setenv(fakeFailEnv, "1")
	c := &Client{binaryPath: testExecutable(t), model: ProviderName}
	_, err := c.SendMessage(t.Context(), "hi")
	assert.ErrorIs(t, err, errUtils.ErrCLIProviderExecFailed)
}

// TestSendMessage_WithMCP exercises the temp-config pass-through path: SendMessage writes
// a temp opencode config, points OPENCODE_CONFIG at it, runs, and cleans it up.
func TestSendMessage_WithMCP(t *testing.T) {
	t.Setenv(fakeStdoutEnv, "ok\n")
	c := &Client{
		binaryPath:    testExecutable(t),
		model:         ProviderName,
		hasMCPServers: true,
		mcpServers: map[string]schema.MCPServerConfig{
			"aws-docs": {Command: "uvx", Args: []string{"docs@latest"}},
		},
	}
	out, err := c.SendMessage(t.Context(), "hi")
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
}

// TestSendMessage_MCPConfigError verifies that when the MCP config can't be written, SendMessage
// fails (and never runs opencode) instead of silently proceeding without the configured servers.
func TestSendMessage_MCPConfigError(t *testing.T) {
	// Force os.CreateTemp to fail by pointing every temp-dir env var at a nonexistent directory
	// (TMPDIR on Unix, TMP/TEMP on Windows).
	bogus := filepath.Join(t.TempDir(), "does-not-exist")
	t.Setenv("TMPDIR", bogus)
	t.Setenv("TMP", bogus)
	t.Setenv("TEMP", bogus)

	// If the subprocess somehow ran, this canned stdout would make it look successful — so a
	// failure here proves we bailed out before cmd.Run().
	t.Setenv(fakeStdoutEnv, "should-not-be-returned\n")

	c := &Client{
		binaryPath:    testExecutable(t),
		model:         ProviderName,
		hasMCPServers: true,
		mcpServers:    map[string]schema.MCPServerConfig{"aws-docs": {Command: "uvx", Args: []string{"docs@latest"}}},
	}
	out, err := c.SendMessage(t.Context(), "hi")
	require.Error(t, err)
	assert.ErrorIs(t, err, errUtils.ErrMCPConfigWriteFailed)
	assert.Empty(t, out)
}

func TestSendMessageWithHistory(t *testing.T) {
	t.Setenv(fakeStdoutEnv, "history-resp\n")
	c := &Client{binaryPath: testExecutable(t), model: ProviderName}
	out, err := c.SendMessageWithHistory(t.Context(), []types.Message{{Role: types.RoleUser, Content: "hi"}})
	require.NoError(t, err)
	assert.Equal(t, "history-resp", out)
}

func TestSendMessageWithSystemPromptAndTools(t *testing.T) {
	t.Setenv(fakeStdoutEnv, "system-resp\n")
	c := &Client{binaryPath: testExecutable(t), model: ProviderName}
	resp, err := c.SendMessageWithSystemPromptAndTools(
		t.Context(), "system", "memory",
		[]types.Message{{Role: types.RoleUser, Content: "hi"}}, nil,
	)
	require.NoError(t, err)
	assert.Equal(t, "system-resp", resp.Content)
	assert.Equal(t, types.StopReasonEndTurn, resp.StopReason)
}
