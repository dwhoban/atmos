package opencode

import (
	"os"
	"strings"
	"testing"
)

// Fake-binary gate env vars. When set, the test binary impersonates the `opencode`
// CLI so SendMessage can be exercised cross-platform without a real binary or a
// platform-specific shell (see CLAUDE.md "Subprocess helpers in tests").
const (
	fakeStdoutEnv  = "_ATMOS_OPENCODE_FAKE_STDOUT"
	fakeFailEnv    = "_ATMOS_OPENCODE_FAKE_FAIL"
	fakeVersionEnv = "_ATMOS_OPENCODE_FAKE_VERSION"
	fakeArgsEnv    = "_ATMOS_OPENCODE_FAKE_ARGS_FILE"
)

// defaultFakeVersion is what the fake binary answers for `--version` when
// fakeVersionEnv is unset; it mirrors the current opencode v2 output format.
const defaultFakeVersion = "opencode v2.0.18"

// TestMain lets the test binary act as a fake `opencode` when a gate env var is set:
// it writes the canned stdout and/or exits non-zero, then returns before running tests.
func TestMain(m *testing.M) {
	// Impersonate `opencode --version` for the NewClient version probe: print the canned
	// version (defaulting to a v2 string) and exit before the run-mode gate env vars.
	// With fakeFailEnv set, --version exits non-zero with no output, simulating a binary
	// whose version probe can't be parsed (NewClient must then assume v2).
	if len(os.Args) > 1 && os.Args[1] == "--version" {
		if os.Getenv(fakeFailEnv) == "1" {
			os.Exit(1)
		}
		out := os.Getenv(fakeVersionEnv)
		if out == "" {
			out = defaultFakeVersion
		}
		_, _ = os.Stdout.WriteString(out + "\n")
		os.Exit(0)
	}
	if out := os.Getenv(fakeStdoutEnv); out != "" {
		// Record argv (minus the binary) so tests can assert on invocation flags like
		// --session/--title. Placed after the --version branch so the version probe
		// never clobbers the recorded run invocation.
		if argsFile := os.Getenv(fakeArgsEnv); argsFile != "" {
			// NUL-separated so multi-line prompts stay a single recordable argument.
			_ = os.WriteFile(argsFile, []byte(strings.Join(os.Args[1:], "\x00")+"\x00"), 0o600)
		}
		_, _ = os.Stdout.WriteString(out)
		if os.Getenv(fakeFailEnv) == "1" {
			os.Exit(1)
		}
		os.Exit(0)
	}
	if os.Getenv(fakeFailEnv) == "1" {
		_, _ = os.Stderr.WriteString("opencode: simulated failure")
		os.Exit(1)
	}
	os.Exit(m.Run())
}
