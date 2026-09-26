package tui

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	errUtils "github.com/cloudposse/atmos/errors"
	"github.com/cloudposse/atmos/pkg/ai"
	"github.com/cloudposse/atmos/pkg/ai/session"
	aiTypes "github.com/cloudposse/atmos/pkg/ai/types"
)

// mockNativeSessionClient implements both ai.Client and types.NativeSessionClient,
// recording native calls so tests can assert what Atmos actually sent.
type mockNativeSessionClient struct {
	mockAIClient

	providerName string

	startCalls    int
	continueCalls int

	lastStartPrompt     string
	lastStartMemory     string
	lastStartMessages   []aiTypes.Message
	lastStartTitle      string
	lastContinueID      string
	lastContinueMessage string

	startReply     string
	startSessionID string
	continueReply  string

	startErr    error
	continueErr error
}

func (m *mockNativeSessionClient) NativeSessionProvider() string {
	if m.providerName == "" {
		return "mock-native"
	}
	return m.providerName
}

func (m *mockNativeSessionClient) StartNativeSession(_ context.Context, systemPrompt, atmosMemory string, messages []aiTypes.Message, title string) (string, string, error) {
	m.startCalls++
	m.lastStartPrompt = systemPrompt
	m.lastStartMemory = atmosMemory
	m.lastStartMessages = messages
	m.lastStartTitle = title
	if m.startErr != nil {
		return "", "", m.startErr
	}
	return m.startReply, m.startSessionID, nil
}

func (m *mockNativeSessionClient) ContinueNativeSession(_ context.Context, sessionID, message string) (string, error) {
	m.continueCalls++
	m.lastContinueID = sessionID
	m.lastContinueMessage = message
	if m.continueErr != nil {
		return "", m.continueErr
	}
	return m.continueReply, nil
}

// newNativeSessionModel returns a bare ChatModel wired only with the given client,
// enough to drive tryNativeSessionResponse without the full TUI.
func newNativeSessionModel(client ai.Client) *ChatModel {
	return &ChatModel{client: client}
}

func TestTryNativeSessionResponse_NonNativeClient(t *testing.T) {
	m := newNativeSessionModel(&mockAIClient{})

	msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
	assert.False(t, handled)
	assert.Nil(t, msg)
	assert.False(t, m.nativeSessionDisabled)
}

func TestTryNativeSessionResponse_FirstTurnStarts(t *testing.T) {
	mock := &mockNativeSessionClient{startReply: "native reply", startSessionID: "ses_1"}
	m := newNativeSessionModel(mock)
	m.sess = &session.Session{Name: "my-chat"}
	messages := []aiTypes.Message{{Role: aiTypes.RoleUser, Content: "hello"}}

	msg, handled := m.tryNativeSessionResponse(t.Context(), "hello", messages)
	require.True(t, handled)
	require.IsType(t, aiResponseMsg{}, msg)
	assert.Equal(t, "native reply", msg.(aiResponseMsg).content)

	assert.Equal(t, 1, mock.startCalls)
	assert.Equal(t, 0, mock.continueCalls)
	// The opening turn carries the full context, exactly like the legacy path.
	assert.Equal(t, defaultSystemPrompt, mock.lastStartPrompt)
	assert.Equal(t, "my-chat", mock.lastStartTitle)
	require.Len(t, mock.lastStartMessages, 1)
	assert.Equal(t, "hello", mock.lastStartMessages[0].Content)

	// The provider session ID is remembered in memory (no manager/sess storage here
	// beyond the bare session, whose metadata update is skipped without a manager).
	assert.Equal(t, "ses_1", m.nativeSessionID)
	assert.Equal(t, nativeSessionMetadataKey(mock.NativeSessionProvider()), m.nativeSessionKey)
}

func TestTryNativeSessionResponse_SecondTurnContinuesWithOnlyNewMessage(t *testing.T) {
	mock := &mockNativeSessionClient{continueReply: "follow-up"}
	m := newNativeSessionModel(mock)
	key := nativeSessionMetadataKey(mock.NativeSessionProvider())
	m.nativeSessionKey = key
	m.nativeSessionID = "ses_1"
	m.nativeSessionSystemPrompt = m.buildSystemPrompt()
	m.nativeSessionAtmosMemory = m.getAtmosMemory()

	msg, handled := m.tryNativeSessionResponse(t.Context(), "next question", nil)
	require.True(t, handled)
	require.IsType(t, aiResponseMsg{}, msg)
	assert.Equal(t, "follow-up", msg.(aiResponseMsg).content)

	// Continuation sends only the new user message — the whole point of the capability.
	assert.Equal(t, 1, mock.continueCalls)
	assert.Equal(t, 0, mock.startCalls)
	assert.Equal(t, "ses_1", mock.lastContinueID)
	assert.Equal(t, "next question", mock.lastContinueMessage)
}

func TestTryNativeSessionResponse_InstructionsChangedRestarts(t *testing.T) {
	mock := &mockNativeSessionClient{startReply: "restarted", startSessionID: "ses_2"}
	m := newNativeSessionModel(mock)
	m.nativeSessionKey = nativeSessionMetadataKey(mock.NativeSessionProvider())
	m.nativeSessionID = "ses_1"
	// A stale stored system prompt (skill switched, etc.) must restart the provider
	// session so the new instructions actually reach the model.
	m.nativeSessionSystemPrompt = "old instructions"
	m.nativeSessionAtmosMemory = m.getAtmosMemory()

	messages := []aiTypes.Message{
		{Role: aiTypes.RoleUser, Content: "one"},
		{Role: aiTypes.RoleAssistant, Content: "two"},
		{Role: aiTypes.RoleUser, Content: "three"},
	}
	msg, handled := m.tryNativeSessionResponse(t.Context(), "three", messages)
	require.True(t, handled)
	assert.Equal(t, 0, mock.continueCalls)
	assert.Equal(t, 1, mock.startCalls)
	// Restart re-sends the full history plus the new turn.
	require.Len(t, mock.lastStartMessages, 3)
	assert.Equal(t, "restarted", msg.(aiResponseMsg).content)
	assert.Equal(t, "ses_2", m.nativeSessionID)
}

func TestTryNativeSessionResponse_StaleSessionIDRestarts(t *testing.T) {
	mock := &mockNativeSessionClient{
		continueErr:    errors.New("session not found"),
		startReply:     "fresh reply",
		startSessionID: "ses_2",
	}
	m := newNativeSessionModel(mock)
	m.nativeSessionKey = nativeSessionMetadataKey(mock.NativeSessionProvider())
	m.nativeSessionID = "ses_gone"
	m.nativeSessionSystemPrompt = m.buildSystemPrompt()
	m.nativeSessionAtmosMemory = m.getAtmosMemory()

	msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
	require.True(t, handled)
	assert.Equal(t, 1, mock.continueCalls)
	assert.Equal(t, 1, mock.startCalls)
	assert.Equal(t, "fresh reply", msg.(aiResponseMsg).content)
	assert.Equal(t, "ses_2", m.nativeSessionID)
}

func TestTryNativeSessionResponse_StartFailureFallsBack(t *testing.T) {
	t.Run("generic start error falls back without disabling", func(t *testing.T) {
		mock := &mockNativeSessionClient{startErr: errors.New("boom")}
		m := newNativeSessionModel(mock)

		msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
		assert.False(t, handled)
		assert.Nil(t, msg)
		assert.False(t, m.nativeSessionDisabled, "transient failures must not permanently disable native sessions")
	})

	t.Run("capability-off sentinel disables for the run", func(t *testing.T) {
		mock := &mockNativeSessionClient{startErr: errUtils.ErrCLIProviderNativeSessionsOff}
		m := newNativeSessionModel(mock)

		msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
		assert.False(t, handled)
		assert.Nil(t, msg)
		assert.True(t, m.nativeSessionDisabled)
	})

	t.Run("capability-off sentinel on continue disables for the run", func(t *testing.T) {
		mock := &mockNativeSessionClient{continueErr: errUtils.ErrCLIProviderNativeSessionsOff}
		m := newNativeSessionModel(mock)
		m.nativeSessionKey = nativeSessionMetadataKey(mock.NativeSessionProvider())
		m.nativeSessionID = "ses_1"
		m.nativeSessionSystemPrompt = m.buildSystemPrompt()
		m.nativeSessionAtmosMemory = m.getAtmosMemory()

		msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
		assert.False(t, handled)
		assert.Nil(t, msg)
		assert.True(t, m.nativeSessionDisabled)
	})

	t.Run("disabled model skips native entirely", func(t *testing.T) {
		mock := &mockNativeSessionClient{startReply: "never"}
		m := newNativeSessionModel(mock)
		m.nativeSessionDisabled = true

		msg, handled := m.tryNativeSessionResponse(t.Context(), "hi", nil)
		assert.False(t, handled)
		assert.Nil(t, msg)
		assert.Equal(t, 0, mock.startCalls)
	})
}

func TestTryNativeSessionResponse_PersistsMetadata(t *testing.T) {
	storage, err := session.NewSQLiteStorage(filepath.Join(t.TempDir(), "sessions.db"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = storage.Close() })
	manager := session.NewManager(storage, t.TempDir(), 10, nil)

	sess, err := manager.CreateSession(t.Context(), session.CreateSessionParams{Name: "chat", Provider: "opencode"})
	require.NoError(t, err)

	mock := &mockNativeSessionClient{startReply: "reply", startSessionID: "ses_persist", providerName: "opencode"}
	m := newNativeSessionModel(mock)
	m.manager = manager
	m.sess = sess

	_, handled := m.tryNativeSessionResponse(t.Context(), "hello", []aiTypes.Message{{Role: aiTypes.RoleUser, Content: "hello"}})
	require.True(t, handled)

	key := nativeSessionMetadataKey("opencode")
	assert.Equal(t, "ses_persist", sess.Metadata[key])

	// Round-trip through storage to prove the metadata (and thus the provider session
	// ID) survives an Atmos restart.
	reloaded, err := manager.GetSession(t.Context(), sess.ID)
	require.NoError(t, err)
	assert.Equal(t, "ses_persist", reloaded.Metadata[key])
}

func TestNativeSessionMetadataKey(t *testing.T) {
	assert.Equal(t, "native_session:opencode", nativeSessionMetadataKey("opencode"))
}
