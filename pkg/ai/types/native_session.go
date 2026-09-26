package types

import "context"

// NativeSessionClient is an optional capability implemented by AI clients that hold
// the conversation server-side and can resume it by ID across invocations (for
// example the opencode CLI provider on v2+, via `run --session`). Atmos chat
// type-asserts its client against this interface; providers that do not implement
// it keep the concatenated-history behavior of SendMessageWithHistory /
// SendMessageWithSystemPromptAndTools, so implementing it is purely additive.
type NativeSessionClient interface {
	// NativeSessionProvider returns the provider-scoped namespace for stored session
	// IDs (e.g. "opencode"). Atmos keys its persisted session metadata by this value
	// so that switching providers mid-chat never feeds one provider's session ID to
	// another provider's ContinueNativeSession.
	NativeSessionProvider() string

	// StartNativeSession sends the opening turn to a newly created provider-side
	// session and returns the reply plus the provider session ID the caller must
	// persist for later ContinueNativeSession calls. The system prompt, Atmos
	// memory, and any prior messages are applied exactly as
	// SendMessageWithSystemPromptAndTools would apply them, so a first native turn
	// is equivalent to a first legacy turn. The title, when non-empty, names the
	// session in the provider's own session list for discoverability.
	StartNativeSession(ctx context.Context, systemPrompt, atmosMemory string, messages []Message, title string) (string, string, error)

	// ContinueNativeSession sends a follow-up turn to an existing provider-side
	// session and returns the reply. The message is the new user input only --
	// prior context lives in the provider-side session, which is the point of the
	// capability: Atmos stops re-serializing the full history every turn.
	ContinueNativeSession(ctx context.Context, sessionID, message string) (string, error)
}
