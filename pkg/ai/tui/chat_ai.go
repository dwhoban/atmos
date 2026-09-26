package tui

import (
	"context"
	"errors"
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	errUtils "github.com/cloudposse/atmos/errors"
	"github.com/cloudposse/atmos/pkg/ai/tools"
	aiTypes "github.com/cloudposse/atmos/pkg/ai/types"
	log "github.com/cloudposse/atmos/pkg/logger"
)

// statusMsg updates the loading status text.
type statusMsg string

// aiRequestCtx bundles the common parameters for AI request handling.
type aiRequestCtx struct {
	ctx                context.Context
	messages           []aiTypes.Message
	availableTools     []tools.Tool
	systemPrompt       string
	atmosMemory        string
	accumulatedContent string
	resultText         string
}

func (m *ChatModel) sendMessage(content string) tea.Cmd {
	return func() tea.Msg {
		return sendMessageMsg(content)
	}
}

// defaultSystemPrompt is the default system prompt used when no skill-specific prompt is configured.
const defaultSystemPrompt = `You are an AI assistant for Atmos infrastructure management. You have access to tools that allow you to perform actions.

IMPORTANT: When you need to perform an action (read files, edit files, search, execute commands, etc.), you MUST use the available tools. Do NOT just describe what you would do - actually use the tools to do it.

For example:
- If you need to read a file, use the read_file tool immediately
- If you need to edit a file, use the edit_file tool immediately
- If you need to search for files, use the search_files tool immediately
- If you need to execute an Atmos command, use the execute_atmos_command tool immediately

Always take action using tools rather than describing what action you would take.

If atmos_list_stacks returns zero stacks, this is a new Atmos project that doesn't have any
stacks written yet. Treat this as an opportunity, not an error: proactively offer to help the
user create their first stack and component rather than just reporting that none exist.`

// buildFilteredMessages builds message history filtered by the current provider.
// Only includes messages from the current provider session for complete isolation.
func (m *ChatModel) buildFilteredMessages() []aiTypes.Message {
	currentProvider := m.getCurrentProvider()

	messages := make([]aiTypes.Message, 0, len(m.messages)+1)
	for _, msg := range m.messages {
		// Skip system messages (UI-only notifications).
		if msg.Role == roleSystem {
			continue
		}

		// Include only messages from the current provider session.
		// This provides complete conversation isolation when switching providers.
		if msg.Provider == currentProvider {
			messages = append(messages, aiTypes.Message{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	return messages
}

// applyHistoryLimits applies sliding window limits (message-based and token-based) to conversation history.
// This helps prevent rate limiting and reduces token usage for long conversations.
func (m *ChatModel) applyHistoryLimits(messages []aiTypes.Message) []aiTypes.Message {
	pruneIndex := 0 // Start of messages to keep (0 = keep all).

	// Apply message-based limit if configured.
	if m.maxHistoryMessages > 0 && len(messages) > m.maxHistoryMessages {
		pruneIndex = len(messages) - m.maxHistoryMessages
	}

	// Apply token-based limit if configured.
	// Count backwards from most recent message and stop when token limit is exceeded.
	if m.maxHistoryTokens > 0 {
		totalTokens := 0
		tokenPruneIndex := len(messages)

		// Count backwards from most recent.
		for i := len(messages) - 1; i >= 0; i-- {
			msgTokens := estimateTokens(messages[i].Content)
			if totalTokens+msgTokens > m.maxHistoryTokens {
				tokenPruneIndex = i + 1 // Keep from i+1 onwards.
				break
			}
			totalTokens += msgTokens
		}

		// Use whichever prune index is more restrictive (further right/more pruning).
		if tokenPruneIndex > pruneIndex {
			pruneIndex = tokenPruneIndex
		}
	}

	// Apply the pruning if needed.
	if pruneIndex > 0 && pruneIndex < len(messages) {
		messages = messages[pruneIndex:]
	}

	return messages
}

// prependMemoryContext prepends a system message with ATMOS.md context if available.
func (m *ChatModel) prependMemoryContext(messages []aiTypes.Message) []aiTypes.Message {
	if m.memoryMgr == nil {
		return messages
	}

	memoryContext := m.memoryMgr.GetContext()
	if memoryContext == "" {
		return messages
	}

	return append([]aiTypes.Message{{
		Role:    aiTypes.RoleSystem,
		Content: memoryContext,
	}}, messages...)
}

// buildSystemPrompt constructs the system prompt from the current skill and skill registry.
func (m *ChatModel) buildSystemPrompt() string {
	systemPrompt := defaultSystemPrompt

	if m.currentSkill != nil && m.currentSkill.SystemPrompt != "" {
		systemPrompt = m.currentSkill.SystemPrompt
	}

	// Append available skills XML to system prompt (Agent Skills integration guide).
	// This helps the model understand what skills are available and their purposes.
	if m.skillRegistry != nil {
		currentSkillName := ""
		if m.currentSkill != nil {
			currentSkillName = m.currentSkill.Name
		}
		skillsXML := m.skillRegistry.ToPromptXML(currentSkillName)
		if skillsXML != "" {
			systemPrompt = systemPrompt + doubleNewline + skillsXML
		}
	}

	return systemPrompt
}

// getAtmosMemory returns the ATMOS.md content for prompt caching.
func (m *ChatModel) getAtmosMemory() string {
	if m.memoryMgr == nil {
		return ""
	}
	return m.memoryMgr.GetContext()
}

// handleNoToolsResponse handles the AI response path when no tools are available.
func (m *ChatModel) handleNoToolsResponse(ctx context.Context, messages []aiTypes.Message) tea.Msg {
	response, err := m.sendAIRequestNoTools(ctx, messages)
	if err != nil {
		return aiErrorMsg(formatAPIError(err))
	}

	return aiResponseMsg{content: response, usage: nil}
}

// handleActionIntentRetry prompts the AI to use tools when it expressed intent but did not.
func (m *ChatModel) handleActionIntentRetry(reqCtx *aiRequestCtx, response *aiTypes.Response) tea.Msg {
	reqCtx.messages = append(reqCtx.messages, aiTypes.Message{
		Role:    aiTypes.RoleAssistant,
		Content: response.Content,
	})
	reqCtx.messages = append(reqCtx.messages, aiTypes.Message{
		Role:    aiTypes.RoleUser,
		Content: "Please use the available tools to perform that action now, rather than just describing what you would do.",
	})

	// Send the prompt again with caching.
	retryResponse, err := m.sendAIRequest(reqCtx.ctx, reqCtx.systemPrompt, reqCtx.atmosMemory, reqCtx.messages, reqCtx.availableTools)
	if err != nil {
		return aiErrorMsg(formatAPIError(err))
	}

	// Check if AI now uses tools.
	if retryResponse.StopReason == aiTypes.StopReasonToolUse && len(retryResponse.ToolCalls) > 0 {
		return m.handleToolExecutionFlow(reqCtx.ctx, retryResponse, reqCtx.messages, reqCtx.availableTools)
	}

	// Handle empty retry response.
	if retryResponse == nil || retryResponse.Content == "" {
		return aiResponseMsg{content: response.Content, usage: response.Usage}
	}

	// If still no tool use after retry, combine both responses.
	combinedContent := response.Content
	if retryResponse.Content != "" {
		if combinedContent != "" {
			combinedContent += doubleNewline
		}
		combinedContent += retryResponse.Content
	}
	return aiResponseMsg{content: combinedContent, usage: combineUsage(response.Usage, retryResponse.Usage)}
}

// handleToolsResponse handles the AI response path when tools are available.
func (m *ChatModel) handleToolsResponse(ctx context.Context, messages []aiTypes.Message, availableTools []tools.Tool) tea.Msg {
	reqCtx := &aiRequestCtx{
		ctx:            ctx,
		messages:       messages,
		availableTools: availableTools,
		systemPrompt:   m.buildSystemPrompt(),
		atmosMemory:    m.getAtmosMemory(),
	}

	// Send messages with system prompt and tools (enables caching).
	response, err := m.sendAIRequest(ctx, reqCtx.systemPrompt, reqCtx.atmosMemory, messages, availableTools)
	if err != nil {
		return aiErrorMsg(formatAPIError(err))
	}

	// Handle empty initial response.
	if response == nil {
		return aiErrorMsg("Received nil response from AI provider")
	}

	// Check if AI wants to use tools.
	if response.StopReason == aiTypes.StopReasonToolUse && len(response.ToolCalls) > 0 {
		return m.handleToolExecutionFlow(ctx, response, messages, availableTools)
	}

	// No tool use - check if AI expressed intent to take action but did not use tools.
	if detectActionIntent(response.Content) {
		return m.handleActionIntentRetry(reqCtx, response)
	}

	// No action intent detected, return the text response.
	return aiResponseMsg{content: response.Content, usage: response.Usage}
}

func (m *ChatModel) getAIResponseWithContext(userMessage string, ctx context.Context) tea.Cmd {
	return func() tea.Msg {
		// Check if context is already cancelled before starting.
		if ctx.Err() != nil {
			return aiErrorMsg("Request cancelled")
		}

		// Build message history filtered by provider.
		messages := m.buildFilteredMessages()

		// Apply sliding window to limit conversation history if configured.
		messages = m.applyHistoryLimits(messages)

		// Add current user message.
		messages = append(messages, aiTypes.Message{
			Role:    aiTypes.RoleUser,
			Content: userMessage,
		})

		// Apply instructions context if available by prepending a system message.
		messages = m.prependMemoryContext(messages)

		// Provider-side session continuation (types.NativeSessionClient, opencode v2+):
		// send only the new turn and let the provider own the context instead of
		// re-serializing the full history every turn. CLI providers never return
		// Atmos-native tool calls (they run their own tools), so this path replaces
		// both the tools and no-tools branches for them. Falls back to the branches
		// below whenever the capability is unavailable or a native turn fails.
		if msg, handled := m.tryNativeSessionResponse(ctx, userMessage, messages); handled {
			return msg
		}

		// Check if tools are available.
		var availableTools []tools.Tool
		if m.executor != nil {
			availableTools = m.executor.ListTools()
		}

		// Use tool calling if tools are available.
		if len(availableTools) > 0 {
			return m.handleToolsResponse(ctx, messages, availableTools)
		}

		// Fallback to message with history but no tools.
		return m.handleNoToolsResponse(ctx, messages)
	}
}

// nativeSessionMetadataKey returns the Atmos session-metadata key that holds the
// provider-side session ID, namespaced by provider (e.g. "native_session:opencode")
// so switching providers mid-chat never resumes another provider's session.
func nativeSessionMetadataKey(provider string) string {
	return "native_session:" + provider
}

// nativeContinueOutcome classifies the outcome of attempting to continue a
// provider-side session, driving whether the caller restarts it, falls back to the
// concatenated-history path, or stops trying native sessions for this chat run.
type nativeContinueOutcome int

const (
	nativeContinueNotAvailable nativeContinueOutcome = iota // No stored session ID to continue.
	nativeContinueSucceeded                                 // The turn was answered by ContinueNativeSession.
	nativeContinueRestart                                   // Continuation failed; re-open the session with full context.
	nativeContinueDisabled                                  // The capability is off; stop trying this run.
)

// Drives a provider-side session when the client implements
// types.NativeSessionClient (opencode v2+). The first turn opens the session with the
// same prompt shaping the legacy path applies (system prompt, Atmos memory, and — for
// a resumed Atmos session or after instructions changed — prior history); later turns
// send only the new user message via ContinueNativeSession. Handled is false when
// native continuation is unavailable or failed, letting the caller fall back to the
// concatenated-history path for that turn.
func (m *ChatModel) tryNativeSessionResponse(ctx context.Context, userMessage string, messages []aiTypes.Message) (tea.Msg, bool) {
	if m.nativeSessionDisabled {
		return nil, false
	}
	nc, ok := m.client.(aiTypes.NativeSessionClient)
	if !ok {
		return nil, false
	}

	switch msg, outcome := m.tryContinueNativeSession(ctx, nc, userMessage); outcome {
	case nativeContinueSucceeded:
		return msg, true
	case nativeContinueDisabled:
		return nil, false
	case nativeContinueNotAvailable, nativeContinueRestart:
		// No session to continue (or a stale one): open it below with full context.
	}

	return m.startNativeSessionTurn(ctx, nc, messages)
}

// resolveNativeSessionID returns the metadata key and stored provider-side session ID
// for the given capability, preferring this chat run's in-memory value and falling
// back to the Atmos session metadata (chats resumed across Atmos restarts).
func (m *ChatModel) resolveNativeSessionID(nc aiTypes.NativeSessionClient) (string, string) {
	key := nativeSessionMetadataKey(nc.NativeSessionProvider())
	if m.nativeSessionKey == key {
		return key, m.nativeSessionID
	}
	if m.sess == nil || m.sess.Metadata == nil {
		return key, ""
	}
	if v, ok := m.sess.Metadata[key].(string); ok {
		return key, v
	}
	return key, ""
}

// tryContinueNativeSession attempts the continuation turn for a stored provider
// session ID, classifying failures so the caller can restart or fall back.
func (m *ChatModel) tryContinueNativeSession(ctx context.Context, nc aiTypes.NativeSessionClient, userMessage string) (tea.Msg, nativeContinueOutcome) {
	_, storedID := m.resolveNativeSessionID(nc)
	if storedID == "" {
		return nil, nativeContinueNotAvailable
	}

	// Instructions that drifted since the session opened (skill switch, ATMOS.md
	// edit) must reach the model; the provider session holds the old ones, so
	// restart it with the full current context.
	if systemPrompt, memory := m.buildSystemPrompt(), m.getAtmosMemory(); systemPrompt != m.nativeSessionSystemPrompt || memory != m.nativeSessionAtmosMemory {
		return nil, nativeContinueRestart
	}

	m.sendTurnStepStarted(turnStepKindAICall, aiCallStepLabel)
	reply, err := nc.ContinueNativeSession(ctx, storedID, userMessage)
	m.sendTurnStepFinished(err)
	switch {
	case err == nil:
		return aiResponseMsg{content: reply, usage: nil}, nativeContinueSucceeded
	case errors.Is(err, errUtils.ErrCLIProviderNativeSessionsOff):
		m.nativeSessionDisabled = true
		return nil, nativeContinueDisabled
	default:
		log.Debugf("Continuing native %s session failed; restarting it with full context: %v", nc.NativeSessionProvider(), err)
		return nil, nativeContinueRestart
	}
}

// startNativeSessionTurn opens (or re-opens) the provider-side session with the full
// current context and remembers its ID. Returns handled=false on failure so the
// caller falls back to the concatenated-history path for this turn.
func (m *ChatModel) startNativeSessionTurn(ctx context.Context, nc aiTypes.NativeSessionClient, messages []aiTypes.Message) (tea.Msg, bool) {
	title := ""
	if m.sess != nil {
		title = m.sess.Name
	}
	systemPrompt, atmosMemory := m.buildSystemPrompt(), m.getAtmosMemory()

	m.sendTurnStepStarted(turnStepKindAICall, aiCallStepLabel)
	reply, sessionID, err := nc.StartNativeSession(ctx, systemPrompt, atmosMemory, messages, title)
	m.sendTurnStepFinished(err)
	if err != nil {
		if errors.Is(err, errUtils.ErrCLIProviderNativeSessionsOff) {
			m.nativeSessionDisabled = true
		}
		log.Debugf("Starting native %s session failed; falling back to concatenated history: %v", nc.NativeSessionProvider(), err)
		return nil, false
	}

	m.rememberNativeSession(ctx, nativeSessionMetadataKey(nc.NativeSessionProvider()), sessionID, systemPrompt, atmosMemory)
	return aiResponseMsg{content: reply, usage: nil}, true
}

// rememberNativeSession stores the provider-side session ID both in memory (for
// chats without Atmos session persistence) and in the Atmos session's metadata (so
// a chat resumed across Atmos restarts reconnects to the same provider session),
// alongside the opening-turn instructions for later drift detection.
func (m *ChatModel) rememberNativeSession(ctx context.Context, key, sessionID, systemPrompt, atmosMemory string) {
	m.nativeSessionKey = key
	m.nativeSessionID = sessionID
	m.nativeSessionSystemPrompt = systemPrompt
	m.nativeSessionAtmosMemory = atmosMemory

	if m.sess == nil || m.manager == nil {
		return
	}
	if m.sess.Metadata == nil {
		m.sess.Metadata = make(map[string]interface{})
	}
	m.sess.Metadata[key] = sessionID
	if err := m.manager.UpdateSession(ctx, m.sess); err != nil {
		log.Debugf("Failed to persist native session ID in Atmos session metadata: %v", err)
	}
}

// aiCallStepLabel is the turn-step label shown while waiting on the AI provider's response.
const aiCallStepLabel = "Waiting for AI response..."

// sendTurnStepStarted records a new turn step (AI call or tool execution) beginning.
func (m *ChatModel) sendTurnStepStarted(kind turnStepKind, label string) {
	if m.program != nil {
		m.program.Send(turnStepStartedMsg{kind: kind, label: label, status: turnStepRunning, startedAt: time.Now()})
	}
}

// sendTurnStepFinished marks the most recently started turn step complete.
func (m *ChatModel) sendTurnStepFinished(err error) {
	if m.program != nil {
		m.program.Send(turnStepFinishedMsg{err: err})
	}
}

// sendAIRequest wraps SendMessageWithSystemPromptAndTools with turn-step progress events.
func (m *ChatModel) sendAIRequest(ctx context.Context, systemPrompt, atmosMemory string, messages []aiTypes.Message, availableTools []tools.Tool) (*aiTypes.Response, error) {
	m.sendTurnStepStarted(turnStepKindAICall, aiCallStepLabel)
	resp, err := m.client.SendMessageWithSystemPromptAndTools(ctx, systemPrompt, atmosMemory, messages, availableTools)
	m.sendTurnStepFinished(err)
	return resp, err
}

// sendAIRequestNoTools wraps SendMessageWithHistory with the same turn-step events, for the
// path where no tools are registered.
func (m *ChatModel) sendAIRequestNoTools(ctx context.Context, messages []aiTypes.Message) (string, error) {
	m.sendTurnStepStarted(turnStepKindAICall, aiCallStepLabel)
	resp, err := m.client.SendMessageWithHistory(ctx, messages)
	m.sendTurnStepFinished(err)
	return resp, err
}

// executeToolCalls executes tool calls and returns the results.
func (m *ChatModel) executeToolCalls(ctx context.Context, toolCalls []aiTypes.ToolCall) []*tools.Result {
	results := make([]*tools.Result, len(toolCalls))

	for i, toolCall := range toolCalls {
		log.Debugf("Executing tool: %s with params: %v", toolCall.Name, toolCall.Input)
		m.sendTurnStepStarted(turnStepKindTool, formatToolStepLabel(toolCall))

		// Execute the tool.
		result, err := m.executor.Execute(ctx, toolCall.Name, toolCall.Input)
		if err != nil {
			results[i] = &tools.Result{
				Success: false,
				Output:  fmt.Sprintf("Error: %v", err),
				Error:   err,
			}
			m.sendTurnStepFinished(err)
			continue
		}
		results[i] = result
		m.sendTurnStepFinished(result.Error)
	}

	return results
}

// handleToolExecutionFlow executes tools, sends results back to AI, and returns the combined response.
func (m *ChatModel) handleToolExecutionFlow(ctx context.Context, response *aiTypes.Response, messages []aiTypes.Message, availableTools []tools.Tool) tea.Msg {
	return m.handleToolExecutionFlowWithAccumulator(ctx, response, messages, availableTools, "")
}

// buildToolDisplayText builds the display output showing tool execution results for the user.
func buildToolDisplayText(response *aiTypes.Response, toolResults []*tools.Result) string {
	var resultText string
	if response.Content != "" {
		resultText = response.Content + doubleNewline
	}

	for i, result := range toolResults {
		if i > 0 {
			resultText += doubleNewline
		}

		displayOutput := resolveToolOutput(result)

		// Build tool header with name and parameters.
		toolHeader := fmt.Sprintf("**Tool:** `%s`", response.ToolCalls[i].Name)

		// Show the actual command/parameters being executed for better visibility.
		if toolParams := formatToolParameters(response.ToolCalls[i]); toolParams != "" {
			toolHeader += newlineChar + toolParams
		}

		// Detect output format and wrap in appropriate code block for syntax highlighting.
		format := detectOutputFormat(displayOutput)
		resultText += fmt.Sprintf("%s\n\n```%s\n%s\n```", toolHeader, format, displayOutput)
	}

	return resultText
}

// buildToolResultsContent builds the tool results content string to send back to the AI.
func buildToolResultsContent(response *aiTypes.Response, toolResults []*tools.Result) string {
	var toolResultsContent string
	for i, result := range toolResults {
		if i > 0 {
			toolResultsContent += doubleNewline
		}

		toolOutput := resolveToolOutput(result)
		toolResultsContent += fmt.Sprintf("Tool: %s\nResult:\n%s", response.ToolCalls[i].Name, toolOutput)
	}
	return toolResultsContent
}

// resolveToolOutput determines the output string from a tool result.
func resolveToolOutput(result *tools.Result) string {
	displayOutput := result.Output
	if displayOutput == "" && result.Error != nil {
		displayOutput = fmt.Sprintf("Error: %v", result.Error)
	}
	if displayOutput == "" {
		displayOutput = "No output returned"
	}
	return displayOutput
}

// appendAccumulated prepends accumulated content with a separator if non-empty.
func appendAccumulated(accumulated, newContent string) string {
	if accumulated != "" {
		return accumulated + doubleNewline + newContent
	}
	return newContent
}

// handleEmptyFollowUpResponse handles when the AI's follow-up response is empty or nil.
func handleEmptyFollowUpResponse(accumulated, resultText string, response *aiTypes.Response) tea.Msg {
	combinedResponse := appendAccumulated(accumulated, resultText+markdownSeparator+"*Note: AI response was empty. This might indicate rate limiting or a timeout.*")
	return aiResponseMsg{content: combinedResponse, usage: response.Usage}
}

// executeToolsAndBuildMessages runs tools, appends results to conversation, and returns display text.
func (m *ChatModel) executeToolsAndBuildMessages(ctx context.Context, response *aiTypes.Response, messages *[]aiTypes.Message) string {
	toolResults := m.executeToolCalls(ctx, response.ToolCalls)
	resultText := buildToolDisplayText(response, toolResults)
	toolResultsContent := buildToolResultsContent(response, toolResults)

	if response.Content != "" {
		*messages = append(*messages, aiTypes.Message{Role: aiTypes.RoleAssistant, Content: response.Content})
	}
	*messages = append(*messages, aiTypes.Message{
		Role:    aiTypes.RoleUser,
		Content: fmt.Sprintf("Tool execution results:\n\n%s\n\nPlease provide your final response based on these results.", toolResultsContent),
	})

	return resultText
}

// handleToolExecutionFlowWithAccumulator executes tools, sends results back to AI, and returns the combined response.
// The accumulatedContent parameter preserves intermediate AI thinking across recursive tool calls.
func (m *ChatModel) handleToolExecutionFlowWithAccumulator(ctx context.Context, response *aiTypes.Response, messages []aiTypes.Message, availableTools []tools.Tool, accumulatedContent string) tea.Msg {
	resultText := m.executeToolsAndBuildMessages(ctx, response, &messages)

	// Get system prompt and ATMOS.md for caching.
	systemPrompt := m.buildSystemPrompt()
	atmosMemory := m.getAtmosMemory()

	// Call AI again with tool results to get final response (with caching).
	finalResponse, err := m.sendAIRequest(ctx, systemPrompt, atmosMemory, messages, availableTools)
	if err != nil {
		return aiErrorMsg(formatAPIError(err))
	}

	// Handle empty or truncated response from AI.
	if finalResponse == nil || (finalResponse.Content == "" && finalResponse.StopReason != aiTypes.StopReasonToolUse) {
		return handleEmptyFollowUpResponse(accumulatedContent, resultText, finalResponse)
	}

	// Check if the final response wants to use more tools.
	if finalResponse.StopReason == aiTypes.StopReasonToolUse && len(finalResponse.ToolCalls) > 0 {
		newAccumulated := appendAccumulated(accumulatedContent, resultText)
		return m.handleToolExecutionFlowWithAccumulator(ctx, finalResponse, messages, availableTools, newAccumulated)
	}

	// Check if AI expressed intent to take more action in the final response.
	if detectActionIntent(finalResponse.Content) {
		reqCtx := &aiRequestCtx{
			ctx:                ctx,
			messages:           messages,
			availableTools:     availableTools,
			systemPrompt:       systemPrompt,
			atmosMemory:        atmosMemory,
			accumulatedContent: accumulatedContent,
			resultText:         resultText,
		}
		return m.handleFollowUpActionIntent(reqCtx, finalResponse)
	}

	// Combine accumulated content + tool execution display + final AI response.
	combinedResponse := appendAccumulated(accumulatedContent, resultText+markdownSeparator+finalResponse.Content)

	// Return the combined response (accumulated + tool results + AI's final analysis).
	return aiResponseMsg{content: combinedResponse, usage: finalResponse.Usage}
}

// handleFollowUpActionIntent handles when the AI expresses intent to take action in a follow-up response.
func (m *ChatModel) handleFollowUpActionIntent(reqCtx *aiRequestCtx, finalResponse *aiTypes.Response) tea.Msg {
	// AI said it would do something else but did not use tools. Prompt it again.
	reqCtx.messages = append(reqCtx.messages, aiTypes.Message{
		Role:    aiTypes.RoleAssistant,
		Content: finalResponse.Content,
	})
	reqCtx.messages = append(reqCtx.messages, aiTypes.Message{
		Role:    aiTypes.RoleUser,
		Content: "Please use the available tools to perform that action now, rather than just describing what you would do.",
	})

	// Retry with the prompt (with caching).
	retryResponse, err := m.sendAIRequest(reqCtx.ctx, reqCtx.systemPrompt, reqCtx.atmosMemory, reqCtx.messages, reqCtx.availableTools)
	if err != nil {
		return aiErrorMsg(formatAPIError(err))
	}

	// Handle empty retry response.
	if retryResponse == nil || (retryResponse.Content == "" && retryResponse.StopReason != aiTypes.StopReasonToolUse) {
		combinedResponse := appendAccumulated(reqCtx.accumulatedContent, reqCtx.resultText+markdownSeparator+finalResponse.Content+doubleNewline+"*Note: AI retry response was empty.*")
		return aiResponseMsg{content: combinedResponse, usage: combineUsage(finalResponse.Usage, retryResponse.Usage)}
	}

	// Check if AI now uses tools.
	if retryResponse.StopReason == aiTypes.StopReasonToolUse && len(retryResponse.ToolCalls) > 0 {
		newAccumulated := appendAccumulated(reqCtx.accumulatedContent, reqCtx.resultText)
		return m.handleToolExecutionFlowWithAccumulator(reqCtx.ctx, retryResponse, reqCtx.messages, reqCtx.availableTools, newAccumulated)
	}

	// If still no tool use, combine all responses.
	combinedResponse := appendAccumulated(reqCtx.accumulatedContent, reqCtx.resultText+markdownSeparator+finalResponse.Content+doubleNewline+retryResponse.Content)
	return aiResponseMsg{content: combinedResponse, usage: combineUsage(finalResponse.Usage, retryResponse.Usage)}
}
