use super::Context;
use helix_view::{
    editor::AgentStatus,
    document::Mode,
};

/// Submit the current agent prompt input.
pub fn agent_submit(cx: &mut Context) {
    // Take the prompt input from the AgentSession (updated by AgentPanel)
    let input = std::mem::take(&mut *cx.editor.agent_session.prompt_input.write().unwrap());
    let input = input.trim().to_string();
    if input.is_empty() {
        return;
    }

    // Signal the prompt to clear itself
    cx.editor
        .agent_session
        .prompt_clear_requested
        .store(true, std::sync::atomic::Ordering::Relaxed);

    // Reset assistant state
    *cx.editor.agent_session.assistant_text.write().unwrap() = None;
    *cx.editor.agent_session.assistant_thinking.write().unwrap() = None;
    cx.editor.agent_session.active_tools.write().unwrap().clear();
    *cx.editor.agent_session.status.write().unwrap() = AgentStatus::Thinking;
    // Reset stop flag for new submission
    cx.editor.agent_session.stop_requested.store(false, std::sync::atomic::Ordering::Relaxed);

    // Create the RunContext bridge
    let helix_ctx = crate::agent_context::HelixRunContext::new(
        cx.editor.agent_session.assistant_text.clone(),
        cx.editor.agent_session.assistant_thinking.clone(),
        cx.editor.agent_session.active_tools.clone(),
        cx.editor.agent_session.status.clone(),
        cx.editor.agent_session.messages.clone(),
        cx.editor.agent_session.steering_queue.clone(),
        cx.editor.agent_session.stop_requested.clone(),
        cx.editor.agent_session.document_updates.clone(),
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")),
    );

    // Get config values
    let config = cx.editor.config();
    let base_url = config.agent.base_url.clone().unwrap_or_else(|| "http://127.0.0.1:8001".to_string());
    let api_key = config.agent.api_key.clone().unwrap_or_else(|| "test-key".to_string());
    let model_id = config.agent.model.clone().unwrap_or_else(|| "test-model".to_string());
    let provider_name = config.agent.provider.clone().unwrap_or_else(|| "openai".to_string());
    let system_prompt = config.agent.system_prompt.clone().unwrap_or_else(|| "You are a helpful coding assistant.".to_string());
    let thinking_level = config.agent.thinking_level;

    // Initialize provider if not already done
    let provider = match &cx.editor.agent_session.provider {
        Some(p) => p.clone(),
        None => {
            let provider = helix_agent::providers::openai::OpenAIProvider::new(
                base_url.clone(),
                api_key.clone(),
            );
            let provider = std::sync::Arc::new(provider);
            cx.editor.agent_session.provider = Some(provider.clone());
            provider
        }
    };

    // Initialize tools if not already done
    let tools = match &cx.editor.agent_session.tools {
        Some(t) => t.clone(),
        None => {
            let tool_config = helix_agent::tools::IntegratedToolConfig::default();
            let tools = std::sync::Arc::new(helix_agent::tools::IntegratedTools::new(tool_config));
            cx.editor.agent_session.tools = Some(tools.clone());
            tools
        }
    };

    // Initialize model if not already done
    let model = match &cx.editor.agent_session.model {
        Some(m) => m.clone(),
        None => {
            let model = std::sync::Arc::new(helix_agent::model::Model {
                id: model_id.clone(),
                name: model_id.clone(),
                api: provider_name.clone(),
                provider: provider_name.clone(),
                base_url: Some(base_url.clone()),
                reasoning: false,
                input: vec![],
                cost: Default::default(),
                context_window: 128000,
                max_tokens: 8192,
            });
            cx.editor.agent_session.model = Some(model.clone());
            model
        }
    };

    // Initialize agent if not already done
    let agent = match &cx.editor.agent_session.agent {
        Some(a) => a.clone(),
        None => {
            let agent_config = helix_agent::agent::AgentConfig {
                thinking_level,
            };
            // Start with empty history; messages will be added by RunContext
            let agent = std::sync::Arc::new(helix_agent::agent::Agent::new(
                agent_config,
                system_prompt,
                model.clone(),
                Vec::new(),
                provider.clone(),
                tools.clone(),
            ));
            cx.editor.agent_session.agent = Some(agent.clone());
            agent
        }
    };

    // Start the agent process - input is added as the first user message
    let process = agent.start(input, helix_ctx);
    cx.editor.agent_session.process = Some(process);

    // Request redraw
    cx.editor.needs_redraw = true;
}

/// Open the agent panel, entering Agent mode.
pub fn agent_toggle(cx: &mut Context) {
    if cx.editor.agent_session.panel_visible {
        // Panel is visible - close it
        agent_close(cx);
    } else {
        // Enter agent mode and open the panel
        cx.editor.enter_mode(Mode::Agent);
        cx.editor.agent_session.panel_visible = true;

        // Push the AgentPanel (output + prompt) onto the compositor as an overlay
        cx.push_layer(Box::new(crate::ui::agent_panel::agent_panel()));

        // Force redraw to update the statusline mode display
        cx.editor.needs_redraw = true;
    }
}

/// Hide the agent panel without stopping the agent process.
/// The agent continues running in the background.
pub fn agent_hide(cx: &mut Context) {
    // Mark panel as not visible
    cx.editor.agent_session.panel_visible = false;

    // Remove the agent panel by id from the compositor
    cx.callback.push(Box::new(
        |compositor: &mut crate::compositor::Compositor, _| {
            compositor.remove(crate::ui::agent_panel::AGENT_PANEL_ID);
        },
    ));

    // Exit agent mode - pops the previous mode from the stack
    cx.editor.exit_mode();
}

/// Show the agent panel if the agent session is active.
pub fn agent_show(cx: &mut Context) {
    if cx.editor.agent_session.panel_visible {
        return; // Already visible
    }

    // Check if agent is doing something (not idle)
    let status = *cx.editor.agent_session.status.read().unwrap();
    if status == AgentStatus::Idle && cx.editor.agent_session.process.is_none() {
        return; // Nothing to show
    }

    // Enter agent mode and open the panel
    cx.editor.enter_mode(Mode::Agent);
    cx.editor.agent_session.panel_visible = true;

    // Push the AgentPanel (output + prompt) onto the compositor as an overlay
    cx.push_layer(Box::new(crate::ui::agent_panel::agent_panel()));

    // Force redraw to update the statusline mode display
    cx.editor.needs_redraw = true;
}

/// Close the agent panel and stop the agent process.
pub fn agent_close(cx: &mut Context) {
    // Stop any running agent process
    if let Some(process) = cx.editor.agent_session.process.take() {
        process.abort();
    }
    *cx.editor.agent_session.status.write().unwrap() = AgentStatus::Idle;
    cx.editor
        .agent_session
        .stop_requested
        .store(true, std::sync::atomic::Ordering::Relaxed);

    // Mark panel as not visible
    cx.editor.agent_session.panel_visible = false;

    // Remove the agent panel by id from the compositor
    cx.callback.push(Box::new(
        |compositor: &mut crate::compositor::Compositor, _| {
            compositor.remove(crate::ui::agent_panel::AGENT_PANEL_ID);
        },
    ));

    // Exit agent mode - pops the previous mode from the stack
    cx.editor.exit_mode();
}

/// Stop the current agent execution.
pub fn agent_stop(cx: &mut Context) {
    cx.editor
        .agent_session
        .stop_requested
        .store(true, std::sync::atomic::Ordering::Relaxed);
    *cx.editor.agent_session.status.write().unwrap() = AgentStatus::Idle;

    if let Some(process) = cx.editor.agent_session.process.take() {
        process.abort();
    }
}

/// Clear the agent conversation history.
pub fn agent_clear(cx: &mut Context) {
    cx.editor.agent_session.messages.write().unwrap().clear();
    *cx.editor.agent_session.assistant_text.write().unwrap() = None;
    *cx.editor.agent_session.assistant_thinking.write().unwrap() = None;
    cx.editor.agent_session.active_tools.write().unwrap().clear();
    *cx.editor.agent_session.status.write().unwrap() = AgentStatus::Idle;

    if let Some(process) = cx.editor.agent_session.process.take() {
        process.abort();
    }
    cx.editor
        .agent_session
        .stop_requested
        .store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Insert a newline in the agent prompt (Shift+Enter or Ctrl+Enter).
pub fn agent_insert_newline(cx: &mut Context) {
    // Add newline to the prompt input
    let mut prompt = cx.editor.agent_session.prompt_input.write().unwrap();
    prompt.push('\n');
    // Trigger redraw so the newline is visible immediately
    helix_event::request_redraw();
    cx.editor.needs_redraw = true;
}

/// Enter insert mode from Agent mode (for editing the prompt).
pub fn agent_insert_mode(cx: &mut Context) {
    cx.editor.enter_mode(Mode::Insert);
}

/// Append mode from Agent mode.
pub fn agent_append_mode(cx: &mut Context) {
    cx.editor.enter_mode(Mode::Insert);
}

/// Insert at line end from Agent mode.
pub fn agent_insert_at_line_end(cx: &mut Context) {
    cx.editor.enter_mode(Mode::Insert);
}

/// Insert at line start from Agent mode.
pub fn agent_insert_at_line_start(cx: &mut Context) {
    cx.editor.enter_mode(Mode::Insert);
}
