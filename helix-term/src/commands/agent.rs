use super::Context;
use helix_view::{
    editor::AgentStatus,
    document::Mode,
};
use helix_core::RopeSlice;

/// Number of context lines to show before and after each cursor/selection.
const CURSOR_CONTEXT_LINES: usize = 4;

/// Submit the current agent prompt input.
pub fn agent_submit(cx: &mut Context) {
    // Take the prompt input from the AgentSession (updated by AgentPanel)
    let input = std::mem::take(&mut *cx.editor.agent_session.prompt_input.write().unwrap());
    let input = input.trim().to_string();
    if input.is_empty() {
        return;
    }

    // Take cursor context and prepend to input
    let cursor_context = std::mem::take(
        &mut *cx.editor.agent_session.cursor_context.write().unwrap(),
    );
    let full_input = if let Some(ref context) = cursor_context {
        format!("{}\n\n{}", context, input)
    } else {
        input
    };

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
    let process = agent.start(full_input, helix_ctx);
    cx.editor.agent_session.process = Some(process);

    // Request redraw
    cx.editor.needs_redraw = true;
}

/// Capture cursor/selection context and open the agent panel with a placeholder.
/// The full context is stored in session.cursor_context and attached on submit.
pub fn agent_ask_at_cursor(cx: &mut Context) {
    let (view, doc) = current_ref!(cx.editor);
    let selection = doc.selection(view.id);
    let text = doc.text().slice(..);

    // Get language name
    let language = doc
        .language_config()
        .map(|c| c.language_id.clone())
        .unwrap_or_else(|| "text".to_string());

    // Get filename
    let filename = std::path::Path::new(doc.display_name().as_ref())
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unnamed".to_string());

    // Build the full context string with cursor marks
    let context = build_cursor_context(selection, text, &language, &filename);

    // Store context for later attachment on submit
    cx.editor
        .agent_session
        .cursor_context
        .write()
        .unwrap()
        .replace(context);

    // Build compact placeholder for the prompt
    let num_selections = selection.len();
    let primary = selection.primary();
    let cursor_line = text.char_to_line(primary.from().min(text.len_chars())) + 1;
    let placeholder = if num_selections > 1 {
        format!("[context: {}:{} ({} selections)] ", filename, cursor_line, num_selections)
    } else {
        format!("[context: {}:{}] ", filename, cursor_line)
    };

    // If panel is already visible, just insert the placeholder into the prompt
    if cx.editor.agent_session.panel_visible {
        let mut prompt = cx.editor.agent_session.prompt_input.write().unwrap();
        if !prompt.is_empty() && !prompt.ends_with('\n') {
            prompt.push('\n');
        }
        prompt.push_str(&placeholder);
        helix_event::request_redraw();
        cx.editor.needs_redraw = true;
        return;
    }

    // Enter agent mode and open the panel
    cx.editor.enter_mode(Mode::Agent);
    cx.editor.agent_session.panel_visible = true;

    // Pre-fill the prompt with the placeholder
    cx.editor
        .agent_session
        .prompt_input
        .write()
        .unwrap()
        .push_str(&placeholder);

    // Push the AgentPanel onto the compositor
    cx.push_layer(Box::new(crate::ui::agent_panel::agent_panel()));

    // Force redraw
    cx.editor.needs_redraw = true;
}

/// Build cursor context string with |N>code<N| markup.
/// Shows context lines around each cursor/selection with skip indicators.
fn build_cursor_context(
    selection: &helix_core::Selection,
    text: RopeSlice<'_>,
    language: &str,
    filename: &str,
) -> String {
    let mut result = format!(
        "Here is the context around my cursor/selection in the file. \
        The markup `|N>text<N|` marks cursor N position (N is 0-indexed selection index). \
        `|0>x<0|` means cursor 0 is on character 'x'. \
        `|0>word<0|` means selection 0 covers 'word'. \
        Use these markers to understand where I am and what I selected.\n\n\
        ```{} {}\n",
        language, filename
    );

    let total_lines = text.len_lines();
    let ranges: Vec<_> = selection.ranges().to_vec();

    // Collect all lines that need to be shown (context around each cursor)
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();

    for (idx, range) in ranges.iter().enumerate() {
        let from_line = text.char_to_line(range.from().min(text.len_chars()));
        let to_line = text.char_to_line(range.to().saturating_sub(1).min(text.len_chars()));

        let block_start = from_line.saturating_sub(CURSOR_CONTEXT_LINES);
        let block_end = (to_line + CURSOR_CONTEXT_LINES + 1).min(total_lines);

        blocks.push((block_start, block_end, idx));
    }

    // Merge overlapping blocks and sort
    blocks.sort_by_key(|(start, _, _)| *start);
    let mut merged: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (start, end, idx) in blocks {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                if !last.2.contains(&idx) {
                    last.2.push(idx);
                }
                continue;
            }
        }
        merged.push((start, end, vec![idx]));
    }

    let mut prev_line = 0;
    for (block_start, block_end, cursor_indices) in &merged {
        // Show skip indicator if there are lines between blocks
        if *block_start > prev_line {
            let skipped = block_start - prev_line;
            result.push_str(&format!("...skip {} lines\n", skipped));
        }

        // Get the text for this block
        let block_start_char = text.line_to_char(*block_start);
        let block_end_char = if *block_end < total_lines {
            text.line_to_char(*block_end)
        } else {
            text.len_chars()
        };
        let block_text = text.slice(block_start_char..block_end_char);

        // For each line in the block, check if any cursor spans this line
        for (i, line) in block_text.lines().enumerate() {
            let current_line = block_start + i;

            // Check which cursors span this line
            let line_start_char = text.line_to_char(current_line);

            let mut marked_line = String::new();
            let mut char_pos = 0;
            // Trim trailing newline from line (RopeSlice::lines() includes it)
            let line_text: String = line.chars().take_while(|&c| c != '\n').collect();
            let line_chars: Vec<char> = line_text.chars().collect();

            for ch in &line_chars {
                let abs_char_pos = line_start_char + char_pos;

                // Check if any cursor marker should be inserted here
                for (cursor_idx, range) in ranges.iter().enumerate() {
                    if !cursor_indices.contains(&cursor_idx) {
                        continue;
                    }
                    // Insert opening marker at range start if on this line
                    if range.from() == abs_char_pos {
                        marked_line.push_str(&format!("|{}>", cursor_idx));
                    }
                    // Insert closing marker at range end if on this line
                    if range.to() == abs_char_pos {
                        marked_line.push_str(&format!("<{}|", cursor_idx));
                    }
                }

                marked_line.push(*ch);
                char_pos += 1;
            }

            // Handle zero-width selections (cursor at a position)
            for (cursor_idx, range) in ranges.iter().enumerate() {
                if !cursor_indices.contains(&cursor_idx) {
                    continue;
                }
                if range.from() == range.to() {
                    let cursor_line = text.char_to_line(range.from().min(text.len_chars()));
                    if cursor_line == current_line {
                        let pos_in_line = range.from() - line_start_char;
                        if pos_in_line < line_chars.len() {
                            let ch = line_chars[pos_in_line];
                            let before: String = line_chars[..pos_in_line].iter().collect();
                            let after: String = line_chars[pos_in_line + 1..].iter().collect();
                            marked_line = format!(
                                "{}|{}>{}<{}|{}",
                                before, cursor_idx, ch, cursor_idx, after
                            );
                        }
                    }
                }
            }

            result.push_str(&marked_line);
            result.push('\n');
        }

        prev_line = *block_end;
    }

    // Show remaining lines count
    if prev_line < total_lines {
        result.push_str(&format!("...{} lines more\n```", total_lines - prev_line));
    } else {
        result.push_str("```");
    }

    result
}
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
