use crate::tools::ToolRegistry;
use crate::types::{ChatRequest, ChatResponse, Message};
use anyhow::{Context, Result};
use reqwest::Client;

/// Configuration for the agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub url: String,
    pub api_key: String,
    /// Maximum number of iterations the agent loop will execute to prevent runaway loops.
    pub max_iterations: usize,
}

/// The core agent that manages conversation history and tool execution.
pub struct Agent {
    pub config: AgentConfig,
    pub history: Vec<Message>,
    client: Client,
}

impl Agent {
    /// Create a new agent with the given configuration and an optional system prompt.
    pub fn new(config: AgentConfig, system_prompt: Option<&str>) -> Self {
        let mut history = Vec::new();
        if let Some(prompt) = system_prompt {
            history.push(Message::system(prompt));
        }
        Self {
            config,
            history,
            client: Client::new(),
        }
    }

    /// Add a user message and run the agent loop until the assistant produces a
    /// final response with no further tool calls (or we hit the iteration limit).
    ///
    /// Returns the assistant's final text content (if any).
    pub async fn chat(
        &mut self,
        user_message: &str,
        tools: &ToolRegistry,
    ) -> Result<Option<String>> {
        self.history.push(Message::user(user_message));
        self.run_agent_loop(tools).await
    }

    /// Execute the agent loop: call the API, process tool calls, repeat.
    async fn run_agent_loop(&mut self, tools: &ToolRegistry) -> Result<Option<String>> {
        let tool_defs = if tools.definitions().is_empty() {
            None
        } else {
            Some(tools.definitions().to_vec())
        };

        for _iteration in 0..self.config.max_iterations {
            let request = ChatRequest {
                model: self.config.model.clone(),
                messages: self.history.clone(),
                tools: tool_defs.clone(),
            };

            let response = self.call_api(&request).await?;

            let choice = response
                .choices
                .into_iter()
                .next()
                .context("API returned no choices")?;

            // Append the assistant's message to history.
            let assistant_msg = Message::from_assistant(&choice.message);
            self.history.push(assistant_msg);

            // Check if the assistant made any tool calls.
            match &choice.message.tool_calls {
                Some(tool_calls) if !tool_calls.is_empty() => {
                    // Execute each tool call and append the results.
                    for tc in tool_calls {
                        let result = match tools.execute(&tc.function.name, &tc.function.arguments)
                        {
                            Ok(output) => output,
                            Err(e) => format!("Error: {e}"),
                        };
                        self.history.push(Message::tool_result(&tc.id, result));
                    }
                    // Continue the loop — the model needs to see the tool results.
                }
                _ => {
                    // No tool calls — the assistant has produced a final answer.
                    return Ok(choice.message.content);
                }
            }
        }

        // If we exhausted iterations, return whatever content the last assistant
        // message had.
        Ok(self
            .history
            .iter()
            .rev()
            .find(|m| m.role == crate::types::Role::Assistant)
            .and_then(|m| m.content.clone()))
    }

    /// Send a request to the chat completions API and parse the typed response.
    async fn call_api(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let http_response = self
            .client
            .post(&self.config.url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(request)
            .send()
            .await
            .context("Failed to send request to API")?;

        let status = http_response.status();
        if !status.is_success() {
            let body = http_response
                .text()
                .await
                .unwrap_or_else(|_| "unable to read body".to_string());
            anyhow::bail!("API returned HTTP {status}: {body}");
        }

        let chat_response: ChatResponse = http_response
            .json()
            .await
            .context("Failed to parse API response")?;

        Ok(chat_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Role;

    #[test]
    fn test_agent_new_without_system_prompt() {
        let config = AgentConfig {
            model: "test-model".to_string(),
            url: "http://localhost".to_string(),
            api_key: "test-key".to_string(),
            max_iterations: 10,
        };
        let agent = Agent::new(config, None);
        assert!(agent.history.is_empty());
    }

    #[test]
    fn test_agent_new_with_system_prompt() {
        let config = AgentConfig {
            model: "test-model".to_string(),
            url: "http://localhost".to_string(),
            api_key: "test-key".to_string(),
            max_iterations: 10,
        };
        let agent = Agent::new(config, Some("You are a helpful assistant."));
        assert_eq!(agent.history.len(), 1);
        assert_eq!(agent.history[0].role, Role::System);
        assert_eq!(
            agent.history[0].content.as_deref(),
            Some("You are a helpful assistant.")
        );
    }

    #[test]
    fn test_agent_config_clone() {
        let config = AgentConfig {
            model: "gpt-4".to_string(),
            url: "https://api.example.com/v1/chat/completions".to_string(),
            api_key: "sk-test".to_string(),
            max_iterations: 5,
        };
        let cloned = config.clone();
        assert_eq!(cloned.model, "gpt-4");
        assert_eq!(cloned.max_iterations, 5);
    }
}
